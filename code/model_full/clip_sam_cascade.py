"""CLIP–SAM cascade for the full SAM‑LLaVA implementation.

This module implements a cascade that first computes a coarse segmentation
using the official CLIP ViT‑B/16 model and then refines it using the
Segment‑Anything Model (SAM) with a ViT‑H backbone.  Both models are
loaded via HuggingFace and the `segment_anything` library respectively.

The cascade follows the description in the SAM‑LLaVA paper: CLIP’s
image encoder produces a sequence of patch embeddings which are
semantically aligned with the text embeddings of a question/description.
These coarse features are projected to a low‑resolution segmentation map
by computing similarity between image and text embeddings and
upsampling.  The SAM model then takes the coarse mask as an input
prompt to produce a high‑resolution segmentation mask【409945085282613†L540-L546】.

Note that loading these models requires internet access on first use
unless you download the checkpoints manually.  See the README for
instructions on where to obtain the weights and how to place them so
that the code can find them without internet access.
"""

from __future__ import annotations

import os
from typing import Tuple

import torch
from torch import nn
from torchvision.transforms.functional import resize

try:
    from transformers import CLIPProcessor, CLIPModel
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "transformers must be installed to use the full SAM‑LLaVA model. "
        "Install with `pip install transformers`.")

try:
    from segment_anything import sam_model_registry, SamPredictor
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "segment_anything must be installed to use the full SAM‑LLaVA model. "
        "Install with `pip install segment_anything`.")


class ClipSamCascadeFull(nn.Module):
    """Cascade that generates coarse and refined segmentation masks.

    Parameters
    ----------
    clip_model_name : str
        HuggingFace model id for the CLIP model.  Defaults to
        ``"openai/clip-vit-base-patch16"`` as used in the paper.
    sam_checkpoint : str
        Path to the SAM ViT‑H checkpoint (``.pth``) file.  See the README
        for download instructions.  If ``None``, the predictor cannot be
        created until the checkpoint is provided at runtime.
    device : torch.device
        Device on which to run the models.  The CLIP model will be
        automatically placed on this device.  The SAM predictor must
        run on CPU or GPU depending on your environment.
    """

    def __init__(self, clip_model_name: str = "openai/clip-vit-base-patch16",
                 sam_checkpoint: str | None = None,
                 device: torch.device | str = "cpu") -> None:
        super().__init__()
        self.device = torch.device(device)
        # Load CLIP model and processor
        self.processor = CLIPProcessor.from_pretrained(clip_model_name)
        self.clip_model = CLIPModel.from_pretrained(clip_model_name)
        self.clip_model.to(self.device)
        self.clip_model.eval()
        # Load SAM predictor lazily because it may require GPU
        self.sam_checkpoint = sam_checkpoint
        self.sam_predictor: SamPredictor | None = None

    def _load_sam_predictor(self) -> None:
        """Load the SAM predictor if it has not been loaded already.

        Raises
        ------
        RuntimeError
            If no checkpoint path is provided.
        """
        if self.sam_predictor is not None:
            return
        if not self.sam_checkpoint:
            raise RuntimeError(
                "SAM checkpoint is not provided. Please set sam_checkpoint in the "
                "ClipSamCascadeFull constructor or supply a valid path to the checkpoint.")
        # The sam_model_registry can load different backbone variants.  We use
        # vit_h (ViT‑H) as specified in the paper【409945085282613†L540-L546】.
        sam = sam_model_registry["vit_h"](checkpoint=self.sam_checkpoint)
        sam.to(self.device)
        self.sam_predictor = SamPredictor(sam)

    @torch.no_grad()
    def forward(self, images: torch.Tensor, text_prompts: list[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute coarse and refined segmentation masks.

        Parameters
        ----------
        images : torch.Tensor
            A batch of images of shape ``(B, C, H, W)`` in ``[0, 1]`` range.
        text_prompts : list[str]
            A list of textual descriptions/questions corresponding to each image.

        Returns
        -------
        coarse_masks : torch.Tensor
            A tensor of shape ``(B, 1, h, w)`` containing low‑resolution coarse masks.
        refined_masks : torch.Tensor
            A tensor of shape ``(B, 1, H, W)`` containing high‑resolution masks
            predicted by SAM.
        """
        B, C, H, W = images.shape
        # Compute CLIP image and text embeddings
        # Convert PyTorch images to PIL compatible format using the processor
        inputs = self.processor(
            images=[img for img in images],
            text=text_prompts,
            return_tensors="pt",
            padding=True
        ).to(self.device)
        outputs = self.clip_model(**inputs)
        # CLIP returns pooled embeddings and per‑token embeddings.  We use
        # the last hidden state of the visual encoder (vision_model_output).
        vision_embeds = outputs.vision_model_output.last_hidden_state  # (B, num_patches+1, D)
        text_embeds = outputs.text_model_output.last_hidden_state  # (B, seq_len, D)
        # Remove the CLS token from image features
        vision_embeds = vision_embeds[:, 1:]  # (B, num_patches, D)
        # Compute similarity matrix between image patch features and text tokens
        # We normalise features to unit length for cosine similarity
        vision_norm = vision_embeds / vision_embeds.norm(dim=-1, keepdim=True)
        text_norm = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
        # Compute similarity (B, num_patches, seq_len)
        sim = torch.einsum("bpd,bsd->bps", vision_norm, text_norm)
        # Average over text tokens to get a single score per patch
        sim_scores = sim.mean(dim=-1)  # (B, num_patches)
        # Reshape to image grid.  For ViT‑B/16 on 224x224 images, num_patches=196=14*14.
        # We infer the spatial dimensions from the number of patches.
        p = int(sim_scores.shape[1] ** 0.5)
        coarse_masks = sim_scores.reshape(B, 1, p, p)
        # Upsample to original image size to prepare input for SAM
        coarse_up = nn.functional.interpolate(coarse_masks, size=(H, W), mode="bilinear", align_corners=False)
        # Ensure SAM predictor is loaded
        self._load_sam_predictor()
        refined_masks = []
        # SAM takes numpy images and masks.  Loop over batch.
        for i in range(B):
            img = images[i].detach().cpu().permute(1, 2, 0).numpy()
            # SAM expects uint8 images in [0,255] RGB order
            img_uint8 = (img * 255).astype("uint8")
            mask_input = coarse_up[i, 0].detach().cpu().numpy()
            # Set image in predictor
            assert self.sam_predictor is not None
            self.sam_predictor.set_image(img_uint8)
            # Provide mask_input and no point prompts; returning only masks
            pred_masks, _, _ = self.sam_predictor.predict(
                point_coords=None,
                point_labels=None,
                mask_input=mask_input[None, :, :],
                multimask_output=False,
            )
            mask = torch.from_numpy(pred_masks[0]).unsqueeze(0)  # (1, H, W)
            refined_masks.append(mask)
        refined_masks = torch.stack(refined_masks, dim=0).to(self.device)  # (B, 1, H, W)
        return coarse_masks, refined_masks
