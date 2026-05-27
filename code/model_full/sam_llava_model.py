"""High‑level SAM‑LLaVA full model using official CLIP, SAM and Vicuna models.

This class integrates the full CLIP–SAM cascade and the Vicuna‑7B with
LoRA to perform joint segmentation and captioning as described in the
SAM‑LLaVA paper【409945085282613†L540-L546】.  It accepts a batch of images and text
prompts and returns refined segmentation masks together with language
model logits for teacher forcing.

The model conditions the language model on visual context by projecting
the refined SAM mask into the Vicuna embedding space and adding this
vector to the embedding of the first token in each sequence.  This is
a simplified approximation of the segmentation‑aware semantic alignment
(SASA) mechanism in the paper but retains the core idea of fusing
visual and textual representations before generation.

Users are expected to supply the paths to the SAM checkpoint and to
ensure that the CLIP and Vicuna weights are available (either via
HuggingFace caching or manual download) as noted in the README.
"""

from __future__ import annotations

from typing import List, Tuple, Optional

import torch
import torch.nn as nn

from .clip_sam_cascade import ClipSamCascadeFull
from .llm import VicunaLLM


class SamLlavaModelFull(nn.Module):
    """Assemble the full SAM‑LLaVA model."""

    def __init__(
        self,
        clip_model_name: str = "openai/clip-vit-base-patch16",
        sam_checkpoint: Optional[str] = None,
        llm_model_name: str = "lmsys/vicuna-7b-v1.5",
        lora_r: int = 16,
        lora_alpha: int = 32,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__()
        self.device = torch.device(device)
        # Cascade of CLIP and SAM
        self.cascade = ClipSamCascadeFull(
            clip_model_name=clip_model_name,
            sam_checkpoint=sam_checkpoint,
            device=self.device,
        )
        # Vicuna LLM with LoRA
        self.llm = VicunaLLM(
            model_name=llm_model_name,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            device=self.device,
        )
        # Determine the hidden size of the LLM to create a context projector
        hidden_size = self.llm.model.config.hidden_size
        # Project flattened masks into the hidden size of Vicuna
        # We flatten H*W -> hidden_size.  The linear layer's input dim will
        # be assigned during the first forward pass based on image size.
        self.context_proj: Optional[nn.Linear] = None

    def _get_context_proj(self, mask: torch.Tensor) -> nn.Linear:
        """Lazy initialisation of the context projection layer.

        This layer maps a flattened segmentation mask of shape (H*W) to the
        Vicuna hidden dimension.  It is created on the first forward pass
        when the spatial dimensions of the mask are known.
        """
        B, C, H, W = mask.shape
        hidden_size = self.llm.model.config.hidden_size
        if self.context_proj is None:
            self.context_proj = nn.Linear(H * W, hidden_size)
        return self.context_proj

    def forward(
        self,
        images: torch.Tensor,
        descriptions: List[str],
        text_condition: Optional[List[str]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Perform a forward pass through the full model.

        Parameters
        ----------
        images : torch.Tensor
            Batch of images of shape (B,C,H,W) in [0,1] range.
        descriptions : List[str]
            Ground truth descriptions to compute language modelling loss
            via teacher forcing.
        text_condition : List[str], optional
            Optional textual queries/prompts used to guide the segmentation
            cascade.  If ``None``, the descriptions themselves are used.

        Returns
        -------
        refined_masks : torch.Tensor
            Refined segmentation masks of shape (B,1,H,W).
        logits : torch.Tensor
            Logits over vocabulary produced by Vicuna of shape (B,L,V).
        """
        B = images.size(0)
        device = self.device
        # Use text_condition if provided, otherwise the descriptions
        prompts = text_condition if text_condition is not None else descriptions
        # Compute coarse and refined masks via cascade
        _, refined_masks = self.cascade(images.to(device), prompts)
        # Flatten mask and project to Vicuna hidden size
        proj_layer = self._get_context_proj(refined_masks)
        # Flatten each mask: (B,1,H,W) -> (B,H*W)
        flat_mask = refined_masks.view(B, -1)
        context_vec = proj_layer(flat_mask)  # (B,hidden_size)
        # Tokenise descriptions
        input_ids = self.llm.encode(descriptions)  # (B,L)
        # Get input embeddings from Vicuna
        embedding_layer = self.llm.model.get_input_embeddings()
        inputs_embeds = embedding_layer(input_ids)  # (B,L,hidden_size)
        # Add context to the first token embedding
        inputs_embeds[:, 0, :] = inputs_embeds[:, 0, :] + context_vec
        # Forward through the language model.  We do not supply an attention
        # mask so the model will use default causal mask.
        logits = self.llm.model(inputs_embeds=inputs_embeds, return_dict=True).logits  # (B,L,V)
        return refined_masks, logits

    def generate(
        self,
        image: torch.Tensor,
        text_prompt: str = "Describe the defect in the image.",
        max_new_tokens: int = 50,
        **gen_kwargs,
    ) -> Tuple[torch.Tensor, List[str]]:
        """Generate a description conditioned on an input image.

        Parameters
        ----------
        image : torch.Tensor
            Single image tensor of shape (1,C,H,W).
        text_prompt : str
            Prompt used to guide the segmentation cascade.  The same prompt
            will be prepended to the generated description.
        max_new_tokens : int, optional
            Maximum number of tokens to generate.  Default is 50.
        gen_kwargs : dict, optional
            Additional keyword arguments passed to `VicunaLLM.generate`.

        Returns
        -------
        refined_mask : torch.Tensor
            Refined segmentation mask of shape (1,1,H,W).
        outputs : List[str]
            Generated text as a list with a single string.
        """
        self.eval()
        with torch.no_grad():
            # Compute refined mask
            _, refined_mask = self.cascade(image.to(self.device), [text_prompt])
            # Project mask to context vector
            proj_layer = self._get_context_proj(refined_mask)
            flat_mask = refined_mask.view(1, -1)
            context_vec = proj_layer(flat_mask)  # (1,hidden_size)
            # Prepend a BOS token to the prompt and tokenise
            prompt_with_bos = text_prompt
            input_ids = self.llm.encode([prompt_with_bos])  # (1,L)
            # Get embeddings and add context to the first token
            embedding_layer = self.llm.model.get_input_embeddings()
            inputs_embeds = embedding_layer(input_ids)
            inputs_embeds[:, 0, :] = inputs_embeds[:, 0, :] + context_vec
            # Generate continuation
            generated_ids = self.llm.model.generate(
                inputs_embeds=inputs_embeds,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.llm.tokenizer.pad_token_id,
                eos_token_id=self.llm.tokenizer.eos_token_id,
                **gen_kwargs,
            )
            # Decode
            texts = self.llm.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            return refined_mask, texts
