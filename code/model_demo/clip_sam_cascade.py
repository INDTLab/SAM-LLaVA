"""CLIP–SAM cascade prompting.

This module implements the coarse‑to‑fine defect localisation mechanism described in
Section 3.1 of the SAM‑LLaVA paper【409945085282613†L324-L336】.  In the original
framework the authors first compute a coarse anomaly heatmap via the contrastive
similarity between CLIP image features and normal/abnormal text embeddings,
binarise it to obtain a coarse mask and then refine that mask using the Segment
Anything Model (SAM)【409945085282613†L330-L337】.  For simplicity we replace the
CLIP and SAM models with lightweight convolutional networks.  The interface
mirrors the intended behaviour: given an input image and an optional list of
support images/masks for few‑shot calibration, the module produces a fine
segmentation mask and image features for downstream processing.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ClipSamCascade(nn.Module):
    """Simplified CLIP–SAM cascade.

    Parameters
    ----------
    in_channels : int, default=3
        Number of channels in the input image.  3 for RGB.
    embed_dim : int, default=64
        Number of channels in the intermediate feature maps.  A larger
        embed_dim yields more expressive features but increases computation.
    """

    def __init__(self, in_channels: int = 3, embed_dim: int = 64) -> None:
        super().__init__()
        # CLIP‑like image encoder: a few downsampling conv layers
        self.clip_encoder = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
        )
        # SAM‑like image encoder for refined features
        self.sam_encoder = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
        )
        # Small convolution to refine the coarse mask into a fine mask
        self.mask_refiner = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        image: torch.Tensor,
        support: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute the fine segmentation mask and image features.

        Parameters
        ----------
        image : torch.Tensor
            Input image tensor of shape (B, 3, H, W) in the range [0,1].
        support : Optional[List[Tuple[torch.Tensor, torch.Tensor]]]
            Optional list of support images and masks used in few‑shot
            calibration.  Each element is a tuple `(support_image,
            support_mask)`.  In this simplified implementation the support
            samples are ignored, but the argument is retained for API
            compatibility.

        Returns
        -------
        fine_mask : torch.Tensor
            A binary mask of shape (B, 1, H, W) representing the refined
            segmentation.  Values are in [0,1].
        features : torch.Tensor
            Image features of shape (B, C, H/4, W/4) to be consumed by
            downstream modules such as the mask decoder and SASA module.
        """
        # Coarse anomaly heatmap: average of CLIP features across channels
        clip_feats = self.clip_encoder(image)  # (B, C, Hc, Wc)
        coarse_heatmap = clip_feats.mean(dim=1, keepdim=True)  # (B,1,Hc,Wc)
        # Threshold the heatmap to obtain a coarse mask.  A simple mean
        # threshold is used in place of Otsu’s method【409945085282613†L341-L346】.
        threshold = coarse_heatmap.mean(dim=[2, 3], keepdim=True)
        coarse_mask = (coarse_heatmap > threshold).float()
        # Upsample coarse mask back to input resolution
        coarse_mask_up = F.interpolate(coarse_mask, size=image.shape[2:], mode="bilinear", align_corners=False)
        # Refine the mask with a small conv net to obtain a smooth fine mask
        fine_mask = self.mask_refiner(coarse_mask_up)
        # Binarise the fine mask at 0.5 for downstream use
        fine_mask = (fine_mask > 0.5).float()
        # Extract SAM features
        features = self.sam_encoder(image)  # (B, C, H/4, W/4)
        return fine_mask, features

    def get_clip_features(self, image: torch.Tensor) -> torch.Tensor:
        """Return CLIP‑like features used for coarse heatmap computation.

        Parameters
        ----------
        image : torch.Tensor
            Input image tensor of shape (B, 3, H, W).

        Returns
        -------
        torch.Tensor
            Feature tensor of shape (B, C, H/4, W/4).
        """
        return self.clip_encoder(image)

    def get_sam_features(self, image: torch.Tensor) -> torch.Tensor:
        """Return SAM‑like features used for mask refinement.

        Parameters
        ----------
        image : torch.Tensor
            Input image tensor of shape (B, 3, H, W).

        Returns
        -------
        torch.Tensor
            Feature tensor of shape (B, C, H/4, W/4).
        """
        return self.sam_encoder(image)