"""Multi‑scale prompt learner.

The SAM‑LLaVA paper introduces a multi‑scale prompt learner and mask decoder to
handle defects at different sizes【409945085282613†L392-L405】.  The idea is to build
prompt features at several spatial scales from the coarse mask and inject them
into the downstream mask decoder.  In the original model this is achieved by
average pooling with various kernel sizes followed by transposed convolutions.
Here we implement a simplified version that pools the input mask at three
different scales, projects the pooled masks into an embedding space and upsamples
them back to the original resolution.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScalePromptLearner(nn.Module):
    """Generate multi‑scale prompts from a coarse/fine mask.

    Given a binary mask of shape (B,1,H,W), the learner produces a tensor
    `(B, 3*C, H, W)` by pooling the mask at three different scales, projecting
    each pooled representation through a 1×1 convolution and concatenating the
    results along the channel dimension【409945085282613†L392-L405】.
    """

    def __init__(self, embed_dim: int = 64) -> None:
        super().__init__()
        # Projection layers for the three scales
        self.proj_large = nn.Conv2d(1, embed_dim, kernel_size=1)
        self.proj_medium = nn.Conv2d(1, embed_dim, kernel_size=1)
        self.proj_small = nn.Conv2d(1, embed_dim, kernel_size=1)
        # Pooling kernels.  Kernel sizes roughly correspond to large,
        # medium and small context regions.  These can be adjusted to
        # match the receptive fields described in the paper【409945085282613†L392-L405】.
        self.pool_large = nn.AvgPool2d(kernel_size=8, stride=8, padding=0)
        self.pool_medium = nn.AvgPool2d(kernel_size=4, stride=4, padding=0)
        self.pool_small = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)

    def forward(self, mask: torch.Tensor) -> torch.Tensor:
        """Compute multi‑scale prompt features from a binary mask.

        Parameters
        ----------
        mask : torch.Tensor
            Input mask of shape (B,1,H,W) with values 0 or 1.

        Returns
        -------
        torch.Tensor
            Concatenated multi‑scale features of shape (B,3*embed_dim,H,W).
        """
        B, C, H, W = mask.shape
        # Pool to create large, medium and small context features
        # Use nearest upsampling to return to original resolution
        large = self.pool_large(mask)  # (B,1,H/8,W/8)
        large = F.interpolate(large, size=(H, W), mode="nearest")
        medium = self.pool_medium(mask)  # (B,1,H/4,W/4)
        medium = F.interpolate(medium, size=(H, W), mode="nearest")
        small = self.pool_small(mask)  # (B,1,H/2,W/2)
        small = F.interpolate(small, size=(H, W), mode="nearest")
        # Project each pooled mask into the embedding space
        large_feat = self.proj_large(large)
        medium_feat = self.proj_medium(medium)
        small_feat = self.proj_small(small)
        # Concatenate along the channel dimension
        multi_scale_prompt = torch.cat([large_feat, medium_feat, small_feat], dim=1)
        return multi_scale_prompt