"""Multi‑scale mask decoder.

The mask decoder receives SAM features and multi‑scale prompt features and
predicts segmentation masks at three resolutions.  In the original SAM‑LLaVA
paper the decoder extends the SAM decoder with parallel prediction branches
optimised by a weighted sum of dice and focal losses【409945085282613†L392-L424】.  This
simplified implementation produces high, medium and low resolution masks by
first computing a shared high resolution prediction and then downsampling to
derive the lower resolutions.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskDecoder(nn.Module):
    """Predict segmentation masks at multiple scales.

    Parameters
    ----------
    feature_dim : int
        Number of channels in the SAM feature map.
    prompt_dim : int
        Number of channels in the multi‑scale prompt.
    hidden_dim : int, default=128
        Number of channels in the intermediate representation used to
        produce the segmentation masks.
    """

    def __init__(
        self,
        feature_dim: int,
        prompt_dim: int,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        # Shared trunk that fuses SAM features and prompt features
        self.fuse = nn.Sequential(
            nn.Conv2d(feature_dim + prompt_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        # Final head for high resolution prediction
        self.head = nn.Conv2d(hidden_dim, 1, kernel_size=1)

    def forward(
        self,
        sam_features: torch.Tensor,
        multi_scale_prompt: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate high, medium and low resolution masks.

        Parameters
        ----------
        sam_features : torch.Tensor
            Feature map extracted by the SAM‑like encoder with shape
            (B, C_sam, H_sam, W_sam).  In our simplified pipeline H_sam and
            W_sam are one quarter of the input image resolution.
        multi_scale_prompt : torch.Tensor
            Multi‑scale prompt features of shape (B, C_prompt, H, W).  The
            spatial dimensions should match the input image resolution.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            The high, medium and low resolution masks with shapes (B,1,H,W),
            (B,1,H/2,W/2) and (B,1,H/4,W/4) respectively.
        """
        # Upsample SAM features to match the image resolution and concatenate
        B, C_sam, H_sam, W_sam = sam_features.shape
        H_img, W_img = multi_scale_prompt.shape[2], multi_scale_prompt.shape[3]
        sam_upsampled = F.interpolate(sam_features, size=(H_img, W_img), mode="bilinear", align_corners=False)
        fused = torch.cat([sam_upsampled, multi_scale_prompt], dim=1)
        fused = self.fuse(fused)
        high_res_logits = self.head(fused)  # (B,1,H,W)
        high_res_mask = torch.sigmoid(high_res_logits)
        # Derive medium and low resolution masks via average pooling
        mid_res_mask = F.avg_pool2d(high_res_mask, kernel_size=2, stride=2)
        low_res_mask = F.avg_pool2d(high_res_mask, kernel_size=4, stride=4)
        return high_res_mask, mid_res_mask, low_res_mask