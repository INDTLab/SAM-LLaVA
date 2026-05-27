"""Segmentation‑Aware Semantic Alignment (SASA) module.

The purpose of the SASA module is to tightly couple mask‑filtered visual
features with textual embeddings via bidirectional cross‑attention【409945085282613†L423-L455】.
By grounding both modalities in the fine segmentation mask, the model can
generate descriptions that are spatially aligned with the defect regions and
reduce hallucinations.  This implementation assumes that both visual and text
embeddings share the same dimensionality; projection layers are included to map
inputs to a common dimension.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SASAModule(nn.Module):
    """Bidirectional cross‑attention between visual and textual features.

    Parameters
    ----------
    vis_dim : int
        Dimensionality of the input visual features.
    text_dim : int
        Dimensionality of the input text embeddings.
    align_dim : int
        Dimensionality of the common alignment space in which cross
        attention is performed.
    num_heads : int, default=4
        Number of attention heads for the multi‑head attention.
    """

    def __init__(
        self,
        vis_dim: int,
        text_dim: int,
        align_dim: int,
        num_heads: int = 4,
    ) -> None:
        super().__init__()
        # Linear projections into alignment space
        self.proj_vis = nn.Linear(vis_dim, align_dim)
        self.proj_text = nn.Linear(text_dim, align_dim)
        # Bidirectional cross‑attention
        self.cross_attn_v2t = nn.MultiheadAttention(align_dim, num_heads, batch_first=True)
        self.cross_attn_t2v = nn.MultiheadAttention(align_dim, num_heads, batch_first=True)
        # Output projection back to original spaces
        self.out_vis = nn.Linear(align_dim, vis_dim)
        self.out_text = nn.Linear(align_dim, text_dim)

    def forward(
        self,
        visual_features: torch.Tensor,
        mask: torch.Tensor,
        text_embeddings: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Align visual and textual features via bidirectional attention.

        Parameters
        ----------
        visual_features : torch.Tensor
            Tensor of shape (B,C_v,H,W) containing SAM‑derived image features.
        mask : torch.Tensor
            Binary mask of shape (B,1,H,W) used to filter the visual features.
        text_embeddings : torch.Tensor
            Token embeddings of shape (B,L,C_t) from the language model’s
            embedding layer.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            The aligned visual features (B,H*W,C_v) and aligned text features
            (B,L,C_t).  These can be concatenated and fed into the language
            model.
        """
        B, C_v, H_v, W_v = visual_features.shape
        # Ensure mask has the same spatial resolution as visual features
        mask_resized = mask
        if mask.shape[2] != H_v or mask.shape[3] != W_v:
            # Use nearest neighbour to downsample/upsample the mask
            mask_resized = torch.nn.functional.interpolate(
                mask, size=(H_v, W_v), mode="nearest"
            )
        # Expand along the channel dimension and apply the mask
        mask_expanded = mask_resized.expand(-1, C_v, -1, -1)  # (B,C_v,H_v,W_v)
        masked_vis = visual_features * mask_expanded  # zero out background regions
        vis_seq = masked_vis.flatten(start_dim=2).transpose(1, 2)  # (B,H_v*W_v,C_v)
        # Project to alignment space
        V0 = self.proj_vis(vis_seq)  # (B,H*W,align_dim)
        T0 = self.proj_text(text_embeddings)  # (B,L,align_dim)
        # Cross attention: vision → text
        V_prime, _ = self.cross_attn_v2t(V0, T0, T0)
        # Cross attention: text → vision
        T_prime, _ = self.cross_attn_t2v(T0, V_prime, V_prime)
        # Project back to original dimensions
        V_out = self.out_vis(V_prime)  # (B,H*W,C_v)
        T_out = self.out_text(T_prime)  # (B,L,C_t)
        return V_out, T_out