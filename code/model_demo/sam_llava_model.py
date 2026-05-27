"""High‑level SAM‑LLaVA model assembly.

This class ties together the individual components – the CLIP–SAM cascade,
multi‑scale prompt learner, mask decoder, SASA module and the language model –
into an end‑to‑end architecture.  During training the model simultaneously
predicts segmentation masks at multiple scales and generates a description of the
defect.  The segmentation loss and language modelling loss are computed
externally in the training script.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .clip_sam_cascade import ClipSamCascade
from .multi_scale_prompt import MultiScalePromptLearner
from .mask_decoder import MaskDecoder
from .sasa_module import SASAModule
from .llm import SimpleTokenizer, SimpleLLM


class SamLlavaModel(nn.Module):
    """Assemble SAM‑LLaVA components into a single PyTorch module."""

    def __init__(
        self,
        image_channels: int = 3,
        clip_embed_dim: int = 64,
        prompt_dim: int = 32,
        llm_embed_dim: int = 128,
        llm_hidden_dim: int = 256,
        llm_layers: int = 2,
        align_dim: int = 64,
        num_heads: int = 4,
        tokenizer: Optional[SimpleTokenizer] = None,
    ) -> None:
        super().__init__()
        # Components
        self.cascade = ClipSamCascade(in_channels=image_channels, embed_dim=clip_embed_dim)
        self.prompt_learner = MultiScalePromptLearner(embed_dim=prompt_dim)
        self.mask_decoder = MaskDecoder(feature_dim=clip_embed_dim, prompt_dim=prompt_dim * 3, hidden_dim=128)
        self.tokenizer = tokenizer or SimpleTokenizer()
        self.llm = SimpleLLM(
            vocab_size=self.tokenizer.vocab_size,
            embed_dim=llm_embed_dim,
            hidden_dim=llm_hidden_dim,
            num_layers=llm_layers,
        )
        self.sasa = SASAModule(
            vis_dim=clip_embed_dim,
            text_dim=llm_embed_dim,
            align_dim=align_dim,
            num_heads=num_heads,
        )
        # Project aggregated SASA context to the dimension expected by the LLM
        self.context_pool = nn.Linear(clip_embed_dim + llm_embed_dim, llm_hidden_dim)

    def forward(
        self,
        images: torch.Tensor,
        descriptions: List[str],
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor]:
        """Perform a forward pass through the model.

        Parameters
        ----------
        images : torch.Tensor
            Input images of shape (B,3,H,W).
        descriptions : List[str]
            Batch of target descriptions, used to compute the language modelling
            loss.  These strings are tokenised on the fly.

        Returns
        -------
        seg_masks : Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            The high, medium and low resolution masks predicted by the mask
            decoder.
        logits : torch.Tensor
            Logits over the vocabulary produced by the language model for
            teacher forcing.  Shape is (B,L,V) where L is the maximum
            sequence length in the batch.
        hidden_context : torch.Tensor
            Context vector of shape (B,hidden_dim) derived from the SASA
            module.  This can be used for inference.
        """
        B = images.shape[0]
        device = images.device
        # Cascade: obtain fine mask and SAM features
        fine_mask, sam_features = self.cascade(images)
        # Multi‑scale prompts
        prompt_features = self.prompt_learner(fine_mask)
        # Decode masks at multiple scales
        high_mask, mid_mask, low_mask = self.mask_decoder(sam_features, prompt_features)
        seg_masks = (high_mask, mid_mask, low_mask)
        # Tokenise descriptions
        token_seqs = [self.tokenizer.encode(desc) for desc in descriptions]
        # Pad sequences to the same length
        max_len = max(len(seq) for seq in token_seqs)
        padded_tokens = torch.full((B, max_len), self.tokenizer.pad_id, dtype=torch.long, device=device)
        for i, seq in enumerate(token_seqs):
            padded_tokens[i, : len(seq)] = torch.tensor(seq, device=device, dtype=torch.long)
        # Compute text embeddings (excluding final token for teacher forcing)
        # We feed the entire sequence including BOS and EOS tokens into the LLM.
        text_embeds = self.llm.embedding(padded_tokens)  # (B,L,embed_dim)
        # Align visual and textual features via SASA
        # Flatten SAM features for SASA: shape (B,C,H,W)
        aligned_vis, aligned_text = self.sasa(sam_features, fine_mask, text_embeds)
        # Aggregate aligned features to a single vector per sample
        vis_avg = aligned_vis.mean(dim=1)  # (B,vis_dim)
        text_avg = aligned_text.mean(dim=1)  # (B,text_dim)
        context_vec = torch.cat([vis_avg, text_avg], dim=1)  # (B,vis_dim + text_dim)
        hidden_context = self.context_pool(context_vec)  # (B,hidden_dim)
        # Create initial hidden state
        init_hidden = self.llm.init_hidden_from_context(hidden_context)
        # Run LLM on all tokens (teacher forcing).  The initial hidden state
        # conditions the generation on the visual context.
        logits, _ = self.llm(padded_tokens, init_hidden)  # (B,L,V)
        return seg_masks, logits, hidden_context

    def generate(
        self,
        image: torch.Tensor,
        max_len: int = 30,
        temperature: float = 1.0,
    ) -> List[str]:
        """Generate a description for an input image.

        Parameters
        ----------
        image : torch.Tensor
            Single image tensor of shape (1,3,H,W).
        max_len : int
            Maximum number of tokens to generate.
        temperature : float
            Temperature for sampling in the language model.

        Returns
        -------
        List[str]
            A list containing the generated description string.
        """
        self.eval()
        with torch.no_grad():
            fine_mask, sam_features = self.cascade(image)
            prompt_features = self.prompt_learner(fine_mask)
            # We do not need the mask decoder outputs for generation
            # Compute SASA alignment
            # Dummy zero sequence for text embeddings (no teacher forcing at inference)
            # Use a single BOS token as placeholder
            bos = torch.tensor([[self.tokenizer.bos_id]], device=image.device)
            text_embeds = self.llm.embedding(bos)  # (1,1,embed_dim)
            aligned_vis, aligned_text = self.sasa(sam_features, fine_mask, text_embeds)
            vis_avg = aligned_vis.mean(dim=1)
            text_avg = aligned_text.mean(dim=1)
            context_vec = torch.cat([vis_avg, text_avg], dim=1)
            hidden_context = self.context_pool(context_vec)  # (1,hidden_dim)
            # Generate tokens
            token_ids_batch = self.llm.generate(
                context=hidden_context,
                tokenizer=self.tokenizer,
                max_len=max_len,
                temperature=temperature,
            )
            # Decode
            descriptions: List[str] = [self.tokenizer.decode(ids) for ids in token_ids_batch]
            return descriptions