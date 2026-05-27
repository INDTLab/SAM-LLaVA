"""A minimal language model and tokenizer for SAM‑LLaVA.

The SAM‑LLaVA framework uses LLaVA, a variant of the Vicuna large language
model, to generate defect descriptions from aligned visual and textual features【409945085282613†L530-L546】.  Since loading a multi‑billion‑parameter model is
impractical in this context, we implement a very small RNN‑based language
model that supports both training and autoregressive generation.  A simple
character‑level tokenizer is provided to convert between strings and integer
token sequences.
"""

from __future__ import annotations

import string
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleTokenizer:
    """A character‑level tokenizer with a small vocabulary.

    The vocabulary includes lowercase letters, digits, common punctuation and
    whitespace.  Four special tokens are added: `<pad>` for padding,
    `<bos>` to indicate the beginning of a sequence, `<eos>` to mark the end
    of a sequence and `<unk>` for unknown characters.
    """

    def __init__(self) -> None:
        # Define the base character set
        chars = list(string.ascii_lowercase + string.digits + " .,;!?:'-()/")
        specials = ["<pad>", "<bos>", "<eos>", "<unk>"]
        self.idx2tok: List[str] = specials + chars
        self.tok2idx: Dict[str, int] = {tok: i for i, tok in enumerate(self.idx2tok)}
        self.pad_token = "<pad>"
        self.bos_token = "<bos>"
        self.eos_token = "<eos>"
        self.unk_token = "<unk>"

    @property
    def pad_id(self) -> int:
        return self.tok2idx[self.pad_token]

    @property
    def bos_id(self) -> int:
        return self.tok2idx[self.bos_token]

    @property
    def eos_id(self) -> int:
        return self.tok2idx[self.eos_token]

    @property
    def unk_id(self) -> int:
        return self.tok2idx[self.unk_token]

    def encode(self, text: str, add_bos: bool = True, add_eos: bool = True) -> List[int]:
        """Convert a string into a list of token IDs.

        Unknown characters are mapped to `<unk>`.
        """
        text = text.lower()
        tokens: List[int] = []
        if add_bos:
            tokens.append(self.bos_id)
        for ch in text:
            tokens.append(self.tok2idx.get(ch, self.unk_id))
        if add_eos:
            tokens.append(self.eos_id)
        return tokens

    def decode(self, token_ids: List[int]) -> str:
        """Convert a list of token IDs back into a string.

        Special tokens are ignored during decoding.
        """
        chars: List[str] = []
        for idx in token_ids:
            tok = self.idx2tok[idx]
            if tok in (self.pad_token, self.bos_token, self.eos_token):
                continue
            chars.append(tok)
        return "".join(chars)

    @property
    def vocab_size(self) -> int:
        return len(self.idx2tok)


class SimpleLLM(nn.Module):
    """A small GRU‑based language model.

    This model takes token embeddings as input and returns logits over the
    vocabulary at each time step.  An initial hidden state can be supplied
    externally – for example, derived from aligned visual features via SASA –
    which allows the model to condition generation on the visual context.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)
        # To project context embeddings from SASA to GRU hidden size
        self.context_proj = nn.Linear(hidden_dim, hidden_dim)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

    def forward(
        self,
        tokens: torch.Tensor,
        hidden: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run the language model on a batch of token IDs.

        Parameters
        ----------
        tokens : torch.Tensor
            Input tensor of shape (B,L) containing token IDs.
        hidden : torch.Tensor
            Initial hidden state of shape (num_layers,B,hidden_dim).

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Logits of shape (B,L,vocab_size) and the final hidden state.
        """
        embeds = self.embedding(tokens)  # (B,L,embed_dim)
        output, h_n = self.gru(embeds, hidden)  # output: (B,L,hidden_dim)
        logits = self.fc_out(output)  # (B,L,vocab_size)
        return logits, h_n

    def init_hidden_from_context(
        self, context: torch.Tensor
    ) -> torch.Tensor:
        """Create an initial hidden state from a context embedding.

        Parameters
        ----------
        context : torch.Tensor
            Tensor of shape (B,hidden_dim) representing the aggregated
            visual/textual context from SASA.  It is projected and expanded
            across GRU layers.

        Returns
        -------
        torch.Tensor
            Initial hidden state for the GRU of shape (num_layers,B,hidden_dim).
        """
        # Project to hidden size and repeat for each layer
        projected = self.context_proj(context)  # (B,hidden_dim)
        # Expand to (num_layers,B,hidden_dim)
        h0 = projected.unsqueeze(0).expand(self.num_layers, -1, -1).contiguous()
        return h0

    def generate(
        self,
        context: torch.Tensor,
        tokenizer: SimpleTokenizer,
        max_len: int = 30,
        temperature: float = 1.0,
    ) -> List[List[int]]:
        """Autoregressively generate text conditioned on a context embedding.

        Parameters
        ----------
        context : torch.Tensor
            Tensor of shape (B,hidden_dim) from which the initial GRU hidden
            state is derived.
        tokenizer : SimpleTokenizer
            Tokenizer used to interpret token IDs.
        max_len : int, default=30
            Maximum length of the generated sequence (including BOS and EOS).
        temperature : float, default=1.0
            Softmax temperature.  Lower values make the distribution more
            peaky; higher values increase randomness.

        Returns
        -------
        List[List[int]]
            A list of generated token ID sequences for each batch element.
        """
        B = context.shape[0]
        hidden = self.init_hidden_from_context(context)  # (num_layers,B,hidden_dim)
        # Start with BOS token
        input_ids = torch.full((B, 1), tokenizer.bos_id, dtype=torch.long, device=context.device)
        generated: List[List[int]] = [[tokenizer.bos_id] for _ in range(B)]
        for _ in range(max_len - 1):
            embeds = self.embedding(input_ids)  # (B,1,embed_dim)
            output, hidden = self.gru(embeds, hidden)  # (B,1,hidden_dim)
            logits = self.fc_out(output.squeeze(1))  # (B,vocab_size)
            logits = logits / temperature
            probs = F.softmax(logits, dim=-1)  # (B,vocab_size)
            # Sample next token (greedy for deterministic generation)
            next_tokens = torch.argmax(probs, dim=-1)
            for i in range(B):
                generated[i].append(next_tokens[i].item())
            # Prepare next input
            input_ids = next_tokens.unsqueeze(1)
            # Stop if all sequences have generated EOS
            if all(tokenizer.eos_id in seq for seq in generated):
                break
        return generated