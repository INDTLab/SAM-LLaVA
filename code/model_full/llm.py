"""Vicuna‑7B‑v1.5 language model with LoRA adapters.

This module defines a wrapper around HuggingFace’s Vicuna‑7B‑v1.5 model
with low‑rank adaptation (LoRA) applied to key/value projections and
feed‑forward network layers as described in the SAM‑LLaVA paper【409945085282613†L540-L546】.  The LoRA
configuration (rank ``r=16``, scaling ``alpha=32``) follows the
experimental setup in the paper【409945085282613†L540-L546】.

The `VicunaLLM` class exposes methods to tokenize input text, generate
text conditioned on image embeddings, and return the underlying model
for training.  It uses the ``peft`` library to insert LoRA modules
efficiently.  When initialised, the class will download the base
Vicuna weights unless they are cached locally.  LoRA parameters are
initialised randomly; you are expected to fine‑tune them during
training.

Requires installation of `transformers` and `peft` packages.
"""

from __future__ import annotations

import os
from typing import List, Optional

import torch
from torch import nn

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "transformers must be installed to use the full SAM‑LLaVA model. "
        "Install with `pip install transformers`.")

try:
    from peft import LoraConfig, get_peft_model
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "peft must be installed to use the full SAM‑LLaVA model. "
        "Install with `pip install peft`.")


class VicunaLLM(nn.Module):
    """Vicuna‑7B language model with LoRA adapters.

    Parameters
    ----------
    model_name : str, optional
        HuggingFace model id for Vicuna.  Defaults to ``"lmsys/vicuna-7b-v1.5"``.
    lora_r : int, optional
        Rank of the LoRA adapters.  Defaults to 16 as per the paper【409945085282613†L540-L546】.
    lora_alpha : int, optional
        Scaling factor for LoRA.  Defaults to 32 as per the paper【409945085282613†L540-L546】.
    target_modules : list[str], optional
        Names of linear modules to which LoRA should be applied.  By
        default, LoRA is inserted into the query/key/value projections
        (``q_proj``, ``v_proj``) and the feed‑forward layers
        (``gate_proj``, ``up_proj``, ``down_proj``).  These names match
        parameter names in HuggingFace’s `Llama` implementation on which
        Vicuna is based.
    device : torch.device or str, optional
        Device on which to run the model.
    """

    def __init__(self,
                 model_name: str = "lmsys/vicuna-7b-v1.5",
                 lora_r: int = 16,
                 lora_alpha: int = 32,
                 target_modules: Optional[List[str]] = None,
                 device: torch.device | str = "cpu") -> None:
        super().__init__()
        self.device = torch.device(device)
        # Load tokenizer and base model.  We disable caching on load
        # because Vicuna uses the same config as LLaMA.
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
            model_name, use_fast=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        # Load model in 16‑bit precision if available to save memory
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
        self.model.to(self.device)
        # Configure LoRA adapters
        if target_modules is None:
            target_modules = [
                "q_proj", "v_proj", "gate_proj", "up_proj", "down_proj"
            ]
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        # Insert LoRA modules into the base model
        self.model = get_peft_model(self.model, lora_config)
        # Put LoRA parameters on the correct device
        self.model.to(self.device)
        self.model.train()

    def encode(self, texts: List[str]) -> torch.Tensor:
        """Tokenise a list of strings into input ids and attention masks.

        Returns
        -------
        input_ids : torch.Tensor
            Tensor of shape (B, L) of token ids.
        """
        encodings = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            add_special_tokens=True,
        ).to(self.device)
        return encodings["input_ids"]

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Run the language model and return logits.

        Parameters
        ----------
        input_ids : torch.Tensor
            Input token ids of shape (B, L).
        attention_mask : torch.Tensor, optional
            Optional attention mask.  If ``None``, the model will create one.

        Returns
        -------
        logits : torch.Tensor
            Model output logits of shape (B, L, vocab_size).
        """
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        return outputs.logits

    def generate(self,
                 input_ids: torch.Tensor,
                 attention_mask: Optional[torch.Tensor] = None,
                 max_new_tokens: int = 50,
                 **gen_kwargs) -> torch.Tensor:
        """Generate continuations given input ids.

        This method wraps ``model.generate`` and uses Greedy decoding by
        default.  You can pass additional generation parameters via
        ``gen_kwargs``.
        """
        # Ensure the model is in evaluation mode for generation
        prev_mode = self.model.training
        self.model.eval()
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                **gen_kwargs
            )
        # Restore training mode
        self.model.train(prev_mode)
        return outputs
