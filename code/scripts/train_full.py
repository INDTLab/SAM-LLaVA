"""Training script for the full SAM‑LLaVA model.

This script reads a YAML configuration file, constructs the dataset and
full SAM‑LLaVA model (CLIP ViT‑B/16, SAM ViT‑H, Vicuna‑7B with LoRA), and
trains the LoRA parameters to perform joint segmentation and captioning.
The heavy backbones (CLIP, SAM, Vicuna) are frozen by default to
replicate the efficient fine‑tuning strategy described in the SAM‑LLaVA
paper【409945085282613†L540-L546】.  Only the LoRA adapters and a small context
projection layer are updated during training.

Usage::

    python train_full.py --config path/to/config.yaml

The configuration file should specify the dataset root and the path to
the SAM ViT‑H checkpoint.  See `code/config/train.yaml` for an example of
how to extend the config for the full model.
"""

from __future__ import annotations

import argparse
import os
import random
import time
from typing import List

import yaml
import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from datasets import DefectDataset  # type: ignore
from model_full import SamLlavaModelFull  # type: ignore
from utils.metrics import dice_loss, bce_loss, language_loss  # type: ignore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the full SAM‑LLaVA model")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML configuration file")
    parser.add_argument("--output", type=str, default="full_checkpoint.pth", help="Output checkpoint file")
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def train_model(cfg: dict, output_path: str) -> None:
    # Set random seeds for reproducibility
    seed = cfg.get("seed", 42)
    random.seed(seed)
    torch.manual_seed(seed)
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load dataset
    dataset_root = cfg["dataset"]["root"]
    full_dataset = DefectDataset(root=dataset_root)
    # Split into train and validation sets
    val_ratio = cfg["dataset"].get("val_ratio", 0.1)
    val_len = int(len(full_dataset) * val_ratio)
    train_len = len(full_dataset) - val_len
    train_dataset, val_dataset = random_split(full_dataset, [train_len, val_len])
    # Data loaders
    batch_size = cfg["train"].get("batch_size", 1)  # using batch_size=1 for memory reasons
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    # Build model
    sam_checkpoint = cfg["model"].get("sam_checkpoint")
    clip_model_name = cfg["model"].get("clip_model_name", "openai/clip-vit-base-patch16")
    llm_model_name = cfg["model"].get("llm_model_name", "lmsys/vicuna-7b-v1.5")
    lora_r = cfg["model"].get("lora_r", 16)
    lora_alpha = cfg["model"].get("lora_alpha", 32)
    model = SamLlavaModelFull(
        clip_model_name=clip_model_name,
        sam_checkpoint=sam_checkpoint,
        llm_model_name=llm_model_name,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        device=device,
    ).to(device)
    # Freeze the backbones: CLIP vision encoder, SAM and Vicuna base parameters
    # Only LoRA adapters and context projection will be trained
    for name, param in model.named_parameters():
        if not any(s in name for s in ["lora", "context_proj"]):
            param.requires_grad = False
    # Optimiser
    lr = cfg["train"].get("lr", 1e-4)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr)
    epochs = cfg["train"].get("epochs", 1)
    # Loss weights
    alpha = cfg["train"].get("alpha", 1.0)
    gamma = cfg["train"].get("gamma", 1.0)
    for epoch in range(1, epochs + 1):
        model.train()
        total_seg_loss = 0.0
        total_lang_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [train]"):
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            descriptions = batch["description"]
            # Forward
            refined_masks, logits = model(images, descriptions)
            # Segmentation loss: dice + BCE on refined mask
            seg_dice = dice_loss(refined_masks, masks)
            seg_bce = bce_loss(refined_masks, masks)
            seg_loss = seg_dice + seg_bce
            # Prepare target tokens
            # We align each description length to the same length by padding with eos
            token_seqs = [model.llm.tokenizer.encode(desc) for desc in descriptions]
            max_len = max(len(seq) for seq in token_seqs)
            target_tokens = torch.full((images.size(0), max_len), model.llm.tokenizer.pad_token_id, dtype=torch.long, device=device)
            for i, seq in enumerate(token_seqs):
                target_tokens[i, : len(seq)] = torch.tensor(seq, device=device)
            # Pad logits if needed
            # logits shape (B,L,V); if L < max_len, pad; if L > max_len, truncate
            logits = logits[:, :max_len, :]
            if logits.size(1) < max_len:
                pad_size = max_len - logits.size(1)
                pad_logits = logits[:, -1:, :].repeat(1, pad_size, 1)
                logits = torch.cat([logits, pad_logits], dim=1)
            lang_loss = language_loss(logits, target_tokens, model.llm.tokenizer.pad_token_id)
            loss = alpha * seg_loss + gamma * lang_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_seg_loss += seg_loss.item()
            total_lang_loss += lang_loss.item()
        # Validation
        model.eval()
        val_seg_loss = 0.0
        val_lang_loss = 0.0
        n_val = len(val_loader)
        if n_val > 0:
            with torch.no_grad():
                for batch in tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} [val]"):
                    images = batch["image"].to(device)
                    masks = batch["mask"].to(device)
                    descriptions = batch["description"]
                    refined_masks, logits = model(images, descriptions)
                    seg_dice = dice_loss(refined_masks, masks)
                    seg_bce = bce_loss(refined_masks, masks)
                    seg_loss = seg_dice + seg_bce
                    # Prepare tokens
                    token_seqs = [model.llm.tokenizer.encode(desc) for desc in descriptions]
                    max_len = max(len(seq) for seq in token_seqs)
                    target_tokens = torch.full((images.size(0), max_len), model.llm.tokenizer.pad_token_id, dtype=torch.long, device=device)
                    for i, seq in enumerate(token_seqs):
                        target_tokens[i, : len(seq)] = torch.tensor(seq, device=device)
                    logits = logits[:, :max_len, :]
                    if logits.size(1) < max_len:
                        pad_size = max_len - logits.size(1)
                        pad_logits = logits[:, -1:, :].repeat(1, pad_size, 1)
                        logits = torch.cat([logits, pad_logits], dim=1)
                    lang_loss = language_loss(logits, target_tokens, model.llm.tokenizer.pad_token_id)
                    val_seg_loss += seg_loss.item()
                    val_lang_loss += lang_loss.item()
        n_train = len(train_loader)
        train_seg_avg = total_seg_loss / n_train
        train_lang_avg = total_lang_loss / n_train
        if n_val > 0:
            val_seg_avg = val_seg_loss / n_val
            val_lang_avg = val_lang_loss / n_val
            print(
                f"Epoch {epoch}: train seg loss {train_seg_avg:.4f}, train lang loss {train_lang_avg:.4f}; "
                f"val seg loss {val_seg_avg:.4f}, val lang loss {val_lang_avg:.4f}"
            )
        else:
            print(
                f"Epoch {epoch}: train seg loss {train_seg_avg:.4f}, train lang loss {train_lang_avg:.4f}"
            )
    # Save checkpoint
    torch.save({"model_state_dict": model.state_dict()}, output_path)
    print(f"Training complete.  Checkpoint saved to {output_path}.")


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    # Resolve dataset root relative to config file location
    dataset_root = cfg["dataset"]["root"]
    if not os.path.isabs(dataset_root):
        cfg["dataset"]["root"] = os.path.normpath(
            os.path.join(os.path.dirname(args.config), dataset_root)
        )
    # Resolve SAM checkpoint relative path if provided
    sam_ckpt = cfg["model"].get("sam_checkpoint")
    if sam_ckpt is not None and not os.path.isabs(sam_ckpt):
        cfg["model"]["sam_checkpoint"] = os.path.normpath(
            os.path.join(os.path.dirname(args.config), sam_ckpt)
        )
    train_model(cfg, args.output)


if __name__ == "__main__":
    main()
