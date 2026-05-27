"""Training script for the simplified SAM‑LLaVA model.

This script reads a YAML configuration file, constructs the dataset and
model, and trains the model end‑to‑end.  Both segmentation and language
losses are optimised jointly.  A small dataset and CPU‑friendly model
allow you to run this script on a laptop without a GPU.

Usage::

    python train.py --config path/to/config.yaml

The configuration file should define at minimum the dataset root and
hyper‑parameters.  See `code/config/train.yaml` for an example.
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

# Extend sys.path so that `code` becomes an importable package when running
# this script directly with `python train.py`.
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from datasets import DefectDataset  # type: ignore
from model import SamLlavaModel, SimpleTokenizer  # type: ignore
from utils.metrics import segmentation_loss, language_loss  # type: ignore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SAM‑LLaVA model")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML configuration file")
    parser.add_argument("--output", type=str, default="checkpoint.pth", help="Output checkpoint file")
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def train_model(cfg: dict, output_path: str) -> None:
    # Set random seeds for reproducibility
    random.seed(cfg.get("seed", 42))
    torch.manual_seed(cfg.get("seed", 42))
    # Device
    device = torch.device("cpu")
    # Load dataset
    dataset_root = cfg["dataset"]["root"]
    tokenizer = SimpleTokenizer()
    # Optionally sample a support set for few‑shot calibration
    support_size = cfg["dataset"].get("support_size", 0)
    full_dataset = DefectDataset(root=dataset_root)
    support_indices: List[int] = []
    if support_size > 0:
        support_indices = list(range(min(support_size, len(full_dataset))))
    # Split into training and validation sets
    val_ratio = cfg["dataset"].get("val_ratio", 0.1)
    val_len = int(len(full_dataset) * val_ratio)
    train_len = len(full_dataset) - val_len
    train_dataset, val_dataset = random_split(full_dataset, [train_len, val_len])
    # Wrap with support indices
    if support_indices:
        train_dataset = DefectDataset(root=dataset_root, support_indices=support_indices)
        val_dataset = DefectDataset(root=dataset_root, support_indices=support_indices)
    # DataLoaders
    batch_size = cfg["train"].get("batch_size", 4)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    # Build model
    model = SamLlavaModel(
        image_channels=3,
        clip_embed_dim=cfg["model"].get("clip_embed_dim", 64),
        prompt_dim=cfg["model"].get("prompt_dim", 32),
        llm_embed_dim=cfg["model"].get("llm_embed_dim", 128),
        llm_hidden_dim=cfg["model"].get("llm_hidden_dim", 256),
        llm_layers=cfg["model"].get("llm_layers", 2),
        align_dim=cfg["model"].get("align_dim", 64),
        num_heads=cfg["model"].get("num_heads", 4),
        tokenizer=tokenizer,
    ).to(device)
    # Optimiser
    lr = cfg["train"].get("lr", 1e-3)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    epochs = cfg["train"].get("epochs", 5)
    # Loss weights
    alpha = cfg["train"].get("alpha", 1.0)
    gamma = cfg["train"].get("gamma", 1.0)
    # Training loop
    for epoch in range(1, epochs + 1):
        model.train()
        total_seg_loss = 0.0
        total_lang_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [train]"):
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            descriptions = batch["description"]
            # Forward pass
            seg_preds, logits, context = model(images, descriptions)
            # Compute losses
            seg_loss = segmentation_loss(seg_preds, masks)
            # Prepare target tokens
            token_seqs = [tokenizer.encode(desc) for desc in descriptions]
            max_len = max(len(seq) for seq in token_seqs)
            target_tokens = torch.full((images.size(0), max_len), tokenizer.pad_id, dtype=torch.long, device=device)
            for i, seq in enumerate(token_seqs):
                target_tokens[i, : len(seq)] = torch.tensor(seq, device=device)
            lang_loss = language_loss(logits, target_tokens, tokenizer.pad_id)
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
                    seg_preds, logits, context = model(images, descriptions)
                    seg_loss = segmentation_loss(seg_preds, masks)
                    token_seqs = [tokenizer.encode(desc) for desc in descriptions]
                    max_len = max(len(seq) for seq in token_seqs)
                    target_tokens = torch.full((images.size(0), max_len), tokenizer.pad_id, dtype=torch.long, device=device)
                    for i, seq in enumerate(token_seqs):
                        target_tokens[i, : len(seq)] = torch.tensor(seq, device=device)
                    lang_loss = language_loss(logits, target_tokens, tokenizer.pad_id)
                    val_seg_loss += seg_loss.item()
                    val_lang_loss += lang_loss.item()
        n_train = len(train_loader)
        # Compute averaged losses
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
    # Resolve dataset root relative to config file location if provided as a
    # relative path.  This makes it easy to keep the dataset directory next
    # to the config file.
    dataset_root = cfg["dataset"]["root"]
    if not os.path.isabs(dataset_root):
        cfg["dataset"]["root"] = os.path.normpath(
            os.path.join(os.path.dirname(args.config), dataset_root)
        )
    train_model(cfg, args.output)


if __name__ == "__main__":
    main()