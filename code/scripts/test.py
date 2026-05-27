"""Evaluation script for the simplified SAM‑LLaVA model.

This script loads a trained checkpoint and evaluates the model on a test split.
It reports the average segmentation and language losses, as well as the Dice
coefficient.  You can adapt the script to compute AUROC or other metrics.

Usage::

    python test.py --config path/to/config.yaml --checkpoint path/to/checkpoint.pth
"""

from __future__ import annotations

import argparse
import os
import yaml

import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from datasets import DefectDataset  # type: ignore
from model import SamLlavaModel, SimpleTokenizer  # type: ignore
from utils.metrics import segmentation_loss, language_loss, dice_loss  # type: ignore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate SAM‑LLaVA model")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML configuration file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def evaluate(cfg: dict, checkpoint_path: str) -> None:
    device = torch.device("cpu")
    tokenizer = SimpleTokenizer()
    # Load dataset
    dataset_root = cfg["dataset"]["root"]
    full_dataset = DefectDataset(root=dataset_root)
    # Split into train/val/test.  Use the validation ratio for test set.
    val_ratio = cfg["dataset"].get("val_ratio", 0.1)
    test_len = int(len(full_dataset) * val_ratio)
    train_len = len(full_dataset) - test_len
    _, test_dataset = random_split(full_dataset, [train_len, test_len])
    test_loader = DataLoader(test_dataset, batch_size=cfg["train"].get("batch_size", 4), shuffle=False, num_workers=0)
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
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    total_seg_loss = 0.0
    total_lang_loss = 0.0
    total_dice = 0.0
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            descriptions = batch["description"]
            seg_preds, logits, context = model(images, descriptions)
            # Compute segmentation loss and dice
            seg_loss = segmentation_loss(seg_preds, masks)
            high_mask = seg_preds[0]
            dice = dice_loss(high_mask, masks)
            # Prepare target tokens
            token_seqs = [tokenizer.encode(desc) for desc in descriptions]
            max_len = max(len(seq) for seq in token_seqs)
            target_tokens = torch.full((images.size(0), max_len), tokenizer.pad_id, dtype=torch.long, device=device)
            for i, seq in enumerate(token_seqs):
                target_tokens[i, : len(seq)] = torch.tensor(seq, device=device)
            lang_loss = language_loss(logits, target_tokens, tokenizer.pad_id)
            total_seg_loss += seg_loss.item()
            total_lang_loss += lang_loss.item()
            total_dice += dice.item()
    n_batches = len(test_loader)
    if n_batches == 0:
        print("Test set is empty.  Please provide more data or adjust the val_ratio in the config.")
        return
    print(
        f"Segmentation loss: {total_seg_loss/n_batches:.4f}, "
        f"Language loss: {total_lang_loss/n_batches:.4f}, "
        f"Dice coefficient: {1 - (total_dice/n_batches):.4f}"
    )


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    dataset_root = cfg["dataset"]["root"]
    if not os.path.isabs(dataset_root):
        cfg["dataset"]["root"] = os.path.normpath(
            os.path.join(os.path.dirname(args.config), dataset_root)
        )
    evaluate(cfg, args.checkpoint)


if __name__ == "__main__":
    main()