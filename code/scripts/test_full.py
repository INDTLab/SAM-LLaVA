"""Evaluation script for the full SAM‑LLaVA model.

This script loads a checkpoint trained with `train_full.py` and evaluates
the model on a held‑out test split.  It reports the average Dice
coefficient, binary cross entropy and language modelling losses.  Note
that because the heavy models are frozen by default, the number of
trainable parameters is small and training may converge slowly on small
datasets.
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
from model_full import SamLlavaModelFull  # type: ignore
from utils.metrics import dice_loss, bce_loss, language_loss  # type: ignore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the full SAM‑LLaVA model")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML configuration file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def evaluate(cfg: dict, checkpoint_path: str) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.eval()
    # Load dataset
    dataset_root = cfg["dataset"]["root"]
    full_dataset = DefectDataset(root=dataset_root)
    # Split into train/val/test: we use val_ratio as test split
    val_ratio = cfg["dataset"].get("val_ratio", 0.1)
    test_len = int(len(full_dataset) * val_ratio)
    train_len = len(full_dataset) - test_len
    _, test_dataset = random_split(full_dataset, [train_len, test_len])
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
    total_dice = 0.0
    total_bce = 0.0
    total_lang = 0.0
    n = 0
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            descriptions = batch["description"]
            refined_masks, logits = model(images, descriptions)
            # Compute segmentation losses
            dloss = dice_loss(refined_masks, masks)
            bloss = bce_loss(refined_masks, masks)
            # Prepare targets
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
            lloss = language_loss(logits, target_tokens, model.llm.tokenizer.pad_token_id)
            total_dice += dloss.item()
            total_bce += bloss.item()
            total_lang += lloss.item()
            n += 1
    if n == 0:
        print("Test set is empty. Adjust val_ratio or provide more data.")
        return
    print(
        f"Dice loss: {total_dice/n:.4f}, BCE loss: {total_bce/n:.4f}, Language loss: {total_lang/n:.4f}"
    )


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    # Resolve dataset root and SAM checkpoint relative paths
    if not os.path.isabs(cfg["dataset"]["root"]):
        cfg["dataset"]["root"] = os.path.normpath(
            os.path.join(os.path.dirname(args.config), cfg["dataset"]["root"])
        )
    sam_ckpt = cfg["model"].get("sam_checkpoint")
    if sam_ckpt is not None and not os.path.isabs(sam_ckpt):
        cfg["model"]["sam_checkpoint"] = os.path.normpath(
            os.path.join(os.path.dirname(args.config), sam_ckpt)
        )
    evaluate(cfg, args.checkpoint)


if __name__ == "__main__":
    main()
