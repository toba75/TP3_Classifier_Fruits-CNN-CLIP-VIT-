#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from training_utils import CLASSES, build_datasets, build_model, evaluate_model, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint on validation split")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--train-dir", default="data/train")
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--split", default="val", choices=["val"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model_name = checkpoint.get("model_name", "convnextv2_tiny")
    img_size = int(checkpoint.get("img_size", args.img_size))

    bundle = build_datasets(
        train_dir=args.train_dir,
        val_split=args.val_split,
        seed=args.seed,
        model_name=model_name,
        img_size=img_size,
        classes=CLASSES,
    )

    dataloader = DataLoader(
        bundle.val_dataset,
        batch_size=max(1, args.batch_size),
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )

    model = build_model(model_name=model_name, num_classes=len(CLASSES), pretrained=False)
    model.load_state_dict(checkpoint["state_dict"], strict=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

    val_loss, val_acc = evaluate_model(model=model, dataloader=dataloader, device=device, criterion=criterion)

    result = {
        "checkpoint": str(args.checkpoint),
        "model": model_name,
        "val_loss": val_loss,
        "val_accuracy": val_acc,
    }

    out_path = args.checkpoint.parent / "eval.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
