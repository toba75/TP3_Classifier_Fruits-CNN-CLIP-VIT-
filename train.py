#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import torch
import torch.nn as nn
from timm.utils import ModelEmaV2
from torch.utils.data import DataLoader
from tqdm import tqdm

from training_utils import (
    CLASSES,
    build_cosine_scheduler,
    build_datasets,
    build_model,
    build_optimizer,
    evaluate_model,
    freeze_backbone,
    mixup_criterion,
    maybe_apply_mixup,
    seed_everything,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ConvNeXtV2 or CLIP ViT-L/14 on flowers")

    parser.add_argument("--model", required=True, choices=["convnextv2_tiny", "convnextv2_base", "clip_vitl14"])
    parser.add_argument("--train-dir", default="data/train")
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--img-size", type=int, default=224)

    parser.add_argument("--optimizer", default="adamw")
    parser.add_argument("--lr-head", type=float, default=1e-3)
    parser.add_argument("--lr-backbone", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.02)
    parser.add_argument("--warmup-epochs", type=int, default=3)
    parser.add_argument("--scheduler", default="cosine")

    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--mixup", type=float, default=0.0)
    parser.add_argument("--cutmix", type=float, default=0.0)
    parser.add_argument("--randaugment", type=int, default=0)
    parser.add_argument("--random-erasing", type=float, default=0.0)

    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--ema", action="store_true")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--freeze-backbone-epochs", type=int, default=0)
    parser.add_argument("--llrd", type=float, default=1.0)
    parser.add_argument("--early-stopping-patience", type=int, default=10)
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.001)

    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    require_cuda = os.getenv("REQUIRE_CUDA", "0").strip().lower() in {"1", "true", "yes", "on"}
    if require_cuda and device.type != "cuda":
        raise RuntimeError(
            "GPU requis mais indisponible (REQUIRE_CUDA=1). "
            "Lance le conteneur avec --gpus all ou désactive REQUIRE_CUDA."
        )

    bundle = build_datasets(
        train_dir=args.train_dir,
        val_split=args.val_split,
        seed=args.seed,
        model_name=args.model,
        img_size=args.img_size,
        classes=CLASSES,
    )

    train_loader = DataLoader(
        bundle.train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        bundle.val_dataset,
        batch_size=max(1, args.batch_size),
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )

    model = build_model(args.model, num_classes=len(CLASSES), pretrained=True).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    freeze_backbone(model, freeze=args.freeze_backbone_epochs > 0)
    optimizer = build_optimizer(
        model=model,
        lr_head=args.lr_head,
        lr_backbone=args.lr_backbone,
        weight_decay=args.weight_decay,
    )
    scheduler = build_cosine_scheduler(optimizer, epochs=args.epochs, warmup_epochs=args.warmup_epochs)

    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and torch.cuda.is_available())
    model_ema = ModelEmaV2(model, decay=0.9998) if args.ema else None

    history = []
    best_val_acc = -1.0
    best_epoch = -1
    epochs_without_improvement = 0
    stopped_early = False

    for epoch in range(args.epochs):
        if args.freeze_backbone_epochs > 0:
            freeze_backbone(model, freeze=epoch < args.freeze_backbone_epochs)

        model.train()
        train_loss_sum = 0.0
        train_count = 0

        progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}", leave=False)
        for images, targets in progress:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            mixed_images, targets_a, targets_b, lam = maybe_apply_mixup(images, targets, args.mixup)

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=device.type, enabled=args.amp and torch.cuda.is_available()):
                logits = model(mixed_images)
                if lam < 1.0:
                    loss = mixup_criterion(criterion, logits, targets_a, targets_b, lam)
                else:
                    loss = criterion(logits, targets)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            if model_ema is not None:
                model_ema.update(model)

            batch_size = targets.size(0)
            train_loss_sum += float(loss.item()) * batch_size
            train_count += batch_size
            progress.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()

        eval_model = model_ema.module if model_ema is not None else model
        val_loss, val_acc = evaluate_model(eval_model, val_loader, device=device, criterion=criterion)
        train_loss = train_loss_sum / max(1, train_count)

        row = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
        }
        history.append(row)

        if val_acc > (best_val_acc + args.early_stopping_min_delta):
            best_val_acc = val_acc
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            checkpoint = {
                "model_name": args.model,
                "num_classes": len(CLASSES),
                "class_names": CLASSES,
                "img_size": args.img_size,
                "state_dict": eval_model.state_dict(),
                "best_val_accuracy": best_val_acc,
                "epoch": best_epoch,
            }
            torch.save(checkpoint, args.output_dir / "best.pth")
        else:
            epochs_without_improvement += 1

        if args.early_stopping_patience > 0 and epochs_without_improvement >= args.early_stopping_patience:
            stopped_early = True
            print(
                f"Early stopping at epoch {epoch + 1}: no val_accuracy improvement "
                f"for {epochs_without_improvement} epoch(s) "
                f"(min_delta={args.early_stopping_min_delta})."
            )
            break

        torch.save(
            {
                "model_name": args.model,
                "num_classes": len(CLASSES),
                "class_names": CLASSES,
                "img_size": args.img_size,
                "state_dict": model.state_dict(),
                "epoch": epoch + 1,
            },
            args.output_dir / "last.pth",
        )

    with (args.output_dir / "history.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss", "val_accuracy"])
        writer.writeheader()
        writer.writerows(history)

    summary = {
        "model": args.model,
        "seed": args.seed,
        "epochs": args.epochs,
        "epochs_trained": len(history),
        "best_epoch": best_epoch,
        "best_val_accuracy": best_val_acc,
        "val_accuracy": best_val_acc,
        "stopped_early": stopped_early,
        "early_stopping_patience": args.early_stopping_patience,
        "early_stopping_min_delta": args.early_stopping_min_delta,
        "train_dir": args.train_dir,
        "output_dir": str(args.output_dir),
    }
    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
