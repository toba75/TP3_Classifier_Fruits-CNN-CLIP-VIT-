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
    build_datasets,
    build_model,
    build_optimizer,
    build_scheduler,
    evaluate_model,
    freeze_backbone,
    mixup_criterion,
    maybe_apply_mixup_or_cutmix,
    seed_everything,
)


SOTA_DEFAULTS = {
    "convnextv2_large": {
        "epochs": 45,
        "batch_size": 32,
        "img_size": 224,
        "optimizer": "adamw",
        "lr_head": 8e-4,
        "lr_backbone": 8e-5,
        "weight_decay": 0.05,
        "warmup_epochs": 5,
        "scheduler": "cosine",
        "label_smoothing": 0.1,
        "mixup": 0.2,
        "cutmix": 1.0,
        "randaugment": 9,
        "random_erasing": 0.25,
        "amp": True,
        "ema": True,
        "grad_clip": 1.0,
        "freeze_backbone_epochs": 3,
        "llrd": 1.0,
        "drop_path_rate": 0.2,
    },
    "clip_vitl14": {
        "epochs": 30,
        "batch_size": 16,
        "img_size": 224,
        "optimizer": "adamw",
        "lr_head": 5e-4,
        "lr_backbone": 5e-6,
        "weight_decay": 0.05,
        "warmup_epochs": 3,
        "scheduler": "cosine",
        "label_smoothing": 0.05,
        "mixup": 0.1,
        "cutmix": 0.5,
        "randaugment": 7,
        "random_erasing": 0.1,
        "amp": True,
        "ema": True,
        "grad_clip": 1.0,
        "freeze_backbone_epochs": 8,
        "llrd": 0.75,
        "drop_path_rate": 0.0,
    },
    "dinov2_vitl14": {
        "epochs": 20,
        "batch_size": 16,
        "img_size": 224,
        "optimizer": "adamw",
        "lr_head": 5e-4,
        "lr_backbone": 5e-6,
        "weight_decay": 0.05,
        "warmup_epochs": 3,
        "scheduler": "cosine",
        "label_smoothing": 0.05,
        "mixup": 0.1,
        "cutmix": 0.5,
        "randaugment": 7,
        "random_erasing": 0.1,
        "amp": True,
        "ema": True,
        "grad_clip": 1.0,
        "freeze_backbone_epochs": 5,
        "llrd": 0.75,
        "drop_path_rate": 0.0,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ConvNeXtV2, CLIP ViT-L/14, or DINOv2 ViT-L/14 on flowers")

    parser.add_argument("--model", required=True, choices=["convnextv2_large", "clip_vitl14", "dinov2_vitl14"])
    parser.add_argument("--train-dir", default="data/train")
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--img-size", type=int, default=None)

    parser.add_argument("--optimizer", default=None, choices=["adamw", "sgd"])
    parser.add_argument("--lr-head", type=float, default=None)
    parser.add_argument("--lr-backbone", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--warmup-epochs", type=int, default=None)
    parser.add_argument("--scheduler", default=None, choices=["cosine", "step", "none"])

    parser.add_argument("--label-smoothing", type=float, default=None)
    parser.add_argument("--mixup", type=float, default=None)
    parser.add_argument("--cutmix", type=float, default=None)
    parser.add_argument("--randaugment", type=int, default=None)
    parser.add_argument("--random-erasing", type=float, default=None)
    parser.add_argument("--erasing-mode", type=str, default=None,
                        help="Random erasing mode (e.g. 'pixel'). Only for backends that support it.")

    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--ema", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--grad-clip", type=float, default=None)
    parser.add_argument("--freeze-backbone-epochs", type=int, default=None)
    parser.add_argument("--llrd", type=float, default=None)
    parser.add_argument("--drop-path-rate", type=float, default=None)
    parser.add_argument("--early-stopping-patience", type=int, default=10)
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.001)

    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    defaults = SOTA_DEFAULTS[args.model]
    for key, value in defaults.items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    return args


def build_effective_hparams(args: argparse.Namespace) -> dict:
    return {
        "model": args.model,
        "train_dir": args.train_dir,
        "val_split": args.val_split,
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "img_size": args.img_size,
        "optimizer": args.optimizer,
        "lr_head": args.lr_head,
        "lr_backbone": args.lr_backbone,
        "weight_decay": args.weight_decay,
        "warmup_epochs": args.warmup_epochs,
        "scheduler": args.scheduler,
        "label_smoothing": args.label_smoothing,
        "mixup": args.mixup,
        "cutmix": args.cutmix,
        "randaugment": args.randaugment,
        "random_erasing": args.random_erasing,
        "amp": args.amp,
        "ema": args.ema,
        "grad_clip": args.grad_clip,
        "freeze_backbone_epochs": args.freeze_backbone_epochs,
        "llrd": args.llrd,
        "drop_path_rate": args.drop_path_rate,
        "early_stopping_patience": args.early_stopping_patience,
        "early_stopping_min_delta": args.early_stopping_min_delta,
        "output_dir": str(args.output_dir),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    effective_hparams = build_effective_hparams(args)

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    require_cuda = os.getenv("REQUIRE_CUDA", "0").strip().lower() in {"1", "true", "yes", "on"}
    if require_cuda and device.type != "cuda":
        raise RuntimeError(
            "GPU requis mais indisponible (REQUIRE_CUDA=1). "
            "Lance le conteneur avec --gpus all ou désactive REQUIRE_CUDA."
        )

    print(
        json.dumps(
            {
                "run_start": {
                    "device": str(device),
                    "effective_hparams": effective_hparams,
                }
            },
            indent=2,
        )
    )

    bundle = build_datasets(
        train_dir=args.train_dir,
        val_split=args.val_split,
        seed=args.seed,
        model_name=args.model,
        img_size=args.img_size,
        randaugment=args.randaugment,
        random_erasing=args.random_erasing,
        erasing_mode=args.erasing_mode,
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

    model = build_model(
        args.model,
        num_classes=len(CLASSES),
        pretrained=True,
        drop_path_rate=args.drop_path_rate,
    ).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    optimizer = build_optimizer(
        model=model,
        optimizer_name=args.optimizer,
        lr_head=args.lr_head,
        lr_backbone=args.lr_backbone,
        weight_decay=args.weight_decay,
        llrd=args.llrd,
    )
    freeze_backbone(model, freeze=args.freeze_backbone_epochs > 0)
    scheduler = build_scheduler(
        optimizer=optimizer,
        scheduler_name=args.scheduler,
        epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
    )

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

            mixed_images, targets_a, targets_b, lam = maybe_apply_mixup_or_cutmix(
                images=images,
                targets=targets,
                mixup_alpha=args.mixup,
                cutmix_alpha=args.cutmix,
            )

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

        if scheduler is not None:
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
        "effective_hparams": effective_hparams,
    }
    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
