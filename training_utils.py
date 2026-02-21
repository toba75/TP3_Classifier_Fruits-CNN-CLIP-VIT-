from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np
import timm
import torch
import torch.nn as nn
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision import transforms

CLASSES: List[str] = ["daisy", "dandelion", "rose", "sunflower", "tulip"]
CNN_MODEL_NAME = "convnextv2_large"
CNN_PRETRAINED_TAG = "fcmae_ft_in22k_in1k"
CNN_TIMM_MODEL_ID = f"{CNN_MODEL_NAME}.{CNN_PRETRAINED_TAG}"

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


@dataclass
class DatasetBundle:
    train_dataset: Dataset
    val_dataset: Dataset
    class_to_idx: Dict[str, int]


class FlowersDataset(Dataset):
    def __init__(
        self,
        items: Sequence[Tuple[Path, int]],
        transform: Callable | None,
    ) -> None:
        self.items = list(items)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        path, label = self.items[index]
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label


class CLIPClassifier(nn.Module):
    def __init__(self, clip_model: nn.Module, embed_dim: int, num_classes: int) -> None:
        super().__init__()
        self.visual = clip_model.visual
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.visual(images)
        if features.ndim > 2:
            features = features.mean(dim=1)
        return self.classifier(features)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(model_name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    if model_name == CNN_MODEL_NAME:
        return timm.create_model(CNN_TIMM_MODEL_ID, pretrained=pretrained, num_classes=num_classes)

    if model_name == "clip_vitl14":
        import open_clip

        pretrained_tag = "openai" if pretrained else None
        clip_model, _, _ = open_clip.create_model_and_transforms("ViT-L-14", pretrained=pretrained_tag)
        embed_dim = int(getattr(clip_model.visual, "output_dim"))
        return CLIPClassifier(clip_model, embed_dim=embed_dim, num_classes=num_classes)

    raise ValueError(f"Unsupported model: {model_name}")


def build_transform(model_name: str, img_size: int, train: bool) -> transforms.Compose:
    if model_name == "clip_vitl14":
        mean, std = CLIP_MEAN, CLIP_STD
    else:
        mean, std = IMAGENET_MEAN, IMAGENET_STD

    if train:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

    return transforms.Compose(
        [
            transforms.Resize(int(img_size * 1.14)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def list_flower_items(data_dir: str | Path, classes: Sequence[str]) -> List[Tuple[Path, int]]:
    data_path = Path(data_dir)
    class_to_idx = {name: idx for idx, name in enumerate(classes)}
    items: List[Tuple[Path, int]] = []

    for class_name in classes:
        class_dir = data_path / class_name
        if not class_dir.exists():
            continue
        for image_path in class_dir.rglob("*"):
            if image_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                items.append((image_path, class_to_idx[class_name]))

    if not items:
        raise FileNotFoundError(f"No images found in {data_dir} for classes: {classes}")

    return items


def build_datasets(
    train_dir: str | Path,
    val_split: float,
    seed: int,
    model_name: str,
    img_size: int,
    classes: Sequence[str] = CLASSES,
) -> DatasetBundle:
    items = list_flower_items(train_dir, classes)
    labels = [label for _, label in items]
    train_items, val_items = train_test_split(
        items,
        test_size=val_split,
        random_state=seed,
        stratify=labels,
    )

    train_transform = build_transform(model_name=model_name, img_size=img_size, train=True)
    val_transform = build_transform(model_name=model_name, img_size=img_size, train=False)

    class_to_idx = {name: idx for idx, name in enumerate(classes)}
    return DatasetBundle(
        train_dataset=FlowersDataset(train_items, train_transform),
        val_dataset=FlowersDataset(val_items, val_transform),
        class_to_idx=class_to_idx,
    )


def maybe_apply_mixup(
    images: torch.Tensor,
    targets: torch.Tensor,
    mixup_alpha: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    if mixup_alpha <= 0:
        return images, targets, targets, 1.0

    lam = np.random.beta(mixup_alpha, mixup_alpha)
    batch_size = images.size(0)
    index = torch.randperm(batch_size, device=images.device)

    mixed_images = lam * images + (1.0 - lam) * images[index]
    targets_a, targets_b = targets, targets[index]
    return mixed_images, targets_a, targets_b, float(lam)


def mixup_criterion(
    criterion: nn.Module,
    preds: torch.Tensor,
    targets_a: torch.Tensor,
    targets_b: torch.Tensor,
    lam: float,
) -> torch.Tensor:
    return lam * criterion(preds, targets_a) + (1.0 - lam) * criterion(preds, targets_b)


def freeze_backbone(model: nn.Module, freeze: bool) -> None:
    for name, param in model.named_parameters():
        is_head = any(key in name.lower() for key in ["head", "fc", "classifier"])
        param.requires_grad = (not freeze) or is_head


def build_optimizer(
    model: nn.Module,
    lr_head: float,
    lr_backbone: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    head_params = []
    backbone_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(key in name.lower() for key in ["head", "fc", "classifier"]):
            head_params.append(param)
        else:
            backbone_params.append(param)

    param_groups = []
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": lr_backbone})
    if head_params:
        param_groups.append({"params": head_params, "lr": lr_head})

    return torch.optim.AdamW(param_groups, weight_decay=weight_decay)


def build_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    epochs: int,
    warmup_epochs: int,
):
    def lr_lambda(current_epoch: int) -> float:
        if current_epoch < warmup_epochs:
            return float(current_epoch + 1) / float(max(1, warmup_epochs))
        progress = (current_epoch - warmup_epochs) / float(max(1, epochs - warmup_epochs))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def evaluate_model(
    model: nn.Module,
    dataloader,
    device: torch.device,
    criterion: nn.Module,
) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            logits = model(images)
            loss = criterion(logits, targets)

            total_loss += float(loss.item()) * targets.size(0)
            preds = logits.argmax(dim=1)
            correct += int((preds == targets).sum().item())
            total += int(targets.size(0))

    avg_loss = total_loss / max(1, total)
    acc = correct / max(1, total)
    return avg_loss, acc
