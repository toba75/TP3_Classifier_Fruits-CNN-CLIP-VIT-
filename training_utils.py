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


class DINOv2Classifier(nn.Module):
    def __init__(self, backbone: nn.Module, embed_dim: int, num_classes: int) -> None:
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.backbone(images)
        if features.ndim > 2:
            features = features.mean(dim=1)
        return self.classifier(features)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(
    model_name: str,
    num_classes: int,
    pretrained: bool = True,
    drop_path_rate: float = 0.0,
) -> nn.Module:
    if model_name == CNN_MODEL_NAME:
        return timm.create_model(
            CNN_TIMM_MODEL_ID,
            pretrained=pretrained,
            num_classes=num_classes,
            drop_path_rate=float(drop_path_rate),
        )

    if model_name == "clip_vitl14":
        import open_clip

        pretrained_tag = "openai" if pretrained else None
        clip_model, _, _ = open_clip.create_model_and_transforms("ViT-L-14", pretrained=pretrained_tag)
        embed_dim = int(getattr(clip_model.visual, "output_dim"))
        return CLIPClassifier(clip_model, embed_dim=embed_dim, num_classes=num_classes)

    if model_name == "dinov2_vitl14":
        backbone = torch.hub.load(
            "facebookresearch/dinov2",
            "dinov2_vitl14",
            pretrained=pretrained,
        )
        embed_dim = backbone.embed_dim  # 1024 for ViT-L
        return DINOv2Classifier(backbone, embed_dim=embed_dim, num_classes=num_classes)

    raise ValueError(f"Unsupported model: {model_name}")


def build_transform(
    model_name: str,
    img_size: int,
    train: bool,
    randaugment: int = 0,
    random_erasing: float = 0.0,
    erasing_mode: str | None = None,
) -> transforms.Compose:
    if model_name == "clip_vitl14":
        mean, std = CLIP_MEAN, CLIP_STD
    else:
        # ConvNeXt V2 and DINOv2 both use ImageNet normalization
        mean, std = IMAGENET_MEAN, IMAGENET_STD

    if train:
        train_transforms = [
            transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
        ]
        if randaugment > 0:
            train_transforms.append(transforms.RandAugment(num_ops=2, magnitude=randaugment))
        else:
            train_transforms.append(
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)
            )

        train_transforms.extend(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
        if random_erasing > 0:
            re_kwargs: dict = {"p": random_erasing}
            if erasing_mode is not None:
                re_kwargs["mode"] = erasing_mode
            train_transforms.append(transforms.RandomErasing(**re_kwargs))

        return transforms.Compose(train_transforms)

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
    randaugment: int = 0,
    random_erasing: float = 0.0,
    erasing_mode: str | None = None,
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

    train_transform = build_transform(
        model_name=model_name,
        img_size=img_size,
        train=True,
        randaugment=randaugment,
        random_erasing=random_erasing,
        erasing_mode=erasing_mode,
    )
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


def _rand_bbox(width: int, height: int, lam: float) -> Tuple[int, int, int, int]:
    cut_ratio = math.sqrt(max(0.0, 1.0 - lam))
    cut_width = int(width * cut_ratio)
    cut_height = int(height * cut_ratio)

    center_x = int(np.random.randint(0, max(1, width)))
    center_y = int(np.random.randint(0, max(1, height)))

    x1 = max(center_x - cut_width // 2, 0)
    y1 = max(center_y - cut_height // 2, 0)
    x2 = min(center_x + cut_width // 2, width)
    y2 = min(center_y + cut_height // 2, height)
    return x1, y1, x2, y2


def maybe_apply_mixup_or_cutmix(
    images: torch.Tensor,
    targets: torch.Tensor,
    mixup_alpha: float,
    cutmix_alpha: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    use_mixup = mixup_alpha > 0
    use_cutmix = cutmix_alpha > 0
    if not use_mixup and not use_cutmix:
        return images, targets, targets, 1.0

    if use_mixup and use_cutmix:
        apply_cutmix = bool(np.random.rand() < 0.5)
    else:
        apply_cutmix = use_cutmix

    if not apply_cutmix:
        return maybe_apply_mixup(images=images, targets=targets, mixup_alpha=mixup_alpha)

    lam = float(np.random.beta(cutmix_alpha, cutmix_alpha))
    batch_size = images.size(0)
    index = torch.randperm(batch_size, device=images.device)

    _, _, height, width = images.shape
    x1, y1, x2, y2 = _rand_bbox(width=width, height=height, lam=lam)
    mixed_images = images.clone()
    mixed_images[:, :, y1:y2, x1:x2] = images[index, :, y1:y2, x1:x2]

    cut_area = max(0, x2 - x1) * max(0, y2 - y1)
    lam_adjusted = 1.0 - (cut_area / float(width * height))
    targets_a, targets_b = targets, targets[index]
    return mixed_images, targets_a, targets_b, float(lam_adjusted)


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
    optimizer_name: str,
    lr_head: float,
    lr_backbone: float,
    weight_decay: float,
    llrd: float = 1.0,
) -> torch.optim.Optimizer:
    llrd = float(max(1e-8, llrd))

    def infer_layer_id(param_name: str) -> int:
        lower_name = param_name.lower()

        if "visual.transformer.resblocks." in lower_name:
            try:
                suffix = lower_name.split("visual.transformer.resblocks.", 1)[1]
                return int(suffix.split(".", 1)[0]) + 1
            except (IndexError, ValueError):
                return 1

        if "visual.trunk.blocks." in lower_name:
            try:
                suffix = lower_name.split("visual.trunk.blocks.", 1)[1]
                return int(suffix.split(".", 1)[0]) + 1
            except (IndexError, ValueError):
                return 1

        # DINOv2: backbone.blocks.X.…
        if "backbone.blocks." in lower_name:
            try:
                suffix = lower_name.split("backbone.blocks.", 1)[1]
                return int(suffix.split(".", 1)[0]) + 1
            except (IndexError, ValueError):
                return 1

        if "stages." in lower_name:
            try:
                suffix = lower_name.split("stages.", 1)[1]
                return int(suffix.split(".", 1)[0]) + 1
            except (IndexError, ValueError):
                return 1

        if "downsample_layers." in lower_name:
            try:
                suffix = lower_name.split("downsample_layers.", 1)[1]
                return int(suffix.split(".", 1)[0])
            except (IndexError, ValueError):
                return 0

        return 0

    def _no_weight_decay(name: str) -> bool:
        lower = name.lower()
        return (
            lower.endswith(".bias")
            or "norm" in lower
            or "bn" in lower
            or "ln" in lower
            or "layernorm" in lower
            or "batchnorm" in lower
        )

    head_decay = []
    head_no_decay = []
    backbone_decay_infos = []
    backbone_no_decay_infos = []

    for name, param in model.named_parameters():
        is_head = any(key in name.lower() for key in ["head", "fc", "classifier"])
        no_wd = _no_weight_decay(name)
        layer_id = 0 if is_head else infer_layer_id(name)

        if is_head:
            (head_no_decay if no_wd else head_decay).append(param)
        else:
            target = backbone_no_decay_infos if no_wd else backbone_decay_infos
            target.append((param, layer_id))

    param_groups = []

    def _add_backbone_groups(
        infos: List[Tuple[torch.nn.Parameter, int]], wd: float
    ) -> None:
        if not infos:
            return
        max_lid = max(lid for _, lid in infos)
        buckets: Dict[float, List[torch.nn.Parameter]] = {}
        for param, lid in infos:
            layer_scale = llrd ** (max_lid - lid)
            layer_lr = round(float(lr_backbone * layer_scale), 12)
            buckets.setdefault(layer_lr, []).append(param)
        for lr_val, params in sorted(buckets.items()):
            param_groups.append({"params": params, "lr": float(lr_val), "weight_decay": wd})

    _add_backbone_groups(backbone_decay_infos, weight_decay)
    _add_backbone_groups(backbone_no_decay_infos, 0.0)

    if head_decay:
        param_groups.append({"params": head_decay, "lr": lr_head, "weight_decay": weight_decay})
    if head_no_decay:
        param_groups.append({"params": head_no_decay, "lr": lr_head, "weight_decay": 0.0})

    optimizer_name = optimizer_name.lower()
    if optimizer_name == "adamw":
        return torch.optim.AdamW(param_groups, weight_decay=weight_decay)
    if optimizer_name == "sgd":
        return torch.optim.SGD(param_groups, lr=lr_backbone, momentum=0.9, nesterov=True, weight_decay=weight_decay)

    raise ValueError(f"Unsupported optimizer: {optimizer_name}")


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


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_name: str,
    epochs: int,
    warmup_epochs: int,
):
    name = scheduler_name.lower()
    if name == "cosine":
        return build_cosine_scheduler(optimizer=optimizer, epochs=epochs, warmup_epochs=warmup_epochs)
    if name == "step":
        step_size = max(1, epochs // 3)
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)
    if name == "none":
        return None

    raise ValueError(f"Unsupported scheduler: {scheduler_name}")


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
