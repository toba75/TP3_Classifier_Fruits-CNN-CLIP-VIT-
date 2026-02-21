from __future__ import annotations

from pathlib import Path

import torch

from training_utils import CLASSES, build_model, build_transform


def _load_model_from_checkpoint(weights_path: str, fallback_model_name: str):
    checkpoint_path = Path(weights_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint introuvable: {weights_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model_name = checkpoint.get("model_name", fallback_model_name)
    img_size = int(checkpoint.get("img_size", 224))

    model = build_model(model_name=model_name, num_classes=len(CLASSES), pretrained=False)
    model.load_state_dict(checkpoint["state_dict"], strict=False)
    model.eval()

    model._model_name = model_name
    model._img_size = img_size
    model._transform = build_transform(model_name=model_name, img_size=img_size, train=False)
    return model


def load_cnn_model(weights_path: str = "cnn_flowers.pth"):
    """Charge et retourne le modèle CNN entraîné."""
    return _load_model_from_checkpoint(weights_path=weights_path, fallback_model_name="convnextv2_tiny")


def load_vit_model(weights_path: str = "vit_flowers.pth"):
    """Charge et retourne le modèle ViT entraîné."""
    return _load_model_from_checkpoint(weights_path=weights_path, fallback_model_name="clip_vitl14")


def predict(model, image_path: str) -> str:
    """Prédit la classe d'une image. Retourne un str parmi CLASSES."""
    from PIL import Image

    transform = getattr(model, "_transform", build_transform(model_name="convnextv2_tiny", img_size=224, train=False))

    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        logits = model(tensor)
        pred_idx = int(torch.argmax(logits, dim=1).item())

    return CLASSES[pred_idx]
