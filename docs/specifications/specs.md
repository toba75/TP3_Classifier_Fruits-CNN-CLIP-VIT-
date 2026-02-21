# Spécification d’implémentation — TP3 Flowers Classifier

## 1) Périmètre et objectif
Cette spécification définit une implémentation **à deux modèles** conforme au sujet TP3 :
- un classifieur **CNN** basé sur **ConvNeXt V2**,
- un classifieur **ViT** basé sur **CLIP (ViT-L/14)**.

L’API de rendu doit rester strictement compatible :
- `load_cnn_model(weights_path='cnn_flowers.pth')`
- `load_vit_model(weights_path='vit_flowers.pth')`
- `predict(model, image_path) -> str`

Classes cibles (évaluation) :
- `daisy`, `dandelion`, `rose`, `sunflower`, `tulip`

---

## 2) Contraintes fonctionnelles
- Inférence offline (pas de téléchargement réseau dans `load_*`/`predict`).
- Chargement CPU-safe (`map_location='cpu'`).
- `predict` retourne exclusivement une des 5 classes.
- Les poids `.pth` sont fournis avec le rendu.

---

## 3) Pipeline données (commun CNN + ViT)

### 3.1 Split et validation
- Split recommandé : **stratifié** train/val (ex. 80/20).
- Fixer une seed globale (Python/NumPy/PyTorch) pour la reproductibilité.
- Early stopping piloté par la meilleure `val_accuracy` (patience 8–12).

### 3.2 Prétraitement inférence (déterministe)
- Convertir en RGB.
- Resize (taille selon backbone), puis center crop.
- Normalisation ImageNet / CLIP selon le modèle.

### 3.3 Augmentation train (SOTA petits datasets)
- `RandomResizedCrop`
- `RandomHorizontalFlip`
- `RandAugment`
- `ColorJitter`
- Régularisation de lot : **Mixup** + **CutMix** (avec probabilité contrôlée)
- Option : Random Erasing légère

---

## 4) Modèle CNN — ConvNeXt V2

### 4.1 Architecture
- Backbone : `convnextv2_{tiny|base}` pré-entraîné ImageNet-1k/22k.
- Remplacer la tête par `Linear(in_features, 5)`.
- Export du meilleur checkpoint en `cnn_flowers.pth`.

### 4.2 Recette d’entraînement SOTA (prescriptive)
- Optimiseur : **AdamW**
- Weight decay : 0.02–0.05
- LR de base : `1e-3` (à scaler selon batch)
- Scheduler : **cosine decay** avec **warmup** (5–10 epochs)
- Loss : `CrossEntropy` + **label smoothing** (0.05–0.1)
- EMA des poids : activée (fortement recommandée)
- AMP (`torch.cuda.amp`) : activé
- Gradient clipping : 1.0
- Stochastic depth / DropPath : activé (si supporté par le backbone)

### 4.3 Stratégie de fine-tuning
1. **Phase A (stabilisation)** : geler partiellement le backbone 2–5 epochs (tête seule ou derniers blocs).
2. **Phase B (full fine-tune)** : dégel complet + LR plus bas.
3. Garder le meilleur modèle sur `val_accuracy`.

---

## 5) Modèle ViT — CLIP (ViT-L/14)

### 5.1 Architecture
- Encodeur image : **CLIP ViT-L/14** pré-entraîné (OpenAI / OpenCLIP).
- Tête classification 5 classes ajoutée sur les features image.
- Export du meilleur checkpoint en `vit_flowers.pth`.

### 5.2 Recette d’entraînement SOTA (prescriptive)
- Optimiseur : **AdamW**
- LR tête : `1e-3` à `5e-4`
- LR backbone : `1e-5` à `5e-6` (très inférieur à la tête)
- Scheduler : **cosine** + warmup 5–10%
- Loss : `CrossEntropy` + label smoothing (0.05)
- **Layer-wise LR decay** (LLRD) : activé
- AMP : activé
- Gradient clipping : 1.0
- EMA : recommandée

### 5.3 Stratégie recommandée en 2 étapes (SOTA en low-data)
1. **Linear probing** : backbone CLIP gelé, entraînement de la tête seule.
2. **Full/partial fine-tuning** : dégel des derniers blocs puis éventuellement dégel complet avec LR très bas.

### 5.4 Option avancée (si budget limité)
- Fine-tuning paramètre-efficace (LoRA/adapters) sur les blocs attention MLP,
- puis léger full fine-tune final si la validation progresse.

---

## 6) Inférence et calibration
- `predict` applique le preprocessing correspondant au type de modèle.
- Retourner `argmax(softmax(logits))` mappé vers `CLASSES`.
- Option recommandée tournoi : **TTA légère** (flip/crop) côté validation interne uniquement.

---

## 7) Protocole expérimental minimal
- Exécuter au moins 3 seeds par modèle.
- Logger : `val_accuracy`, `train_loss`, `val_loss`, confusion matrix.
- Conserver :
  - meilleur checkpoint par seed,
  - meilleur checkpoint global par modèle.

---

## 8) Critères d’acceptation
- API demandée disponible et importable.
- `load_cnn_model` charge ConvNeXt V2 + tête 5 classes.
- `load_vit_model` charge CLIP ViT-L/14 + tête 5 classes.
- `predict` retourne un `str` valide dans `CLASSES`.
- Les fichiers `cnn_flowers.pth` et `vit_flowers.pth` sont fournis.

---

## 9) Valeurs de départ recommandées

### CNN ConvNeXt V2
- Image size : 224
- Batch size : 32 (adapter selon VRAM)
- Epochs : 30–60
- LR : 1e-3 (tête), 2e-4 (full FT)

### CLIP ViT-L/14
- Image size : 224 ou 336 (selon variante)
- Batch size : 16 (ou accumulation)
- Epochs : 15–40
- LR : 1e-3 (tête), 1e-5 (backbone)

Ces valeurs servent de baseline et doivent être ajustées via validation.
