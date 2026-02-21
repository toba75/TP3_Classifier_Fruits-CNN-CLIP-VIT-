# TP3 Flowers Classifier (CNN + CLIP ViT)

Projet de classification de fleurs conforme au TP3, avec deux familles de modèles :
- **CNN** : ConvNeXt V2 (`convnextv2_tiny`, `convnextv2_base`)
- **ViT** : CLIP ViT-L/14 (`clip_vitl14` via OpenCLIP)

Le projet fournit :
- un pipeline d'entraînement/evaluation,
- un orchestrateur d'expériences multi-phases,
- une API d'inférence simple (`load_*`, `predict`),
- un environnement Docker GPU prêt à l'emploi.

---

## 1) Arborescence du projet

- `train.py` : entraînement d'un run (CNN ou CLIP)
- `eval.py` : évaluation d'un checkpoint sur split validation
- `training_utils.py` : datasets, transforms, modèles, optimizer, scheduler, eval
- `run_experiments.sh` : pipeline complet A→F (smoke, coarse search, multi-seed, final)
- `flowers_classifier.py` : API de rendu (`load_cnn_model`, `load_vit_model`, `predict`)
- `extract.py` : extraction `train/` et `test/` depuis un zip de dataset
- `Dockerfile` : image basée sur `nvcr.io/nvidia/pytorch:25.01-py3`
- `requirements.txt` : dépendances Python
- `data/` : dataset local
- `runs/` : sorties d'entraînement (checkpoints, métriques, historique)

---

## 2) Dataset attendu

Le code entraîne sur 5 classes cibles :
- `daisy`, `dandelion`, `rose`, `sunflower`, `tulip`

Structure attendue :

```text
data/
  train/
    daisy/
    dandelion/
    rose/
    sunflower/
    tulip/
  test/
```

### Extraire un zip de dataset

```bash
python3 extract.py --zip-path docs/sujet/fleurs.zip --out-dir data
```

Ou laisser l'auto-détection :

```bash
python3 extract.py
```

---

## 3) Installation locale (sans Docker)

> Recommandé : environnement virtuel Python 3.10+

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

Si vous voulez CUDA en local, installez une version PyTorch compatible GPU (selon votre machine) avant ou à la place de la version CPU.

---

## 4) Docker GPU (recommandé)

Le `Dockerfile` utilise l'image NVIDIA :
- `nvcr.io/nvidia/pytorch:25.01-py3`

Profils Docker machine-spécifiques disponibles :
- `Dockerfile.spark` + `scripts/run_spark.sh`
- `Dockerfile.rtx5090` + `scripts/run_rtx5090.sh`

Le cache des modèles est configuré automatiquement dans `/workspace/.cache`.
Donc, si vous montez le projet avec `-v "$PWD:/workspace"`, les poids téléchargés sont **persistés sur l'hôte** et **réutilisés automatiquement** aux runs suivants.

Caches couverts automatiquement :
- PyTorch / timm (`TORCH_HOME`)
- Hugging Face (`HF_HOME`, `TRANSFORMERS_CACHE`)
- OpenCLIP (`OPENCLIP_CACHE_DIR`)

Le conteneur impose aussi `REQUIRE_CUDA=1` par défaut : l'entraînement/évaluation échoue immédiatement si aucun GPU CUDA n'est disponible.

### Build

```bash
docker build -t tp3-experiments .
```

### Build + run profil NVIDIA Spark

```bash
./scripts/run_spark.sh
```

### Build + run profil PC RTX 5090

```bash
./scripts/run_rtx5090.sh
```

### Run pipeline complet

```bash
docker run --rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "$PWD:/workspace" \
  tp3-experiments bash /workspace/run_experiments.sh
```

Le premier run peut télécharger les poids pré-entraînés (timm/OpenCLIP). Les runs suivants réutilisent le cache local dans `.cache/`.

### Run pipeline complet avec early stopping custom

```bash
docker run --rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -e EARLY_STOPPING_PATIENCE=8 \
  -e EARLY_STOPPING_MIN_DELTA=0.001 \
  -v "$PWD:/workspace" \
  tp3-experiments bash /workspace/run_experiments.sh
```

---

## 5) Entraîner un modèle (commande unitaire)

Exemple ConvNeXt V2 tiny :

```bash
python3 train.py \
  --model convnextv2_tiny \
  --train-dir data/train \
  --val-split 0.2 \
  --seed 11 \
  --epochs 30 \
  --batch-size 16 \
  --img-size 224 \
  --optimizer adamw \
  --lr-head 1e-3 \
  --lr-backbone 2e-4 \
  --weight-decay 0.02 \
  --warmup-epochs 3 \
  --scheduler cosine \
  --label-smoothing 0.05 \
  --mixup 0.2 \
  --cutmix 0.0 \
  --randaugment 0 \
  --random-erasing 0.0 \
  --amp --ema \
  --grad-clip 1.0 \
  --freeze-backbone-epochs 3 \
  --llrd 1.0 \
  --early-stopping-patience 10 \
  --early-stopping-min-delta 0.001 \
  --output-dir runs/MY_RUN
```

Sorties dans `runs/MY_RUN/` :
- `best.pth` : meilleur checkpoint (sur `val_accuracy`)
- `last.pth` : dernier checkpoint
- `history.csv` : historique par epoch
- `metrics.json` : résumé du run

---

## 6) Évaluer un checkpoint

```bash
python3 eval.py \
  --checkpoint runs/MY_RUN/best.pth \
  --train-dir data/train \
  --val-split 0.2 \
  --seed 11 \
  --batch-size 32
```

Sortie : `runs/MY_RUN/eval.json`

---

## 7) Pipeline complet d'expériences (`run_experiments.sh`)

Le script exécute automatiquement :

1. **Phase A** : smoke tests (CNN + CLIP)
2. **Phase B** : grille coarse CNN
3. **Phase C** : multi-seed sur Top-2 CNN
4. **Phase D** : grille coarse CLIP
5. **Phase E** : multi-seed sur Top-2 CLIP
6. **Phase F** : runs finaux (CNN + CLIP)

Puis exporte si présents :
- `cnn_flowers.pth`
- `vit_flowers.pth`

Variables d'environnement supportées :
- `PYTHON_BIN` (défaut `python`)
- `TRAIN_SCRIPT` (défaut `train.py`)
- `TRAIN_DIR` (défaut `data/train`)
- `VAL_SPLIT` (défaut `0.2`)
- `EARLY_STOPPING_PATIENCE` (défaut `10`)
- `EARLY_STOPPING_MIN_DELTA` (défaut `0.001`)

---

## 8) API d'inférence (rendu)

Fichier : `flowers_classifier.py`

Fonctions exposées :
- `load_cnn_model(weights_path='cnn_flowers.pth')`
- `load_vit_model(weights_path='vit_flowers.pth')`
- `predict(model, image_path) -> str`

Exemple :

```python
from flowers_classifier import load_cnn_model, predict

model = load_cnn_model('cnn_flowers.pth')
label = predict(model, 'data/test/example.jpg')
print(label)
```

---

## 9) Lutte contre l'overfitting

Déjà en place dans le code :
- split stratifié train/val,
- augmentation (`RandomResizedCrop`, `RandomHorizontalFlip`, `ColorJitter`),
- label smoothing,
- mixup,
- warmup + scheduler cosinus,
- EMA,
- early stopping (`--early-stopping-patience`, `--early-stopping-min-delta`).

Conseils de réglage :
- CNN : `patience=8`, `min_delta=0.001`
- CLIP : `patience=10`, `min_delta=0.001`

---

## 10) Vérifier que le GPU est bien utilisé

Dans Docker :

```bash
docker run --rm --gpus all tp3-experiments python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
```

Si `False`, vérifier :
- driver NVIDIA installé côté host (`nvidia-smi`),
- Docker avec support GPU (`--gpus all`),
- NVIDIA Container Toolkit correctement installé.

---

## 11) Dépannage

### Fichiers `runs/` non supprimables (permissions)

Quand des runs sont créés par root dans conteneur, vous pouvez nettoyer via conteneur :

```bash
docker run --rm -v "$PWD:/workspace" nvcr.io/nvidia/pytorch:25.01-py3 \
  bash -lc "rm -rf /workspace/runs/A01_CNN_SMOKE /workspace/runs/A02_CLIP_SMOKE"
```

### Warning GB10

Un warning de compatibilité peut apparaître avec certains GPU récents (ex: GB10) selon la version de l'image. Si `torch.cuda.is_available()` est `True` et que les epochs avancent, l'entraînement GPU est opérationnel.

---

## 12) Références

- Spécification fonctionnelle : `docs/specifications/specs.md`
- Plan expérimental : `docs/specifications/experiments_plan.md`
- Sujet TP : `docs/sujet/TP3-FlowersClassifier.md`
