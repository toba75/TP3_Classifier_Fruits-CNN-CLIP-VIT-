# Plan d’expériences exécutable — TP3 (ConvNeXt V2 + CLIP ViT-L/14)

## 1) Objectif
Planifier des expériences **exécutables et ordonnées** pour sélectionner :
- 1 meilleur modèle CNN (ConvNeXt V2),
- 1 meilleur modèle ViT (CLIP ViT-L/14),
avec reproductibilité (multi-seed) et livraison finale des poids :
- `cnn_flowers.pth`
- `vit_flowers.pth`

---

## 2) Contrat CLI requis (pour exécution)
Le plan suppose un script unique :

```bash
python train.py \
  --model {convnextv2_large,clip_vitl14} \
  --train-dir data/train \
  --val-split 0.2 \
  --seed <int> \
  --epochs <int> \
  --batch-size <int> \
  --img-size <int> \
  --optimizer adamw \
  --lr-head <float> \
  --lr-backbone <float> \
  --weight-decay <float> \
  --warmup-epochs <int> \
  --scheduler cosine \
  --label-smoothing <float> \
  --mixup <float> \
  --cutmix <float> \
  --randaugment <int> \
  --random-erasing <float> \
  --amp \
  --ema \
  --grad-clip 1.0 \
  --freeze-backbone-epochs <int> \
  --llrd <float> \
  --output-dir runs/<run_id>
```

Évaluation (optionnel mais recommandé) :

```bash
python eval.py --checkpoint runs/<run_id>/best.pth --split val
```

---

## 3) Règles globales (fixes pour tous les runs)
- Classes : `daisy,dandelion,rose,sunflower,tulip`
- Split : stratifié, `val_split=0.2`
- Métrique primaire : `val_accuracy`
- Métrique secondaire : macro-F1
- Early stopping : patience 10
- Sauvegarde : meilleur checkpoint par run (`best.pth`)
- Logging minimal : hyperparamètres + courbes loss/acc + matrice de confusion

Seeds de référence : `11`, `23`, `47`

---

## 4) Grille d’hyperparamètres

## 4.1 CNN (ConvNeXt V2)
- `model`: `{convnextv2_large}`
- `img_size`: `{224}`
- `batch_size`: `{32, 64}` (si VRAM insuffisante, réduire)
- `epochs`: `45`
- `lr_head`: `{1e-3, 7e-4}`
- `lr_backbone`: `{2e-4, 1e-4}`
- `weight_decay`: `{0.02, 0.05}`
- `label_smoothing`: `{0.05, 0.1}`
- `mixup`: `0.2`
- `cutmix`: `1.0`
- `randaugment`: `9`
- `random_erasing`: `0.1`
- `freeze_backbone_epochs`: `{3, 5}`

## 4.2 ViT (CLIP ViT-L/14)
- `model`: `{clip_vitl14}`
- `img_size`: `{224}` (336 en variante tardive si budget)
- `batch_size`: `{16}`
- `epochs`: `30`
- `lr_head`: `{1e-3, 5e-4}`
- `lr_backbone`: `{1e-5, 5e-6}`
- `weight_decay`: `{0.02, 0.05}`
- `label_smoothing`: `{0.05}`
- `mixup`: `{0.1, 0.2}`
- `cutmix`: `{0.5, 1.0}`
- `randaugment`: `7`
- `random_erasing`: `0.1`
- `freeze_backbone_epochs`: `{8}` (linear probing initial)
- `llrd`: `{0.75, 0.85}`

---

## 5) Ordre exact des runs

## 5.1 Phase A — Smoke tests (bloquants)
Exécuter dans cet ordre, ne pas continuer si un run échoue techniquement.

1. `A01_CNN_SMOKE`
2. `A02_CLIP_SMOKE`

Commandes :

```bash
python train.py --model convnextv2_large --train-dir data/train --val-split 0.2 --seed 11 --epochs 2 --batch-size 16 --img-size 224 --optimizer adamw --lr-head 1e-3 --lr-backbone 2e-4 --weight-decay 0.02 --warmup-epochs 1 --scheduler cosine --label-smoothing 0.05 --mixup 0.0 --cutmix 0.0 --randaugment 0 --random-erasing 0.0 --amp --ema --grad-clip 1.0 --freeze-backbone-epochs 1 --llrd 1.0 --output-dir runs/A01_CNN_SMOKE
python train.py --model clip_vitl14 --train-dir data/train --val-split 0.2 --seed 11 --epochs 2 --batch-size 8 --img-size 224 --optimizer adamw --lr-head 1e-3 --lr-backbone 1e-5 --weight-decay 0.02 --warmup-epochs 1 --scheduler cosine --label-smoothing 0.05 --mixup 0.0 --cutmix 0.0 --randaugment 0 --random-erasing 0.0 --amp --ema --grad-clip 1.0 --freeze-backbone-epochs 2 --llrd 0.85 --output-dir runs/A02_CLIP_SMOKE
```

Critère de passage : entraînement + checkpoint + éval val sans erreur.

---

## 5.2 Phase B — Sélection CNN (coarse)
Exécuter les 6 runs ci-dessous dans l’ordre :

3. `B01_CNN_lr3e4_wd0.02_fb3`
4. `B02_CNN_lr7e4_wd0.05_fb5`
5. `B03_CNN_lr1e3_wd0.02_fb3`
6. `B04_CNN_lr7e4_wd0.05_fb5`
7. `B05_CNN_lr1e3_wd0.05_fb3_ls0.1`
8. `B06_CNN_lr7e4_wd0.02_fb5_ls0.1`

Template commande (adapter par run) :

```bash
python train.py --model <convnext_variant> --train-dir data/train --val-split 0.2 --seed 11 --epochs 45 --batch-size 32 --img-size 224 --optimizer adamw --lr-head <lr_head> --lr-backbone <lr_backbone> --weight-decay <wd> --warmup-epochs 5 --scheduler cosine --label-smoothing <ls> --mixup 0.2 --cutmix 1.0 --randaugment 9 --random-erasing 0.1 --amp --ema --grad-clip 1.0 --freeze-backbone-epochs <freeze_epochs> --llrd 1.0 --output-dir runs/<run_id>
```

Sélection : garder **Top-2 CNN** sur `val_accuracy` (tie-break macro-F1).

---

## 5.3 Phase C — Confirmation CNN multi-seed
Pour les 2 configurations CNN retenues : exécuter les seeds `23` puis `47`.

9. `C01_CNN_top1_seed23`
10. `C02_CNN_top1_seed47`
11. `C03_CNN_top2_seed23`
12. `C04_CNN_top2_seed47`

Sélection finale CNN : meilleure moyenne `val_accuracy` sur seeds `{11,23,47}`.

---

## 5.4 Phase D — Sélection CLIP (coarse)
Exécuter les 6 runs ci-dessous dans l’ordre :

13. `D01_CLIP_lrh1e3_lrb1e5_wd0.02_mix0.1_cut0.5_llrd0.85`
14. `D02_CLIP_lrh5e4_lrb5e6_wd0.05_mix0.2_cut1.0_llrd0.75`
15. `D03_CLIP_lrh1e3_lrb5e6_wd0.02_mix0.2_cut0.5_llrd0.85`
16. `D04_CLIP_lrh5e4_lrb1e5_wd0.05_mix0.1_cut1.0_llrd0.75`
17. `D05_CLIP_lrh1e3_lrb1e5_wd0.05_mix0.2_cut1.0_llrd0.85`
18. `D06_CLIP_lrh5e4_lrb5e6_wd0.02_mix0.1_cut0.5_llrd0.75`

Template commande :

```bash
python train.py --model clip_vitl14 --train-dir data/train --val-split 0.2 --seed 11 --epochs 30 --batch-size 16 --img-size 224 --optimizer adamw --lr-head <lr_head> --lr-backbone <lr_backbone> --weight-decay <wd> --warmup-epochs 3 --scheduler cosine --label-smoothing 0.05 --mixup <mixup> --cutmix <cutmix> --randaugment 7 --random-erasing 0.1 --amp --ema --grad-clip 1.0 --freeze-backbone-epochs 8 --llrd <llrd> --output-dir runs/<run_id>
```

Sélection : garder **Top-2 CLIP** sur `val_accuracy`.

---

## 5.5 Phase E — Confirmation CLIP multi-seed
Pour les 2 configurations CLIP retenues : exécuter seeds `23` puis `47`.

19. `E01_CLIP_top1_seed23`
20. `E02_CLIP_top1_seed47`
21. `E03_CLIP_top2_seed23`
22. `E04_CLIP_top2_seed47`

Sélection finale ViT : meilleure moyenne `val_accuracy` sur `{11,23,47}`.

---

## 5.6 Phase F — Run final de production
23. `F01_CNN_FINAL_FULLTRAIN`
24. `F02_CLIP_FINAL_FULLTRAIN`

Paramètres = configuration gagnante de chaque famille. Réentraîner sur `train+val` (optionnel selon protocole TP) ou conserver le meilleur modèle validé selon la règle du cours.

Exports attendus :
- `cnn_flowers.pth` depuis `runs/F01_CNN_FINAL_FULLTRAIN/best.pth`
- `vit_flowers.pth` depuis `runs/F02_CLIP_FINAL_FULLTRAIN/best.pth`

---

## 6) Tableau de suivi (à remplir)

| Run ID | Modèle | Seed | Statut | Val Acc | Macro-F1 | Checkpoint |
|---|---|---:|---|---:|---:|---|
| A01_CNN_SMOKE | ConvNeXtV2-Large | 11 | TODO | - | - | - |
| A02_CLIP_SMOKE | CLIP ViT-L/14 | 11 | TODO | - | - | - |
| ... | ... | ... | ... | ... | ... | ... |

---

## 7) Règles de décision (strictes)
1. Aucun run suivant si la phase courante n’est pas terminée.
2. Sélection toujours sur `val_accuracy` moyenne multi-seed.
3. En cas d’égalité : macro-F1, puis plus faible variance inter-seed.
4. Le modèle final doit être testable via `load_*` + `predict` sans accès réseau.

---

## 8) Budget compute (indicatif)
- Total : 24 runs
- Si budget réduit : exécuter A + B (4 runs au lieu de 6) + D (4 runs au lieu de 6) + une seule seed de confirmation.
