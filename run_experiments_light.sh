#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# TP3 Experiments (LIGHT VERSION)
# - Runs phases A -> F in order but with fewer configurations
# - Reduces total runs from 24 to ~12
# ============================================================

# Fix: Force ROOT_DIR to /workspace if running inside container
if [[ -d "/workspace" ]]; then
  ROOT_DIR="/workspace"
else
  ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
fi
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-train.py}"
EVAL_SCRIPT="${EVAL_SCRIPT:-eval.py}"
TRAIN_DIR="${TRAIN_DIR:-data/train}"
VAL_SPLIT="${VAL_SPLIT:-0.2}"
EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-4}"
EARLY_STOPPING_MIN_DELTA="${EARLY_STOPPING_MIN_DELTA:-0.001}"

SEED_MAIN=11
SEED_2=23
# Removed SEED_3 (47) to save compute

mkdir -p runs

log() {
  printf "[%s] %s\n" "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

require_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo "ERROR: missing required file: $path" >&2
    exit 1
  fi
}

score_of_run() {
  local run_id="$1"
  local run_dir="runs/$run_id"

  "$PYTHON_BIN" - "$run_dir" <<'PY'
import json
import os
import sys
import csv

run_dir = sys.argv[1]

json_candidates = [
    os.path.join(run_dir, "metrics.json"),
    os.path.join(run_dir, "summary.json"),
    os.path.join(run_dir, "results.json"),
    os.path.join(run_dir, "val_metrics.json"),
    os.path.join(run_dir, "eval.json"),
]

keys = [
    "best_val_accuracy",
    "val_accuracy",
    "best_accuracy",
    "accuracy",
    "acc",
]

def extract_from_dict(d):
    if not isinstance(d, dict):
        return None
    for k in keys:
        if k in d and isinstance(d[k], (int, float)):
            return float(d[k])
    if "metrics" in d and isinstance(d["metrics"], dict):
        for k in keys:
            if k in d["metrics"] and isinstance(d["metrics"][k], (int, float)):
                return float(d["metrics"][k])
    return None

for path in json_candidates:
    if os.path.isfile(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            v = extract_from_dict(data)
            if v is not None:
                print(v)
                sys.exit(0)
        except Exception:
            pass

csv_candidates = [
    os.path.join(run_dir, "metrics.csv"),
    os.path.join(run_dir, "history.csv"),
    os.path.join(run_dir, "results.csv"),
]

for path in csv_candidates:
    if os.path.isfile(path):
        try:
            best = None
            with open(path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    for col in ["val_accuracy", "best_val_accuracy", "accuracy", "acc"]:
                        if col in row and row[col] not in (None, ""):
                            val = float(row[col])
                            best = val if best is None else max(best, val)
            if best is not None:
                print(best)
                sys.exit(0)
        except Exception:
            pass

print("")
PY
}

run_train() {
  local run_id="$1"
  shift

  local run_dir="runs/$run_id"
  mkdir -p "$run_dir"

  log "START $run_id"
  "$PYTHON_BIN" "$TRAIN_SCRIPT" "$@" --output-dir "$run_dir"
  log "DONE  $run_id"
}

rank_top1() {
  # Selects only TOP 1 instead of TOP 2 to save time
  # usage: rank_top1 RUN_ID_1 RUN_ID_2 ...
  local tmp_file
  tmp_file="$(mktemp)"

  local rid score
  for rid in "$@"; do
    score="$(score_of_run "$rid" | tr -d '[:space:]')"
    if [[ -z "$score" ]]; then
      echo "ERROR: Impossible de lire la métrique val_accuracy pour $rid" >&2
      rm -f "$tmp_file"
      exit 1
    fi
    echo "$rid,$score" >> "$tmp_file"
  done

  sort -t, -k2,2gr "$tmp_file" | head -n 1 | cut -d, -f1
  rm -f "$tmp_file"
}

cnn_args_for_config() {
  local cfg="$1"
  local seed="$2"
  case "$cfg" in
    B01)
      echo "--model convnextv2_large --train-dir $TRAIN_DIR --val-split $VAL_SPLIT --seed $seed --epochs 20 --batch-size 32 --img-size 224 --optimizer adamw --lr-head 8e-4 --lr-backbone 8e-5 --weight-decay 0.05 --warmup-epochs 5 --scheduler cosine --label-smoothing 0.1 --mixup 0.2 --cutmix 1.0 --randaugment 9 --random-erasing 0.25 --amp --ema --grad-clip 1.0 --freeze-backbone-epochs 3 --llrd 1.0"
      ;;
    B03)
      echo "--model convnextv2_large --train-dir $TRAIN_DIR --val-split $VAL_SPLIT --seed $seed --epochs 20 --batch-size 32 --img-size 224 --optimizer adamw --lr-head 1e-3 --lr-backbone 1e-4 --weight-decay 0.05 --warmup-epochs 5 --scheduler cosine --label-smoothing 0.1 --mixup 0.2 --cutmix 1.0 --randaugment 9 --random-erasing 0.25 --amp --ema --grad-clip 1.0 --freeze-backbone-epochs 3 --llrd 1.0"
      ;;
    B05)
      echo "--model convnextv2_large --train-dir $TRAIN_DIR --val-split $VAL_SPLIT --seed $seed --epochs 20 --batch-size 32 --img-size 224 --optimizer adamw --lr-head 8e-4 --lr-backbone 8e-5 --weight-decay 0.05 --warmup-epochs 5 --scheduler cosine --label-smoothing 0.1 --mixup 0.3 --cutmix 1.0 --randaugment 10 --random-erasing 0.3 --amp --ema --grad-clip 1.0 --freeze-backbone-epochs 3 --llrd 1.0"
      ;;
    *)
      echo "ERROR: unknown CNN cfg: $cfg" >&2
      exit 1
      ;;
  esac
}

clip_args_for_config() {
  local cfg="$1"
  local seed="$2"
  case "$cfg" in
    D01)
      echo "--model clip_vitl14 --train-dir $TRAIN_DIR --val-split $VAL_SPLIT --seed $seed --epochs 15 --batch-size 16 --img-size 224 --optimizer adamw --lr-head 5e-4 --lr-backbone 5e-6 --weight-decay 0.05 --warmup-epochs 3 --scheduler cosine --label-smoothing 0.05 --mixup 0.1 --cutmix 0.5 --randaugment 7 --random-erasing 0.1 --amp --ema --grad-clip 1.0 --freeze-backbone-epochs 8 --llrd 0.75"
      ;;
    D03)
      echo "--model clip_vitl14 --train-dir $TRAIN_DIR --val-split $VAL_SPLIT --seed $seed --epochs 15 --batch-size 16 --img-size 224 --optimizer adamw --lr-head 7e-4 --lr-backbone 1e-5 --weight-decay 0.05 --warmup-epochs 3 --scheduler cosine --label-smoothing 0.05 --mixup 0.1 --cutmix 0.5 --randaugment 7 --random-erasing 0.1 --amp --ema --grad-clip 1.0 --freeze-backbone-epochs 8 --llrd 0.85"
      ;;
    D05)
      echo "--model clip_vitl14 --train-dir $TRAIN_DIR --val-split $VAL_SPLIT --seed $seed --epochs 15 --batch-size 16 --img-size 224 --optimizer adamw --lr-head 5e-4 --lr-backbone 5e-6 --weight-decay 0.02 --warmup-epochs 3 --scheduler cosine --label-smoothing 0.05 --mixup 0.1 --cutmix 0.5 --randaugment 7 --random-erasing 0.1 --amp --ema --grad-clip 1.0 --freeze-backbone-epochs 8 --llrd 0.75"
      ;;
    *)
      echo "ERROR: unknown CLIP cfg: $cfg" >&2
      exit 1
      ;;
  esac
}

dino_args_for_config() {
  local cfg="$1"
  local seed="$2"
  case "$cfg" in
    G01)
      echo "--model dinov2_vitl14 --train-dir $TRAIN_DIR --val-split $VAL_SPLIT --seed $seed --epochs 15 --batch-size 16 --img-size 224 --optimizer adamw --lr-head 5e-4 --lr-backbone 5e-6 --weight-decay 0.05 --warmup-epochs 3 --scheduler cosine --label-smoothing 0.05 --mixup 0.1 --cutmix 0.5 --randaugment 7 --random-erasing 0.1 --amp --ema --grad-clip 1.0 --freeze-backbone-epochs 5 --llrd 0.75"
      ;;
    G03)
      echo "--model dinov2_vitl14 --train-dir $TRAIN_DIR --val-split $VAL_SPLIT --seed $seed --epochs 15 --batch-size 16 --img-size 224 --optimizer adamw --lr-head 7e-4 --lr-backbone 1e-5 --weight-decay 0.05 --warmup-epochs 3 --scheduler cosine --label-smoothing 0.05 --mixup 0.1 --cutmix 0.5 --randaugment 7 --random-erasing 0.1 --amp --ema --grad-clip 1.0 --freeze-backbone-epochs 5 --llrd 0.85"
      ;;
    G05)
      echo "--model dinov2_vitl14 --train-dir $TRAIN_DIR --val-split $VAL_SPLIT --seed $seed --epochs 15 --batch-size 16 --img-size 224 --optimizer adamw --lr-head 5e-4 --lr-backbone 5e-6 --weight-decay 0.02 --warmup-epochs 3 --scheduler cosine --label-smoothing 0.05 --mixup 0.1 --cutmix 0.5 --randaugment 7 --random-erasing 0.1 --amp --ema --grad-clip 1.0 --freeze-backbone-epochs 5 --llrd 0.75"
      ;;
    *)
      echo "ERROR: unknown DINO cfg: $cfg" >&2
      exit 1
      ;;
  esac
}

run_is_complete() {
  # A run is considered complete if metrics.json exists and contains a val_accuracy
  local run_id="$1"
  local metrics="runs/$run_id/metrics.json"
  [[ -f "$metrics" ]] && grep -q "val_accuracy" "$metrics"
}

run_with_args_string() {
  local run_id="$1"
  local args_str="$2"
  local early_stopping_args="--early-stopping-patience $EARLY_STOPPING_PATIENCE --early-stopping-min-delta $EARLY_STOPPING_MIN_DELTA"

  if run_is_complete "$run_id"; then
    log "SKIP  $run_id (already complete)"
    return 0
  fi

  # Clean up any partial/incomplete run
  if [[ -d "runs/$run_id" ]]; then
    log "CLEAN $run_id (incomplete, restarting)"
    rm -rf "runs/$run_id"
  fi

  # shellcheck disable=SC2086
  run_train "$run_id" $args_str $early_stopping_args
}

log "Checking required scripts"
require_file "$TRAIN_SCRIPT"

# ---------------------------
# Phase A: Smoke tests (SKIPPED)
# ---------------------------
# log "Phase A: smoke tests"
# run_with_args_string "A01_CNN_SMOKE" "--model convnextv2_large --train-dir $TRAIN_DIR --val-split $VAL_SPLIT --seed $SEED_MAIN --epochs 2 --batch-size 16 --img-size 224 --optimizer adamw --lr-head 8e-4 --lr-backbone 8e-5 --weight-decay 0.05 --warmup-epochs 1 --scheduler cosine --label-smoothing 0.1 --mixup 0.2 --cutmix 1.0 --randaugment 9 --random-erasing 0.25 --amp --ema --grad-clip 1.0 --freeze-backbone-epochs 1 --llrd 1.0"
# run_with_args_string "A02_CLIP_SMOKE" "--model clip_vitl14 --train-dir $TRAIN_DIR --val-split $VAL_SPLIT --seed $SEED_MAIN --epochs 2 --batch-size 8 --img-size 224 --optimizer adamw --lr-head 5e-4 --lr-backbone 5e-6 --weight-decay 0.05 --warmup-epochs 1 --scheduler cosine --label-smoothing 0.05 --mixup 0.1 --cutmix 0.5 --randaugment 7 --random-erasing 0.1 --amp --ema --grad-clip 1.0 --freeze-backbone-epochs 2 --llrd 0.75"

# ---------------------------
# Phase B: CNN coarse (Reduced: 3 configs instead of 6)
# ---------------------------
log "Phase B: CNN coarse (LIGHT: B01, B03, B05 only)"
for cfg in B01 B03 B05; do
  run_with_args_string "${cfg}_CNN" "$(cnn_args_for_config "$cfg" "$SEED_MAIN")"
done

log "Selecting Top-1 CNN from phase B (LIGHT MODE)"
CNN_TOP_IDS=( $(rank_top1 B01_CNN B03_CNN B05_CNN) )
if [[ "${#CNN_TOP_IDS[@]}" -ne 1 ]]; then
  echo "ERROR: CNN top-1 selection failed" >&2
  exit 1
fi
CNN_TOP1_CFG="${CNN_TOP_IDS[0]%%_CNN}"
log "CNN top-1: $CNN_TOP1_CFG"

# ---------------------------
# Phase C: CNN multi-seed (Reduced: Only 1 extra seed, only for Top-1)
# ---------------------------
log "Phase C: CNN multi-seed (LIGHT: Only Top-1 on seed 23)"
run_with_args_string "C01_CNN_top1_seed23" "$(cnn_args_for_config "$CNN_TOP1_CFG" "$SEED_2")"

# ---------------------------
# Phase D: CLIP coarse (Reduced: 3 configs instead of 6)
# ---------------------------
log "Phase D: CLIP coarse (LIGHT: D01, D03, D05 only)"
for cfg in D01 D03 D05; do
  run_with_args_string "${cfg}_CLIP" "$(clip_args_for_config "$cfg" "$SEED_MAIN")"
done

log "Selecting Top-1 CLIP from phase D (LIGHT MODE)"
CLIP_TOP_IDS=( $(rank_top1 D01_CLIP D03_CLIP D05_CLIP) )
if [[ "${#CLIP_TOP_IDS[@]}" -ne 1 ]]; then
  echo "ERROR: CLIP top-1 selection failed" >&2
  exit 1
fi
CLIP_TOP1_CFG="${CLIP_TOP_IDS[0]%%_CLIP}"
log "CLIP top-1: $CLIP_TOP1_CFG"

# ---------------------------
# Phase E: CLIP multi-seed (Only 1 extra seed, only for Top-1)
# ---------------------------
log "Phase E: CLIP multi-seed (LIGHT: Only Top-1 on seed 23)"
run_with_args_string "E01_CLIP_top1_seed23" "$(clip_args_for_config "$CLIP_TOP1_CFG" "$SEED_2")"

# ---------------------------
# Phase G: DINOv2 coarse (3 configs, same structure as CLIP)
# ---------------------------
log "Phase G: DINOv2 coarse (LIGHT: G01, G03, G05)"
for cfg in G01 G03 G05; do
  run_with_args_string "${cfg}_DINO" "$(dino_args_for_config "$cfg" "$SEED_MAIN")"
done

log "Selecting Top-1 DINO from phase G (LIGHT MODE)"
DINO_TOP_IDS=( $(rank_top1 G01_DINO G03_DINO G05_DINO) )
if [[ "${#DINO_TOP_IDS[@]}" -ne 1 ]]; then
  echo "ERROR: DINO top-1 selection failed" >&2
  exit 1
fi
DINO_TOP1_CFG="${DINO_TOP_IDS[0]%%_DINO}"
log "DINO top-1: $DINO_TOP1_CFG"

# ---------------------------
# Phase H: DINOv2 multi-seed (Only 1 extra seed, only for Top-1)
# ---------------------------
log "Phase H: DINO multi-seed (LIGHT: Only Top-1 on seed 23)"
run_with_args_string "H01_DINO_top1_seed23" "$(dino_args_for_config "$DINO_TOP1_CFG" "$SEED_2")"

# ---------------------------
# ViT Winner Selection: CLIP vs DINO
# ---------------------------
log "Selecting ViT winner: CLIP ($CLIP_TOP1_CFG) vs DINO ($DINO_TOP1_CFG)"

CLIP_SCORE_1="$(score_of_run "${CLIP_TOP1_CFG}_CLIP" | tr -d '[:space:]')"
CLIP_SCORE_2="$(score_of_run "E01_CLIP_top1_seed23" | tr -d '[:space:]')"
DINO_SCORE_1="$(score_of_run "${DINO_TOP1_CFG}_DINO" | tr -d '[:space:]')"
DINO_SCORE_2="$(score_of_run "H01_DINO_top1_seed23" | tr -d '[:space:]')"

VIT_WINNER=$("$PYTHON_BIN" - "$CLIP_SCORE_1" "$CLIP_SCORE_2" "$DINO_SCORE_1" "$DINO_SCORE_2" <<'PY'
import sys
clip1, clip2, dino1, dino2 = [float(x) for x in sys.argv[1:5]]
clip_mean = (clip1 + clip2) / 2
dino_mean = (dino1 + dino2) / 2
print(f"CLIP mean={clip_mean:.6f}, DINO mean={dino_mean:.6f}", file=sys.stderr)
print("dino" if dino_mean > clip_mean else "clip")
PY
)
log "ViT winner: $VIT_WINNER"

# ---------------------------
# Phase F: Final production runs
# ---------------------------
log "Phase F: final production runs"
CNN_FINAL_CFG="$CNN_TOP1_CFG"

if [[ "$VIT_WINNER" == "dino" ]]; then
  VIT_FINAL_CFG="$DINO_TOP1_CFG"
  log "ViT final model: DINOv2 config $VIT_FINAL_CFG"
  run_with_args_string "F01_CNN_FINAL_FULLTRAIN" "$(cnn_args_for_config "$CNN_FINAL_CFG" "$SEED_MAIN")"
  run_with_args_string "F02_VIT_FINAL_FULLTRAIN" "$(dino_args_for_config "$VIT_FINAL_CFG" "$SEED_MAIN")"
else
  VIT_FINAL_CFG="$CLIP_TOP1_CFG"
  log "ViT final model: CLIP config $VIT_FINAL_CFG"
  run_with_args_string "F01_CNN_FINAL_FULLTRAIN" "$(cnn_args_for_config "$CNN_FINAL_CFG" "$SEED_MAIN")"
  run_with_args_string "F02_VIT_FINAL_FULLTRAIN" "$(clip_args_for_config "$VIT_FINAL_CFG" "$SEED_MAIN")"
fi

# Export expected names
if [[ -f runs/F01_CNN_FINAL_FULLTRAIN/best.pth ]]; then
  cp runs/F01_CNN_FINAL_FULLTRAIN/best.pth cnn_flowers.pth
  log "Exported cnn_flowers.pth"
fi
if [[ -f runs/F02_VIT_FINAL_FULLTRAIN/best.pth ]]; then
  cp runs/F02_VIT_FINAL_FULLTRAIN/best.pth vit_flowers.pth
  log "Exported vit_flowers.pth"
fi

log "All experiment phases completed successfully (LIGHT MODE)"
