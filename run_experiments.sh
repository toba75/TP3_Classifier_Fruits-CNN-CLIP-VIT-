#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# TP3 Experiments Orchestrator
# - Runs phases A -> F in strict order
# - Auto-selects Top-2 CNN after phase B
# - Auto-selects Top-2 CLIP after phase D
# - Re-runs selected configs on seeds 23 and 47
# ============================================================

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-train.py}"
EVAL_SCRIPT="${EVAL_SCRIPT:-eval.py}"
TRAIN_DIR="${TRAIN_DIR:-data/train}"
VAL_SPLIT="${VAL_SPLIT:-0.2}"
EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-10}"
EARLY_STOPPING_MIN_DELTA="${EARLY_STOPPING_MIN_DELTA:-0.001}"

SEED_MAIN=11
SEED_2=23
SEED_3=47

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

rank_top2() {
  # usage: rank_top2 RUN_ID_1 RUN_ID_2 ...
  local tmp_file
  tmp_file="$(mktemp)"

  local rid score
  for rid in "$@"; do
    score="$(score_of_run "$rid" | tr -d '[:space:]')"
    if [[ -z "$score" ]]; then
      echo "ERROR: Impossible de lire la métrique val_accuracy pour $rid (attendu metrics.json/csv)." >&2
      rm -f "$tmp_file"
      exit 1
    fi
    echo "$rid,$score" >> "$tmp_file"
  done

  sort -t, -k2,2gr "$tmp_file" | head -n 2 | cut -d, -f1
  rm -f "$tmp_file"
}

cnn_args_for_config() {
  local cfg="$1"
  local seed="$2"
  case "$cfg" in
    B01)
      echo "--model convnextv2_large --train-dir $TRAIN_DIR --val-split $VAL_SPLIT --seed $seed --epochs 45 --batch-size 32 --img-size 224 --optimizer adamw --lr-head 8e-4 --lr-backbone 8e-5 --weight-decay 0.05 --warmup-epochs 5 --scheduler cosine --label-smoothing 0.1 --mixup 0.2 --cutmix 1.0 --randaugment 9 --random-erasing 0.25 --amp --ema --grad-clip 1.0 --freeze-backbone-epochs 3 --llrd 1.0"
      ;;
    B02)
      echo "--model convnextv2_large --train-dir $TRAIN_DIR --val-split $VAL_SPLIT --seed $seed --epochs 45 --batch-size 32 --img-size 224 --optimizer adamw --lr-head 6e-4 --lr-backbone 6e-5 --weight-decay 0.02 --warmup-epochs 5 --scheduler cosine --label-smoothing 0.1 --mixup 0.1 --cutmix 0.5 --randaugment 9 --random-erasing 0.1 --amp --ema --grad-clip 1.0 --freeze-backbone-epochs 2 --llrd 1.0"
      ;;
    B03)
      echo "--model convnextv2_large --train-dir $TRAIN_DIR --val-split $VAL_SPLIT --seed $seed --epochs 45 --batch-size 32 --img-size 224 --optimizer adamw --lr-head 1e-3 --lr-backbone 1e-4 --weight-decay 0.05 --warmup-epochs 5 --scheduler cosine --label-smoothing 0.1 --mixup 0.2 --cutmix 1.0 --randaugment 9 --random-erasing 0.25 --amp --ema --grad-clip 1.0 --freeze-backbone-epochs 3 --llrd 1.0"
      ;;
    B04)
      echo "--model convnextv2_large --train-dir $TRAIN_DIR --val-split $VAL_SPLIT --seed $seed --epochs 45 --batch-size 32 --img-size 224 --optimizer adamw --lr-head 7e-4 --lr-backbone 5e-5 --weight-decay 0.05 --warmup-epochs 5 --scheduler cosine --label-smoothing 0.1 --mixup 0.2 --cutmix 1.0 --randaugment 9 --random-erasing 0.2 --amp --ema --grad-clip 1.0 --freeze-backbone-epochs 5 --llrd 1.0"
      ;;
    B05)
      echo "--model convnextv2_large --train-dir $TRAIN_DIR --val-split $VAL_SPLIT --seed $seed --epochs 45 --batch-size 32 --img-size 224 --optimizer adamw --lr-head 8e-4 --lr-backbone 8e-5 --weight-decay 0.05 --warmup-epochs 5 --scheduler cosine --label-smoothing 0.1 --mixup 0.3 --cutmix 1.0 --randaugment 10 --random-erasing 0.3 --amp --ema --grad-clip 1.0 --freeze-backbone-epochs 3 --llrd 1.0"
      ;;
    B06)
      echo "--model convnextv2_large --train-dir $TRAIN_DIR --val-split $VAL_SPLIT --seed $seed --epochs 45 --batch-size 32 --img-size 224 --optimizer adamw --lr-head 5e-4 --lr-backbone 5e-5 --weight-decay 0.05 --warmup-epochs 5 --scheduler cosine --label-smoothing 0.05 --mixup 0.1 --cutmix 0.5 --randaugment 7 --random-erasing 0.15 --amp --ema --grad-clip 1.0 --freeze-backbone-epochs 3 --llrd 1.0"
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
      echo "--model clip_vitl14 --train-dir $TRAIN_DIR --val-split $VAL_SPLIT --seed $seed --epochs 30 --batch-size 16 --img-size 224 --optimizer adamw --lr-head 5e-4 --lr-backbone 5e-6 --weight-decay 0.05 --warmup-epochs 3 --scheduler cosine --label-smoothing 0.05 --mixup 0.1 --cutmix 0.5 --randaugment 7 --random-erasing 0.1 --amp --ema --grad-clip 1.0 --freeze-backbone-epochs 8 --llrd 0.75"
      ;;
    D02)
      echo "--model clip_vitl14 --train-dir $TRAIN_DIR --val-split $VAL_SPLIT --seed $seed --epochs 30 --batch-size 16 --img-size 224 --optimizer adamw --lr-head 5e-4 --lr-backbone 5e-6 --weight-decay 0.05 --warmup-epochs 3 --scheduler cosine --label-smoothing 0.05 --mixup 0.0 --cutmix 0.0 --randaugment 5 --random-erasing 0.05 --amp --ema --grad-clip 1.0 --freeze-backbone-epochs 8 --llrd 0.75"
      ;;
    D03)
      echo "--model clip_vitl14 --train-dir $TRAIN_DIR --val-split $VAL_SPLIT --seed $seed --epochs 30 --batch-size 16 --img-size 224 --optimizer adamw --lr-head 7e-4 --lr-backbone 1e-5 --weight-decay 0.05 --warmup-epochs 3 --scheduler cosine --label-smoothing 0.05 --mixup 0.1 --cutmix 0.5 --randaugment 7 --random-erasing 0.1 --amp --ema --grad-clip 1.0 --freeze-backbone-epochs 8 --llrd 0.85"
      ;;
    D04)
      echo "--model clip_vitl14 --train-dir $TRAIN_DIR --val-split $VAL_SPLIT --seed $seed --epochs 30 --batch-size 16 --img-size 224 --optimizer adamw --lr-head 4e-4 --lr-backbone 5e-6 --weight-decay 0.05 --warmup-epochs 3 --scheduler cosine --label-smoothing 0.05 --mixup 0.2 --cutmix 1.0 --randaugment 7 --random-erasing 0.1 --amp --ema --grad-clip 1.0 --freeze-backbone-epochs 8 --llrd 0.75"
      ;;
    D05)
      echo "--model clip_vitl14 --train-dir $TRAIN_DIR --val-split $VAL_SPLIT --seed $seed --epochs 30 --batch-size 16 --img-size 224 --optimizer adamw --lr-head 5e-4 --lr-backbone 5e-6 --weight-decay 0.02 --warmup-epochs 3 --scheduler cosine --label-smoothing 0.05 --mixup 0.1 --cutmix 0.5 --randaugment 7 --random-erasing 0.1 --amp --ema --grad-clip 1.0 --freeze-backbone-epochs 8 --llrd 0.75"
      ;;
    D06)
      echo "--model clip_vitl14 --train-dir $TRAIN_DIR --val-split $VAL_SPLIT --seed $seed --epochs 30 --batch-size 16 --img-size 224 --optimizer adamw --lr-head 3e-4 --lr-backbone 3e-6 --weight-decay 0.05 --warmup-epochs 3 --scheduler cosine --label-smoothing 0.05 --mixup 0.0 --cutmix 0.5 --randaugment 5 --random-erasing 0.05 --amp --ema --grad-clip 1.0 --freeze-backbone-epochs 8 --llrd 0.75"
      ;;
    *)
      echo "ERROR: unknown CLIP cfg: $cfg" >&2
      exit 1
      ;;
  esac
}

run_with_args_string() {
  local run_id="$1"
  local args_str="$2"
  local early_stopping_args="--early-stopping-patience $EARLY_STOPPING_PATIENCE --early-stopping-min-delta $EARLY_STOPPING_MIN_DELTA"

  # shellcheck disable=SC2086
  run_train "$run_id" $args_str $early_stopping_args
}

log "Checking required scripts"
require_file "$TRAIN_SCRIPT"

# ---------------------------
# Phase A: Smoke tests
# ---------------------------
log "Phase A: smoke tests"
run_with_args_string "A01_CNN_SMOKE" "--model convnextv2_large --train-dir $TRAIN_DIR --val-split $VAL_SPLIT --seed $SEED_MAIN --epochs 2 --batch-size 16 --img-size 224 --optimizer adamw --lr-head 8e-4 --lr-backbone 8e-5 --weight-decay 0.05 --warmup-epochs 1 --scheduler cosine --label-smoothing 0.1 --mixup 0.2 --cutmix 1.0 --randaugment 9 --random-erasing 0.25 --amp --ema --grad-clip 1.0 --freeze-backbone-epochs 1 --llrd 1.0"
run_with_args_string "A02_CLIP_SMOKE" "--model clip_vitl14 --train-dir $TRAIN_DIR --val-split $VAL_SPLIT --seed $SEED_MAIN --epochs 2 --batch-size 8 --img-size 224 --optimizer adamw --lr-head 5e-4 --lr-backbone 5e-6 --weight-decay 0.05 --warmup-epochs 1 --scheduler cosine --label-smoothing 0.05 --mixup 0.1 --cutmix 0.5 --randaugment 7 --random-erasing 0.1 --amp --ema --grad-clip 1.0 --freeze-backbone-epochs 2 --llrd 0.75"

# ---------------------------
# Phase B: CNN coarse
# ---------------------------
log "Phase B: CNN coarse"
for cfg in B01 B02 B03 B04 B05 B06; do
  run_with_args_string "${cfg}_CNN" "$(cnn_args_for_config "$cfg" "$SEED_MAIN")"
done

log "Selecting Top-2 CNN from phase B"
CNN_TOP_IDS=( $(rank_top2 B01_CNN B02_CNN B03_CNN B04_CNN B05_CNN B06_CNN) )
if [[ "${#CNN_TOP_IDS[@]}" -ne 2 ]]; then
  echo "ERROR: CNN top-2 selection failed" >&2
  exit 1
fi
CNN_TOP1_CFG="${CNN_TOP_IDS[0]%%_CNN}"
CNN_TOP2_CFG="${CNN_TOP_IDS[1]%%_CNN}"
log "CNN top-1: $CNN_TOP1_CFG | top-2: $CNN_TOP2_CFG"

# ---------------------------
# Phase C: CNN multi-seed
# ---------------------------
log "Phase C: CNN multi-seed"
run_with_args_string "C01_CNN_top1_seed23" "$(cnn_args_for_config "$CNN_TOP1_CFG" "$SEED_2")"
run_with_args_string "C02_CNN_top1_seed47" "$(cnn_args_for_config "$CNN_TOP1_CFG" "$SEED_3")"
run_with_args_string "C03_CNN_top2_seed23" "$(cnn_args_for_config "$CNN_TOP2_CFG" "$SEED_2")"
run_with_args_string "C04_CNN_top2_seed47" "$(cnn_args_for_config "$CNN_TOP2_CFG" "$SEED_3")"

# ---------------------------
# Phase D: CLIP coarse
# ---------------------------
log "Phase D: CLIP coarse"
for cfg in D01 D02 D03 D04 D05 D06; do
  run_with_args_string "${cfg}_CLIP" "$(clip_args_for_config "$cfg" "$SEED_MAIN")"
done

log "Selecting Top-2 CLIP from phase D"
CLIP_TOP_IDS=( $(rank_top2 D01_CLIP D02_CLIP D03_CLIP D04_CLIP D05_CLIP D06_CLIP) )
if [[ "${#CLIP_TOP_IDS[@]}" -ne 2 ]]; then
  echo "ERROR: CLIP top-2 selection failed" >&2
  exit 1
fi
CLIP_TOP1_CFG="${CLIP_TOP_IDS[0]%%_CLIP}"
CLIP_TOP2_CFG="${CLIP_TOP_IDS[1]%%_CLIP}"
log "CLIP top-1: $CLIP_TOP1_CFG | top-2: $CLIP_TOP2_CFG"

# ---------------------------
# Phase E: CLIP multi-seed
# ---------------------------
log "Phase E: CLIP multi-seed"
run_with_args_string "E01_CLIP_top1_seed23" "$(clip_args_for_config "$CLIP_TOP1_CFG" "$SEED_2")"
run_with_args_string "E02_CLIP_top1_seed47" "$(clip_args_for_config "$CLIP_TOP1_CFG" "$SEED_3")"
run_with_args_string "E03_CLIP_top2_seed23" "$(clip_args_for_config "$CLIP_TOP2_CFG" "$SEED_2")"
run_with_args_string "E04_CLIP_top2_seed47" "$(clip_args_for_config "$CLIP_TOP2_CFG" "$SEED_3")"

# ---------------------------
# Pick final winner by mean score
# ---------------------------
mean_of_three() {
  local id1="$1" id2="$2" id3="$3"
  "$PYTHON_BIN" - "$(score_of_run "$id1")" "$(score_of_run "$id2")" "$(score_of_run "$id3")" <<'PY'
import sys
vals = []
for x in sys.argv[1:]:
    x = (x or "").strip()
    if not x:
        print("")
        sys.exit(0)
    vals.append(float(x))
print(sum(vals) / len(vals))
PY
}

log "Selecting final CNN winner"
CNN_TOP1_MEAN="$(mean_of_three "${CNN_TOP1_CFG}_CNN" "C01_CNN_top1_seed23" "C02_CNN_top1_seed47" | tr -d '[:space:]')"
CNN_TOP2_MEAN="$(mean_of_three "${CNN_TOP2_CFG}_CNN" "C03_CNN_top2_seed23" "C04_CNN_top2_seed47" | tr -d '[:space:]')"
if [[ -z "$CNN_TOP1_MEAN" || -z "$CNN_TOP2_MEAN" ]]; then
  echo "ERROR: cannot compute CNN final means" >&2
  exit 1
fi
CNN_FINAL_CFG="$CNN_TOP1_CFG"
"$PYTHON_BIN" - "$CNN_TOP1_MEAN" "$CNN_TOP2_MEAN" <<'PY' >/tmp/cnn_pick.txt
import sys
m1, m2 = map(float, sys.argv[1:])
print("top2" if m2 > m1 else "top1")
PY
if [[ "$(cat /tmp/cnn_pick.txt)" == "top2" ]]; then
  CNN_FINAL_CFG="$CNN_TOP2_CFG"
fi

log "Selecting final CLIP winner"
CLIP_TOP1_MEAN="$(mean_of_three "${CLIP_TOP1_CFG}_CLIP" "E01_CLIP_top1_seed23" "E02_CLIP_top1_seed47" | tr -d '[:space:]')"
CLIP_TOP2_MEAN="$(mean_of_three "${CLIP_TOP2_CFG}_CLIP" "E03_CLIP_top2_seed23" "E04_CLIP_top2_seed47" | tr -d '[:space:]')"
if [[ -z "$CLIP_TOP1_MEAN" || -z "$CLIP_TOP2_MEAN" ]]; then
  echo "ERROR: cannot compute CLIP final means" >&2
  exit 1
fi
CLIP_FINAL_CFG="$CLIP_TOP1_CFG"
"$PYTHON_BIN" - "$CLIP_TOP1_MEAN" "$CLIP_TOP2_MEAN" <<'PY' >/tmp/clip_pick.txt
import sys
m1, m2 = map(float, sys.argv[1:])
print("top2" if m2 > m1 else "top1")
PY
if [[ "$(cat /tmp/clip_pick.txt)" == "top2" ]]; then
  CLIP_FINAL_CFG="$CLIP_TOP2_CFG"
fi

rm -f /tmp/cnn_pick.txt /tmp/clip_pick.txt

# ---------------------------
# Phase F: Final production runs
# ---------------------------
log "Phase F: final production runs"
run_with_args_string "F01_CNN_FINAL_FULLTRAIN" "$(cnn_args_for_config "$CNN_FINAL_CFG" "$SEED_MAIN")"
run_with_args_string "F02_CLIP_FINAL_FULLTRAIN" "$(clip_args_for_config "$CLIP_FINAL_CFG" "$SEED_MAIN")"

# Export expected names
if [[ -f runs/F01_CNN_FINAL_FULLTRAIN/best.pth ]]; then
  cp runs/F01_CNN_FINAL_FULLTRAIN/best.pth cnn_flowers.pth
  log "Exported cnn_flowers.pth"
fi
if [[ -f runs/F02_CLIP_FINAL_FULLTRAIN/best.pth ]]; then
  cp runs/F02_CLIP_FINAL_FULLTRAIN/best.pth vit_flowers.pth
  log "Exported vit_flowers.pth"
fi

log "All experiment phases completed successfully"
