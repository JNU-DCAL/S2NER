#!/usr/bin/env bash
# Reconstructed paper commands; does not claim a new full training run.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if (( $# > 1 )); then
  echo "Usage: $0 [all|sillokbert|classical_chinese|sikuroberta]" >&2; exit 2
fi
case "${1:-all}" in
  all) MODELS=(sikuroberta sillokbert classical_chinese) ;;
  sillokbert|classical_chinese|sikuroberta) MODELS=("$1") ;;
  *) echo "Unknown model: $1" >&2; exit 2 ;;
esac
: "${DATA_DIR:?Set DATA_DIR to the directory containing the three BIO JSONL files}"
: "${GPUS:?Set GPUS to four idle GPU indices, e.g. 0,1,2,3}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$SCRIPT_DIR/outputs}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DRY_RUN="${DRY_RUN:-0}"
[[ "$DRY_RUN" == 0 || "$DRY_RUN" == 1 ]] || { echo 'DRY_RUN must be 0 or 1' >&2; exit 2; }
IFS=',' read -r -a GPU_IDS <<< "$GPUS"
if [[ ! "$GPUS" =~ ^[0-9]+,[0-9]+,[0-9]+,[0-9]+$ ]]; then
  echo 'Paper protocol requires exactly four GPU indices.' >&2; exit 2
fi
declare -A USED=()
for gpu in "${GPU_IDS[@]}"; do
  if [[ -n "${USED[$gpu]:-}" ]]; then
    echo "Duplicate GPU: $gpu" >&2; exit 2
  fi
  USED[$gpu]=1
done
for file in Sillok_train_final.jsonl Sillok_dev_final.jsonl SJW.jsonl; do
  test -f "$DATA_DIR/$file" || { echo "Missing $DATA_DIR/$file" >&2; exit 1; }
done
export CUDA_VISIBLE_DEVICES="$GPUS"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for model in "${MODELS[@]}"; do
  if [[ "$DRY_RUN" == 0 ]]; then
    nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv -i "$GPUS"
    for gpu in "${GPU_IDS[@]}"; do
      used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu")
      used="${used//[[:space:]]/}"
      if [[ ! "$used" =~ ^[0-9]+$ ]] || (( used > 16 )); then
        echo "GPU $gpu is occupied or its state is unknown ($used MiB); aborting." >&2
        exit 1
      fi
    done
  fi
  cmd=("$PYTHON_BIN" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=4
    "$SCRIPT_DIR/train.py" --model "$model"
    --train_jsonl "$DATA_DIR/Sillok_train_final.jsonl"
    --dev_jsonl "$DATA_DIR/Sillok_dev_final.jsonl"
    --test_jsonl "$DATA_DIR/SJW.jsonl"
    --output_dir "$OUTPUT_ROOT/${model}_crf"
    --lr 6e-5 --epochs 4 --train_batch_size 128 --eval_batch_size 512
    --gradient_accumulation_steps 1 --precision bf16 --seed 42
    --max_len 510 --stride 256 --num_workers 8 --seen_reference train+dev)
  printf 'Running: '; printf '%q ' "${cmd[@]}"; printf '\n'
  if [[ "$DRY_RUN" == 0 ]]; then "${cmd[@]}"; fi
done
