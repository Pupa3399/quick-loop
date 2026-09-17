#!/usr/bin/env bash
set -euo pipefail

STEPS="${1:-1}"
MODEL_VARIANT="${2:-${SEARCH_GRPO_MODEL:-searchr1_3b}}"
case "$STEPS" in
  1|5|20) ;;
  *) echo "Usage: $0 {1|5|20} [searchr1_3b|searchr1_7b|ouro_r3]" >&2; exit 2 ;;
esac
case "$MODEL_VARIANT" in
  searchr1_3b|searchr1_7b)
    unset VLLM_PLUGINS
    export VLLM_ATTENTION_BACKEND=FLASH_ATTN
    ;;
  ouro_r3)
    export VLLM_PLUGINS=ouro_search
    export VLLM_ATTENTION_BACKEND=FLASH_ATTN
    ;;
  *)
    echo "Unknown model variant: $MODEL_VARIANT" >&2
    echo "Choose searchr1_3b, searchr1_7b, or ouro_r3." >&2
    exit 2
    ;;
esac

EXPERIMENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$EXPERIMENT_DIR/../../.." && pwd)"
MODEL_ROOT="$PROJECT_ROOT/main/models"
RAY_TEMP_ROOT="${SEARCH_GRPO_RAY_TEMP_ROOT:-$PROJECT_ROOT/.venv/runtime/ray-search-grpo/$MODEL_VARIANT}"
HF_ENDPOINT="https://hf-mirror.com"
PYPI_INDEX="https://mirrors.aliyun.com/pypi/simple/"
export HF_ENDPOINT PYPI_INDEX
export HF_HOME="${HF_HOME:-$MODEL_ROOT/huggingface}"
export HF_MODULES_CACHE="${HF_MODULES_CACHE:-$HF_HOME/modules}"
export PYTHONPATH="$PROJECT_ROOT/.venv/src/verl:$HF_MODULES_CACHE${PYTHONPATH:+:$PYTHONPATH}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$PROJECT_ROOT/.venv/cache/uv}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_DIR="${WANDB_DIR:-$EXPERIMENT_DIR/results/$MODEL_VARIANT/wandb}"
export VLLM_USE_V1=1
# Ray 2.49's uv hook crashes on veRL's intentional runtime_env working_dir=None.
# Workers share this local environment, so disable only uv environment propagation.
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
PYTHON_DEV_ROOT="$PROJECT_ROOT/.venv/toolchain/python-dev"
export C_INCLUDE_PATH="$PYTHON_DEV_ROOT/usr/include/python3.10:$PYTHON_DEV_ROOT/usr/include"
export LIBRARY_PATH="$PYTHON_DEV_ROOT/usr/lib/x86_64-linux-gnu"

TRAIN_GPU_IDS="${SEARCH_GRPO_TRAIN_GPU_IDS:-0,1,2,3,4,5,6,7}"
IFS=',' read -r -a TRAIN_GPUS <<< "$TRAIN_GPU_IDS"
GPU_COUNT="${#TRAIN_GPUS[@]}"
RUN_NAME="${SEARCH_GRPO_RUN_NAME:-$MODEL_VARIANT-$STEPS-step}"
MIN_FREE_MIB="${SEARCH_GRPO_MIN_FREE_MIB:-70000}"
if (( GPU_COUNT < 1 )); then
  echo "SEARCH_GRPO_TRAIN_GPU_IDS must name at least one GPU." >&2
  exit 2
fi

if [[ "${SEARCH_GRPO_REQUIRE_IDLE_GPUS:-1}" == "1" ]]; then
  for gpu in "${TRAIN_GPUS[@]}"; do
    free_memory="$(nvidia-smi --id="$gpu" --query-gpu=memory.free --format=csv,noheader,nounits)"
    if (( free_memory < MIN_FREE_MIB )); then
      echo "GPU $gpu has only $free_memory MiB free; refusing to disturb another job." >&2
      exit 2
    fi
  done
fi
export CUDA_VISIBLE_DEVICES="$TRAIN_GPU_IDS"
export SEARCH_GRPO_TRAJECTORY_DIR="$PROJECT_ROOT/main/data/trajectories/search_grpo/$MODEL_VARIANT/$RUN_NAME"

curl --fail --silent --show-error http://127.0.0.1:8000/health >/dev/null
mkdir -p \
  "$RAY_TEMP_ROOT" \
  "$EXPERIMENT_DIR/results/$MODEL_VARIANT" \
  "$PROJECT_ROOT/main/checkpoints/search_grpo/$MODEL_VARIANT/$RUN_NAME" \
  "$SEARCH_GRPO_TRAJECTORY_DIR"

cd "$PROJECT_ROOT"
uv run --no-sync python -m verl.trainer.main_ppo \
  --config-path="$EXPERIMENT_DIR" \
  --config-name=config \
  experiment_model="$MODEL_VARIANT" \
  trainer.total_training_steps="$STEPS" \
  trainer.n_gpus_per_node="$GPU_COUNT" \
  trainer.default_local_dir="$PROJECT_ROOT/main/checkpoints/search_grpo/$MODEL_VARIANT/$RUN_NAME" \
  trainer.rollout_data_dir="$SEARCH_GRPO_TRAJECTORY_DIR" \
  trainer.save_freq="$STEPS" \
  trainer.test_freq=-1 \
  trainer.val_before_train=false \
  trainer.logger='[console,wandb]' \
  +ray_kwargs.ray_init._temp_dir="$RAY_TEMP_ROOT" \
  data.train_batch_size="$GPU_COUNT" \
  data.val_batch_size="$GPU_COUNT" \
  actor_rollout_ref.actor.ppo_mini_batch_size="$GPU_COUNT" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.agent.num_workers="$GPU_COUNT" \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  2>&1 | tee "$EXPERIMENT_DIR/results/$MODEL_VARIANT/$RUN_NAME.log"
