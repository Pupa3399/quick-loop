#!/usr/bin/env bash
set -euo pipefail

STEPS="${1:-1}"
case "$STEPS" in
  1|5|20) ;;
  *) echo "Usage: $0 {1|5|20}" >&2; exit 2 ;;
esac

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="/data2/wuguanting/quick_loop/data"
RAY_TEMP_ROOT="${OURO_RAY_TEMP_ROOT:-/data2/wuguanting/ray-ouro}"
HF_ENDPOINT="https://hf-mirror.com"
PYPI_INDEX="https://mirrors.aliyun.com/pypi/simple/"
export HF_ENDPOINT PYPI_INDEX
export HF_HOME="${HF_HOME:-$DATA_ROOT/huggingface}"
export HF_MODULES_CACHE="${HF_MODULES_CACHE:-$HF_HOME/modules}"
export PYTHONPATH="$HF_MODULES_CACHE${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE="${WANDB_MODE:-offline}"
export VLLM_PLUGINS=ouro_search
export VLLM_USE_V1=1
# Ray 2.49's uv hook crashes on veRL's intentional runtime_env working_dir=None.
# Workers share this local environment, so disable only uv environment propagation.
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
export C_INCLUDE_PATH="$DATA_ROOT/python-dev/usr/include/python3.10:$DATA_ROOT/python-dev/usr/include"
export LIBRARY_PATH="$DATA_ROOT/python-dev/usr/lib/x86_64-linux-gnu"

if [[ "${OURO_REQUIRE_IDLE_GPUS:-1}" == "1" ]]; then
  mapfile -t FREE_MEMORY < <(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)
  if [[ "${#FREE_MEMORY[@]}" -ne 8 ]]; then
    echo "Expected 8 GPUs, found ${#FREE_MEMORY[@]}; refusing to start." >&2
    exit 2
  fi
  for gpu in "${!FREE_MEMORY[@]}"; do
    if (( FREE_MEMORY[gpu] < 70000 )); then
      echo "GPU $gpu has only ${FREE_MEMORY[gpu]} MiB free; refusing to disturb another job." >&2
      exit 2
    fi
  done
fi

curl --fail --silent --show-error http://127.0.0.1:8000/health >/dev/null
mkdir -p \
  "$RAY_TEMP_ROOT" \
  "$PROJECT_ROOT/outputs/logs" \
  "$PROJECT_ROOT/outputs/checkpoints/grpo-r3-$STEPS-step"

cd "$PROJECT_ROOT"
uv run --no-sync python -m verl.trainer.main_ppo \
  --config-path="$PROJECT_ROOT/configs/train" \
  --config-name=verl_searchr1_grpo \
  trainer.total_training_steps="$STEPS" \
  trainer.default_local_dir="outputs/checkpoints/grpo-r3-$STEPS-step" \
  trainer.save_freq="$STEPS" \
  trainer.test_freq=-1 \
  trainer.val_before_train=false \
  trainer.logger='[console,wandb]' \
  +ray_kwargs.ray_init._temp_dir="$RAY_TEMP_ROOT" \
  data.train_batch_size=8 \
  data.val_batch_size=8 \
  actor_rollout_ref.actor.ppo_mini_batch_size=8 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  2>&1 | tee "outputs/logs/grpo-r3-$STEPS-step.log"
