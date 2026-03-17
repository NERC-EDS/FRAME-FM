#!/bin/bash
#SBATCH --job-name=era5_mmmae_short_gpu
#SBATCH --output=era5_mmmae_short_gpu.out
#SBATCH --gpus=1
#SBATCH --time=00:25:00

set -euo pipefail

# Log GPU utilization while the Python job runs.
LOG="gpu_util_${SLURM_JOB_ID}.csv"
echo "timestamp,util.gpu %,util.mem %,mem.used MiB,mem.total MiB,temp C,power W" > "$LOG"

# Start a background sampler (5-second cadence) for the lifetime of the job
( while true; do
      printf "%s," "$(date --iso-8601=seconds)"
      nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw \
                 --format=csv,noheader,nounits
      sleep 5
  done ) >> "$LOG" 2>/dev/null &

LOGGER_PID=$!
trap 'kill "$LOGGER_PID" 2>/dev/null || true; wait "$LOGGER_PID" 2>/dev/null || true' EXIT

source .venv/bin/activate
export HYDRA_FULL_ERROR=1
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

srun python src/FRAME_FM/training/train.py \
  data=era5_spatial_pixels_demo \
  data.model_ready_inputs=true \
  model=era5_demo_mmmae \
  trainer=default \
  trainer.max_epochs=1 \
  +trainer.max_steps=20 \
  +trainer.limit_train_batches=20 \
  +trainer.limit_val_batches=5
