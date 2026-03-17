#!/bin/bash
#SBATCH --job-name=era5_mmmae_throughput
#SBATCH --output=era5_mmmae_throughput.out
#SBATCH --gpus=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=00:15:00

set -euo pipefail

LOG="gpu_util_${SLURM_JOB_ID}.csv"
echo "timestamp,util.gpu %,util.mem %,mem.used MiB,mem.total MiB,temp C,power W" > "$LOG"

# Sample frequently so short runs still capture utilization.
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
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

srun --ntasks-per-node=1 python src/FRAME_FM/training/train.py \
  data=era5_spatial_pixels_demo \
  data.model_ready_inputs=true \
  data.debug=false \
  data.batch_size=8 \
  data.num_workers=24 \
  data.persistent_workers=true \
  data.pin_memory=true \
  model=era5_demo_mmmae \
  trainer=default \
  trainer.max_epochs=1000 \
  trainer.accelerator=gpu \
  trainer.devices=1 \
  trainer.precision=32 \
  trainer.log_every_n_steps=50 \
  +trainer.limit_val_batches=20 \
  +trainer.val_check_interval=20 \
  +trainer.num_sanity_val_steps=0 \
  +trainer.benchmark=true \
  +trainer.max_time=00:00:10:00