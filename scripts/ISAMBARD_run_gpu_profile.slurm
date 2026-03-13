#!/bin/bash
#SBATCH --job-name=train_eurosat_gpu
#SBATCH --output=train_eurosat_gpu.out
#SBATCH --gpus=1
#SBATCH --time=00:25:00         # Hours:Mins:Secs

# Log GPU utilization while the Python job runs.
#Query key metrics: utilization.gpu, utilization.memory, memory.used/total, temperature, power.
#These indicate how "busy" the GPU is, whether you're memory-bound, thermal throttling, etc.

LOG="gpu_util_${SLURM_JOB_ID}.csv"
echo "timestamp,util.gpu %,util.mem %,mem.used MiB,mem.total MiB,temp C,power W" > "$LOG"

# Start a background sampler (120-second cadence) for the lifetime of the job
( while true; do
      printf "%s," "$(date --iso-8601=seconds)"
      nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw \
                 --format=csv,noheader,nounits
      sleep 120
  done ) >> "$LOG" 2>/dev/null &

LOGGER_PID=$!

#Run training
source .venv/bin/activate
srun python src/FRAME_FM/training/train.py

#Clean up logger 
kill $LOGGER_PID
wait $LOGGER_PID 2>/dev/null || true

