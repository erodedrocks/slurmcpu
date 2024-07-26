#!/usr/bin/env sh

#SBATCH --job-name=run-cpu-benchmark
#SBATCH --account=ddp324
#SBATCH --clusters=expanse
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=%x.o%A.%a.%N

cd "${SLURM_SUBMIT_DIR}"
cd ".."

python3 cpu-benchmark.py -n "torch-model-training" -c "16,32" -j 10 -o 4 -i 0.8 -f "efficiency_bound"

echo "Job completed"