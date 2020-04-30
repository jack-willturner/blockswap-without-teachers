#!/bin/bash
#SBATCH --nodes=1
#SBATCH --requeue
#SBATCH --output=logs/snip.out
#SBATCH --job-name=snip
#SBATCH --gres=gpu:1
#SBATCH --mem=14000
#SBATCH --time=10000

export PATH="$HOME/miniconda/bin:$PATH"

cd ..
nvidia-smi

source activate bertie
echo "bertie activated\n"

python prune.py
