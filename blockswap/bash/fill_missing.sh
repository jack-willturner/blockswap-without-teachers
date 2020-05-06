#!/bin/bash
#SBATCH --nodes=1
#SBATCH --output=logs/blockswap3.out
#SBATCH --job-name=bs3
#SBATCH --gres=gpu:1
#SBATCH --mem 14000
#SBATCH --time=10000

export PATH="$HOME/miniconda/bin:$PATH"

cd ..
nvidia-smi

source activate bertie

python train.py --wrn_depth 40 --wrn_width 2 --data_loc='../datasets/cifar' --checkpoint bs800k_3 --seed 3 --from_genotype './genotypes/bs800k.csv'

python train.py --wrn_depth 40 --wrn_width 2 --data_loc='../datasets/cifar' --checkpoint bs500k_3 --seed 3 --from_genotype './genotypes/bs500k.csv'

