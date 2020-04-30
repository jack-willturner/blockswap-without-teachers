#!/bin/bash
#SBATCH --nodes=1
#SBATCH --requeue
#SBATCH --output=logs/blockswap1.out
#SBATCH --job-name=blockswap
#SBATCH --gres=gpu:1
#SBATCH --mem=14000
#SBATCH --time=10000

export PATH="$HOME/miniconda/bin:$PATH"

cd ..
nvidia-smi

source activate bertie
echo "bertie activated\n"

for i in 1 2 3
do
  python train.py --wrn_depth 40 --wrn_width 2 --data_loc='../datasets/cifar' --checkpoint bs150k_$i --GPU 0 --seed $i --from_genotype './genotypes/bs150k.csv' &
  python train.py --wrn_depth 40 --wrn_width 2 --data_loc='../datasets/cifar' --checkpoint bs200k_$i --GPU 1 --seed $i --from_genotype './genotypes/bs200k.csv' &
  python train.py --wrn_depth 40 --wrn_width 2 --data_loc='../datasets/cifar' --checkpoint bs250k_$i --GPU 2 --seed $i --from_genotype './genotypes/bs250k.csv' &
  python train.py --wrn_depth 40 --wrn_width 2 --data_loc='../datasets/cifar' --checkpoint bs400k_$i --GPU 3 --seed $i --from_genotype './genotypes/bs400k.csv'

  python train.py --wrn_depth 40 --wrn_width 2 --data_loc='../datasets/cifar' --checkpoint bs500k_$i --GPU 0 --seed $i --from_genotype './genotypes/bs500k.csv' &
  python train.py --wrn_depth 40 --wrn_width 2 --data_loc='../datasets/cifar' --checkpoint bs800k_$i --GPU 1 --seed $i --from_genotype './genotypes/bs800k.csv' &
  python train.py --wrn_depth 40 --wrn_width 2 --data_loc='../datasets/cifar' --checkpoint bs1.6m_$i --GPU 2 --seed $i --from_genotype './genotypes/bs1.6m.csv'
done
