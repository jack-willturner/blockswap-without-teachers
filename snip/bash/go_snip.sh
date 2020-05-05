#!/bin/bash
#SBATCH --nodes=1
#SBATCH --requeue
#SBATCH --output=logs/snip.out
#SBATCH --job-name=snip
#SBATCH --gres=gpu:4
#SBATCH --mem=56000
#SBATCH --time=10000

export PATH="$HOME/miniconda/bin:$PATH"

cd ..
nvidia-smi

source activate bertie
echo "bertie activated\n"

for i in 1 2 3
do
  python prune.py --seed $i --num 0
  python prune.py --seed $i --num 1
  python prune.py --seed $i --num 2
  python prune.py --seed $i --num 3
  python prune.py --seed $i --num 4
  python prune.py --seed $i --num 5
done

for i in 1 2 3
do
  python train.py --seed $i --num 0 --GPU 0 &
  python train.py --seed $i --num 1 --GPU 1 &
  python train.py --seed $i --num 2 --GPU 2 &
  python train.py --seed $i --num 3 --GPU 3
  python train.py --seed $i --num 4 --GPU 0 &
  python train.py --seed $i --num 5 --GPU 1
done
