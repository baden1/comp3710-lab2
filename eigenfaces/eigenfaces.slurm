#!/bin/bash
#SBATCH -p a100-test
#SBATCH --gres=shard:4    
#SBATCH -J eigenfaces
#SBATCH -o out.%j.log
#SBATCH -e err.%j.log

python eigenfaces_torch.py