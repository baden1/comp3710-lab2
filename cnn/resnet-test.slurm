#!/bin/bash
#SBATCH -p a100-test
#SBATCH -J resnet_job
#SBATCH --gres=shard:3
#SBATCH -o cnn/resnet.out.log
#SBATCH -e cnn/resnet.err.log

python cnn/resnet.py