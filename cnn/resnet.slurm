#!/bin/bash
#SBATCH -p a100
#SBATCH -J resnet
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00
#SBATCH -o cnn/resnet.out.log
#SBATCH -e cnn/resnet.err.log

python cnn/resnet.py --epochs 20