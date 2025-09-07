#!/bin/bash
#SBATCH -p a100-test
#SBATCH --gres=shard:4    
#SBATCH -J fourier_job
#SBATCH -o out.%j.log
#SBATCH -e err.%j.log

python fourier/fourier.py