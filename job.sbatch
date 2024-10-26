#!/bin/bash

#SBATCH --job-name=nlp
#SBATCH --nodes=1
#SBATCH --partition=gpu_prod_long
#SBATCH --time=24:00:00
#SBATCH --output=logslurms/slurm-%A_%a.out
#SBATCH --error=logslurms/slurm-%A_%a.err

current_dir=pwd
export PATH=$PATH:~/.local/bin

echo "Running on " $(hostname)

source venv/bin/activate

echo "Starting function"
python training.py

if [[ $? != 0 ]]; then
    exit -1
fi
