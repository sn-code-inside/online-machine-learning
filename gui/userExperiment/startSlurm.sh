#!/bin/bash
 
### Vergabe von Ressourcen
#SBATCH --job-name=CH10_Test
#SBATCH --ntasks-per-node=64
#SBATCH --gres=gpu:1
#SBATCH --time=8:00:00
#SBATCH --error=job.%J.err
#SBATCH --output=job.%J.out
#----
#SBATCH --partition=gpu

if [ -z "$1" ]; then
    echo "Usage: $0 <path_to_spot.pkl>"
    exit 1
fi

SPOT_PKL=$1

module load conda

conda activate spot

python startPython.py "$SPOT_PKL"
