#!/bin/bash
#SBATCH --job-name=create_cohort
#SBATCH --output=slurm_out/create_cohort.out
#SBATCH --error=slurm_out/create_cohort.err
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --mem=40GB
#SBATCH -c 20
#SBATCH -p normal,gpu

source activate prediction_utils

python -m sepsis.create_cohort \
    --data_path="/share/pi/nigam/projects/sepsis/extraction_201003"