#!/bin/bash 
#SBATCH --job-name=bootstrapping_subsample_ped
#SBATCH --output=slurm_out/bootstrapping_subsample_ped.out
#SBATCH --error=slurm_out/bootstrapping_subsample_ped.err
#SBATCH --time=7-00:00:00
#SBATCH --nodes=1
#SBATCH --mem=20GB
#SBATCH --array=0-9
#SBATCH -c 4
#SBATCH -p normal,gpu,nigam

source activate /share/pi/nigam/envs/anaconda/envs/prediction_utils

DATA_PATH="/share/pi/nigam/projects/sepsis/extraction_201003"
COHORT_PATH=$DATA_PATH"/cohort/cohort_cv.parquet"
FEATURES_ROW_ID_MAP_PATH=$DATA_PATH"/merged_features_binary/features_sparse/features_row_id_map.parquet"
N_BOOT=1000
SEED=718

EXPERIMENT_NAME='baseline_subsample_ped_'$SLURM_ARRAY_TASK_ID
python -m sepsis.bootstrapping \
    --data_path=$DATA_PATH \
    --cohort_path=$COHORT_PATH \
    --features_row_id_map_path=$FEATURES_ROW_ID_MAP_PATH \
    --experiment_name=$EXPERIMENT_NAME \
    --n_boot=$N_BOOT \
    --seed=$SEED