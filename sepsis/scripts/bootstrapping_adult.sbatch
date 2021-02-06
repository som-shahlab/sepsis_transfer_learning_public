#!/bin/bash 
#SBATCH --job-name=bootstrapping_baseline
#SBATCH --output=slurm_out/bootstrapping_baseline.out
#SBATCH --error=slurm_out/bootstrapping_baseline.err
#SBATCH --time=7-00:00:00
#SBATCH --nodes=1
#SBATCH --mem=20GB
#SBATCH -c 4
#SBATCH -p normal,gpu,nigam

source activate /share/pi/nigam/envs/anaconda/envs/prediction_utils

DATA_PATH="/share/pi/nigam/projects/sepsis/extraction_201003"
COHORT_PATH=$DATA_PATH"/cohort/cohort_cv.parquet"
FEATURES_ROW_ID_MAP_PATH=$DATA_PATH"/merged_features_binary/features_sparse/features_row_id_map.parquet"
EXPERIMENT_NAME='baseline_tuning_fold_1_10_adult'
N_BOOT=1000
SEED=718

python -m sepsis.bootstrapping \
    --data_path=$DATA_PATH \
    --cohort_path=$COHORT_PATH \
    --features_row_id_map_path=$FEATURES_ROW_ID_MAP_PATH \
    --experiment_name=$EXPERIMENT_NAME \
    --n_boot=$N_BOOT \
    --seed=$SEED