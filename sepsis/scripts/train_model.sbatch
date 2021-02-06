#!/bin/bash
#SBATCH --job-name=train_model_sepsis
#SBATCH --output=slurm_out/train_model_sepsis.out
#SBATCH --error=slurm_out/train_model_sepsis.err
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --mem=20GB
#SBATCH -c 5
#SBATCH -p gpu,nigam
#SBATCH --gres=gpu:1

source activate /share/pi/nigam/envs/anaconda/envs/prediction_utils

EXPERIMENT_NAME="scratch"
DATA_PATH="/share/pi/nigam/projects/sepsis/extraction_201003/"
FEATURES_PATH=$DATA_PATH"/merged_features_binary/features_sparse/features.gz"
COHORT_PATH=$DATA_PATH"/cohort/cohort.parquet"
VOCAB_PATH=$DATA_PATH"/merged_features_binary/vocab/vocab.parquet"
FEATURES_ROW_ID_MAP_PATH=$DATA_PATH"/merged_features_binary/features_sparse/features_row_id_map.parquet"

python -m sepsis.train_model \
    --data_path=$DATA_PATH \
    --features_path=$FEATURES_PATH \
    --cohort_path=$COHORT_PATH \
    --vocab_path=$VOCAB_PATH \
    --features_row_id_map_path=$FEATURES_ROW_ID_MAP_PATH \
    --experiment_name=$EXPERIMENT_NAME \
    --eval_attributes='adult_at_admission'