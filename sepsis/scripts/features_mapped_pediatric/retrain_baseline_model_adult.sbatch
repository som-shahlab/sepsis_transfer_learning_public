#!/bin/bash 
#SBATCH --job-name=retrain_baseline_model_adult
#SBATCH --output=slurm_out/retrain_baseline_model_adult_%a.out
#SBATCH --error=slurm_out/retrain_baseline_model_adult_%a.err
#SBATCH --time=7-00:00:00
#SBATCH --nodes=1
#SBATCH --mem=20GB
#SBATCH -c 5
#SBATCH --array=1-10
#SBATCH -p gpu,nigam
#SBATCH --gres=gpu:1

source activate /share/pi/nigam/envs/anaconda/envs/prediction_utils

EXPERIMENT_NAME='features_mapped_pediatric/baseline_tuning_fold_1_10_adult'
BASE_PATH="/share/pi/nigam/projects/sepsis/extraction_201003/"
TASK="early_sepsis"
CONFIG_EXPERIMENT_NAME='baseline_config'

python -m sepsis.train_model \
    --data_path=$BASE_PATH \
    --features_path=$BASE_PATH'/features_mapped_pediatric/features_sparse/features.gz' \
    --cohort_path=$BASE_PATH'/cohort/cohort_cv.parquet' \
    --vocab_path=$BASE_PATH'/features_mapped_pediatric/vocab/vocab.parquet' \
    --features_row_id_map_path=$BASE_PATH'/features_mapped_pediatric/features_sparse/features_row_id_map.parquet' \
    --config_path=$BASE_PATH'/experiments/'$EXPERIMENT_NAME'/selected_models/config/early_sepsis/best_config.yaml' \
    --experiment_name=$EXPERIMENT_NAME'/retrain_selected_models' \
    --save_checkpoint_path=$BASE_PATH'/experiments/'$EXPERIMENT_NAME'/retrain_selected_models/checkpoints/'$TASK'/'$SLURM_ARRAY_TASK_ID'/checkpoint.chk' \
    --num_workers=5 \
    --data_mode="array" \
    --label_col=$TASK \
    --fold_id=$SLURM_ARRAY_TASK_ID \
    --run_evaluation \
    --run_evaluation_group \
    --run_evaluation_group_standard \
    --eval_attributes "adult_at_admission" \
    --subset_train_attribute="adult_at_admission" \
    --subset_train_group="1"