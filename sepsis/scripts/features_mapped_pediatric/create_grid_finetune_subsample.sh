#!/bin/bash

DATA_PATH="/share/pi/nigam/projects/sepsis/extraction_201003/"
BASE_EXPERIMENT_NAME="features_mapped_pediatric/baseline_tuning_fold_1_10_adult"
BASE_CONFIG_PATH=$DATA_PATH'/experiments/'$BASE_EXPERIMENT_NAME'/selected_models/config/early_sepsis/best_config.yaml'
EXPERIMENT_NAME='features_mapped_pediatric/finetune_config'

python -m sepsis.create_grid_finetune \
    --data_path=$DATA_PATH \
    --experiment_name=$EXPERIMENT_NAME \
    --base_config_path=$BASE_CONFIG_PATH \
    --tasks="early_sepsis"