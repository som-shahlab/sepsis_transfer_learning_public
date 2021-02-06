#!/bin/bash

BASE_PATH="/share/pi/nigam/projects/sepsis/extraction_201003"
TASK="early_sepsis"
COHORT_NAME='cohort_pediatric_subsample.parquet'
EXPERIMENT_NAME='scratch'
SUBSAMPLE_ID='9'
CONFIG_EXPERIMENT_NAME='baseline_subsample_ped_'$SUBSAMPLE_ID

train_model_func() {    
    python -m sepsis.train_model \
        --data_path=$BASE_PATH \
        --features_path=$BASE_PATH'/merged_features_binary/features_sparse/features.gz' \
        --cohort_path=$BASE_PATH'/cohort/'$COHORT_NAME \
        --vocab_path=$BASE_PATH'/merged_features_binary/vocab/vocab.parquet' \
        --features_row_id_map_path=$BASE_PATH'/merged_features_binary/features_sparse/features_row_id_map.parquet' \
        --config_path=$BASE_PATH'/experiments/'$CONFIG_EXPERIMENT_NAME'/config/selected_models/best_config.yaml' \
        --experiment_name=$EXPERIMENT_NAME \
        --num_workers=5 \
        --data_mode="array" \
        --label_col=$TASK \
        --fold_id=$1 \
        --run_evaluation \
        --run_evaluation_group \
        --run_evaluation_group_standard \
        --eval_attributes "adult_at_admission" \
        --subset_train_attribute="subsample_"$2 \
        --subset_train_group="1"
}

train_model_func 1 $SUBSAMPLE_ID