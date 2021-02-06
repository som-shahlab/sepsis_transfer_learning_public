FOLD_ID="1"
EXPERIMENT_NAME="scratch"

DATA_PATH="/share/pi/nigam/projects/sepsis/extraction_201003"
FEATURES_PATH=$DATA_PATH"/merged_features_binary/features_sparse/features.gz"
COHORT_PATH=$DATA_PATH"/cohort/cohort_cv.parquet"
VOCAB_PATH=$DATA_PATH"/merged_features_binary/vocab/vocab.parquet"
FEATURES_ROW_ID_MAP_PATH=$DATA_PATH"/merged_features_binary/features_sparse/features_row_id_map.parquet"

COHORT_PATH=$DATA_PATH"/cohort/cohort_pediatric_subsample.parquet"
BASE_EXPERIMENT_NAME='baseline_tuning_fold_1_10_pediatric'
CONFIG_PATH='/share/pi/nigam/projects/sepsis/extraction_201003/experiments/'$BASE_EXPERIMENT_NAME'/selected_models/config/early_sepsis/best_config.yaml'

python -m sepsis.train_model \
    --data_path=$DATA_PATH \
    --features_path=$FEATURES_PATH \
    --cohort_path=$COHORT_PATH \
    --vocab_path=$VOCAB_PATH \
    --features_row_id_map_path=$FEATURES_ROW_ID_MAP_PATH \
    --experiment_name=$EXPERIMENT_NAME \
    --fold_id=$FOLD_ID \
    --config_path=$CONFIG_PATH \
    --eval_attributes="adult_at_admission" \
    --subset_train_attribute="subsample_9" \
    --subset_train_group="1"