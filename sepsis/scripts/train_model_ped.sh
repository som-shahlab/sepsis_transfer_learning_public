FOLD_ID="1"
COHORT_NAME='cohort_pediatric_subsample.parquet'

EXPERIMENT_NAME="scratch"

DATA_PATH="/share/pi/nigam/projects/sepsis/extraction_201003"
FEATURES_PATH=$DATA_PATH"/merged_features_binary/features_sparse/features.gz"
COHORT_PATH=$DATA_PATH"/cohort/"$COHORT_NAME
VOCAB_PATH=$DATA_PATH"/merged_features_binary/vocab/vocab.parquet"
FEATURES_ROW_ID_MAP_PATH=$DATA_PATH"/merged_features_binary/features_sparse/features_row_id_map.parquet"
subset_id=1

python -m sepsis.train_model \
    --data_path=$DATA_PATH \
    --features_path=$FEATURES_PATH \
    --cohort_path=$COHORT_PATH \
    --vocab_path=$VOCAB_PATH \
    --features_row_id_map_path=$FEATURES_ROW_ID_MAP_PATH \
    --experiment_name=$EXPERIMENT_NAME \
    --eval_attributes='adult_at_admission' \
    --subset_train_attribute="subsample_"$subset_id \
    --subset_train_group="1" \
    --apply_subset_train_only \
    --fold_id=$FOLD_ID \
    --config_path=$CONFIG_PATH \
    --num_epochs=25 \
    --drop_prob=0.9 \
    --gamma=1.0