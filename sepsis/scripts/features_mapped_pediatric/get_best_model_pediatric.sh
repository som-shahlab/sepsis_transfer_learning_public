
DATA_PATH="/share/pi/nigam/projects/sepsis/extraction_201003"
EXPERIMENT_NAME='features_mapped_pediatric/baseline_tuning_fold_1_10_pediatric'
CONFIG_EXPERIMENT_NAME='baseline_config'

python -m sepsis.get_best_model \
    --data_path=$DATA_PATH \
    --experiment_name=$EXPERIMENT_NAME \
    --config_experiment_name=$CONFIG_EXPERIMENT_NAME