#!/bin/bash
DATA_PATH="/share/pi/nigam/projects/sepsis/extraction_201003"
CONFIG_EXPERIMENT_NAME='baseline_config'

my_func() {
    EXPERIMENT_NAME='features_mapped_pediatric/baseline_subsample_ped_'$1
    python -m sepsis.get_best_model \
        --data_path=$DATA_PATH \
        --experiment_name=$EXPERIMENT_NAME \
        --config_experiment_name=$CONFIG_EXPERIMENT_NAME
}

for i in {0..9} 
do
    my_func $i
done