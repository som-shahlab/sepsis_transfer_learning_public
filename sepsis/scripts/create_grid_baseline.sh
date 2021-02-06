#!/bin/bash

EXPERIMENT_NAME="baseline_config"
DATA_PATH="/share/pi/nigam/projects/sepsis/extraction_201003/"

python -m sepsis.create_grid_baseline \
    --data_path=$DATA_PATH \
    --experiment_name=$EXPERIMENT_NAME \
    --tasks="early_sepsis"