#!/bin/bash

COHORT_PATH="/share/pi/nigam/projects/sepsis/extraction_201003/cohort"
python -m sepsis.split_cohort \
    --cohort_path_initial=$COHORT_PATH'/cohort_initial.parquet' \
    --cohort_path_cv=$COHORT_PATH'/cohort_cv.parquet' \
    --seed=926