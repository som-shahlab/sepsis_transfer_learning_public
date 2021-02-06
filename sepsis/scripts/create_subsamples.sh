#!/bin/bash

ROOT_COHORT_PATH="/share/pi/nigam/projects/sepsis/extraction_201003/cohort"

python -m sepsis.create_subsamples \
    --cohort_path=$ROOT_COHORT_PATH'/cohort_cv.parquet' \
    --cohort_path_pediatric_subsample=$ROOT_COHORT_PATH'/cohort_pediatric_subsample.parquet' \
    --cohort_path_subsample=$ROOT_COHORT_PATH'/cohort_subsample.parquet'