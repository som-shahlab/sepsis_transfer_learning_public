#!/bin/bash
#SBATCH --job-name=extract_features_sepsis
#SBATCH --output=slurm_out/extract_features_sepsis.out
#SBATCH --error=slurm_out/extract_features_sepsis.err
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --mem=100GB
#SBATCH -c 5
#SBATCH -p normal,gpu,nigam

source activate /share/pi/nigam/envs/anaconda/envs/prediction_utils
DATASET="starr_omop_cdm5_deid_latest"
RS_DATASET="temp_dataset"
COHORT_NAME="sepsis_cohort_final"
DATA_PATH="/share/pi/nigam/projects/sepsis/extraction_201003"

FEATURES_DATASET="temp_dataset"
GCLOUD_PROJECT="som-nero-phi-nigam-starr"
DATASET_PROJECT="som-rit-phi-starr-prod"
RS_DATASET_PROJECT="som-nero-phi-nigam-starr"
FEATURES_PREFIX="features_"$USER
INDEX_DATE_FIELD='admit_date'
ROW_ID_FIELD='prediction_id'
MERGED_NAME='merged_features_binary'

python -m prediction_utils.extraction_utils.extract_features \
    --data_path=$DATA_PATH \
    --features_by_analysis_path="features_by_analysis" \
    --dataset=$DATASET \
    --rs_dataset=$RS_DATASET \
    --cohort_name=$COHORT_NAME \
    --gcloud_project=$GCLOUD_PROJECT \
    --dataset_project=$DATASET_PROJECT \
    --rs_dataset_project=$RS_DATASET_PROJECT \
    --features_dataset=$FEATURES_DATASET \
    --features_prefix=$FEATURES_PREFIX \
    --index_date_field=$INDEX_DATE_FIELD \
    --row_id_field=$ROW_ID_FIELD \
    --merged_name=$MERGED_NAME \
    --exclude_analysis_ids "note_nlp" "note_nlp_dt" "note_nlp_delayed" \
    --time_bins "-365" "-30" \
    --binary \
    --featurize \
    --no_cloud_storage \
    --merge_features \
    --create_sparse \
    --no_create_parquet \
    --overwrite