#!/bin/bash
#SBATCH --job-name=tune_baseline_model_subsample_ped
#SBATCH --output=slurm_out/tune_baseline_model_subsample_ped_%a.out
#SBATCH --error=slurm_out/tune_baseline_model_subsample_ped_%a.err
#SBATCH --time=7-00:00:00
#SBATCH --nodes=1
#SBATCH --mem=20GB
#SBATCH -c 5
#SBATCH --array=0-49%10
#SBATCH -p gpu,nigam
#SBATCH --gres=gpu:1

source activate /share/pi/nigam/envs/anaconda/envs/prediction_utils

BASE_PATH="/share/pi/nigam/projects/sepsis/extraction_201003/"
TASK="early_sepsis"
COHORT_NAME='cohort_pediatric_subsample.parquet'
CONFIG_EXPERIMENT_NAME='baseline_config'

train_model_func() {
    EXPERIMENT_NAME='features_mapped_pediatric/baseline_subsample_ped_'$2
    python -m sepsis.train_model \
        --data_path=$BASE_PATH \
        --features_path=$BASE_PATH'/features_mapped_pediatric/features_sparse/features.gz' \
        --cohort_path=$BASE_PATH'/cohort/cohort_pediatric_subsample.parquet'\
        --vocab_path=$BASE_PATH'/features_mapped_pediatric/vocab/vocab.parquet' \
        --features_row_id_map_path=$BASE_PATH'/features_mapped_pediatric/features_sparse/features_row_id_map.parquet' \
        --config_path=$BASE_PATH'/experiments/'$CONFIG_EXPERIMENT_NAME'/config/'$TASK'/'$SLURM_ARRAY_TASK_ID'.yaml' \
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

for i in {0..9}
do
    for fold_id in {1..10}
    do
        train_model_func $fold_id $i
    done
done