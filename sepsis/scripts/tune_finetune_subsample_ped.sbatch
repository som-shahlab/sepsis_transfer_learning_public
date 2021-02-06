#!/bin/bash 
#SBATCH --job-name=tune_finetune_subsample_ped
#SBATCH --output=slurm_out/tune_finetune_subsample_ped_%a.out
#SBATCH --error=slurm_out/tune_finetune_subsample_ped_%a.err
#SBATCH --time=7-00:00:00
#SBATCH --nodes=1
#SBATCH --mem=20GB
#SBATCH -c 5
#SBATCH --array=0-19%20
#SBATCH -p gpu,nigam
#SBATCH --gres=gpu:1

source activate /share/pi/nigam/envs/anaconda/envs/prediction_utils

TASK="early_sepsis"
COHORT_NAME='cohort_pediatric_subsample.parquet'

BASE_EXPERIMENT_NAME="baseline_tuning_fold_1_10_adult"
DATA_PATH="/share/pi/nigam/projects/sepsis/extraction_201003"
FEATURES_PATH=$DATA_PATH"/merged_features_binary/features_sparse/features.gz"
VOCAB_PATH=$DATA_PATH"/merged_features_binary/vocab/vocab.parquet"
FEATURES_ROW_ID_MAP_PATH=$DATA_PATH"/merged_features_binary/features_sparse/features_row_id_map.parquet"
CONFIG_EXPERIMENT_NAME='finetune_config'

train_model_func() {
    EXPERIMENT_NAME='finetune_subsample_ped_'$2
    LOAD_CHECKPOINT_PATH=$DATA_PATH'/experiments/'$BASE_EXPERIMENT_NAME'/retrain_selected_models/checkpoints/early_sepsis/'$1'/checkpoint.chk'
    CONFIG_PATH=$DATA_PATH'/experiments/'$CONFIG_EXPERIMENT_NAME'/config/'$TASK'/'$SLURM_ARRAY_TASK_ID'.yaml'
    python -m sepsis.train_model \
        --data_path=$DATA_PATH \
        --features_path=$FEATURES_PATH \
        --cohort_path=$DATA_PATH'/cohort/'$COHORT_NAME \
        --vocab_path=$VOCAB_PATH \
        --features_row_id_map_path=$FEATURES_ROW_ID_MAP_PATH \
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
        --subset_train_group="1" \
        --apply_subset_train_only \
        --config_path=$CONFIG_PATH \
        --load_checkpoint_path=$LOAD_CHECKPOINT_PATH
}

for i in {0..9}
do
    for fold_id in {1..10}
    do
        train_model_func $fold_id $i
    done
done