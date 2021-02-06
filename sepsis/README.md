## Sepsis transfer learning


### Directory structure
    * Python files are located here `sepsis/*.py`
    * SLURM scripts are located in the `scripts` directory.

### The workflow
    
    * Enter the scripts directory and activate environment (must be active for `.sh` scripts)
        * `cd scripts`
        * `conda activate prediction_utils`

    * Create the cohort in BigQuery
        * `sbatch create_cohort.sbatch`

    * Extracts features from the database and writes results to a CSR matrix.
        * `sbatch extract_features.sbatch`
    
    * Split data
        * `bash split_cohort.sh`

    * Create subsamples
        * `bash create_subsamples.sh`

    * Generate hyperparameter tuning grid (To run experiments with existing cohort files, start here)
        * `bash create_grid_baseline.sh`
    
    * Evaluate all elements in the grid for the pooled, adult, and pediatric population (can be executed in parallel)
        * `sbatch tune_baseline_model.sbatch`
        * `sbatch tune_baseline_model_adult.sbatch`
        * `sbatch tune_baseline_model_pediatric.sbatch`
        * `sbatch tune_baseline_subsample_ped.sbatch`
    
    * Perform model selection (run after the corresponding tuning is complete)
        * `bash get_best_model_baseline.sh`
        * `bash get_best_model_adult.sh`
        * `bash get_best_model_pediatric.sh`
        * `bash get_best_model_subsample_ped.sh`

    * Generate bootstrap results (run after the corresponding get_best_model is complete)
        * `sbatch bootstrapping_baseline.sbatch`
        * `sbatch bootstrapping_adult.sbatch`
        * `sbatch bootstrapping_pediatric.sbatch`
        * `sbatch bootstrapping_subsample_ped.sbatch`
    
    * Retrain the adult model for further fine tuning (after running above pipeline for the adult model)
        * `sbatch retrain_baseline_model_adult.sbatch`

    * Generate hyperparameter tuning grid for fine tuning (after adult model is done re-training)
        * `bash create_grid_finetune_subsample.sh`
    
    * Fine tune model(s) using the checkpoints (after adult model is done re-training)
        * `sbatch tune_finetune_subsample_ped.sbatch`

    * Perform model selection for the finetuning model (after tuning the fine tuning is complete)
        * `bash get_best_model_finetune_subsample_ped.sh`

    * Generate bootstrap results (after previous step is complete)
        * `sbatch bootstrapping_finetune_subsample_ped.sbatch`

To experiment with overwriting results, you can play around with the file `train_model.sh`.
Setting EXPERIMENT_NAME='scratch' will ensure that results do not overwrite existing results. By default, the above workflow WILL overwrite existing experimental results.

### Running on the pediatric feature space

    * To map the data to pediatric feature space, execute the notebook `features_mapped_pediatric.ipynb`
        * This creates a parallel directory to `merged_features_binary` called `features_mapped_pediatric` with a new feature matrix and vocab file
    * The model training pipeline is replicated in a parallel set of scripts in a subdirectory of scripts called `features_mapped_pediatric`. Only a subset of the scripts in the original pipeline are provided, because we are going to reuse the grid of config files generated for the original experiments
    * The experiment names have been mirrored in a subdirectory of `experiments` called `features_mapped_pediatric`

#### Workflow
    
    * Enter the scripts/features_mapped_pediatric directory and activate environment (must be active for `.sh` scripts)
        * `cd scripts/features_mapped_pediatric`
        * `conda activate prediction_utils`

    * Evaluate all elements in the grid for the pooled, adult, and pediatric population (can be executed in parallel)
        * `sbatch tune_baseline_model.sbatch`
        * `sbatch tune_baseline_model_adult.sbatch`
        * `sbatch tune_baseline_model_pediatric.sbatch`
        * `sbatch tune_baseline_subsample_ped.sbatch`

    * Perform model selection (run after the corresponding tuning is complete)
        * `bash get_best_model_baseline.sh`
        * `bash get_best_model_adult.sh`
        * `bash get_best_model_pediatric.sh`
        * `bash get_best_model_subsample_ped.sh`

    * Generate bootstrap results (run after the corresponding get_best_model is complete)
        * `sbatch bootstrapping_baseline.sbatch`
        * `sbatch bootstrapping_adult.sbatch`
        * `sbatch bootstrapping_pediatric.sbatch`
        * `sbatch bootstrapping_subsample_ped.sbatch`

    * Retrain the adult model for further fine tuning (after running above pipeline for the adult model)
        * `sbatch retrain_baseline_model_adult.sbatch`

    * Generate hyperparameter tuning grid for fine tuning (after adult model is done re-training)
        * `bash create_grid_finetune_subsample.sh`

    * Fine tune model(s) using the checkpoints (after adult model is done re-training)
        * `sbatch tune_finetune_subsample_ped.sbatch`

    * Perform model selection for the finetuning model (after tuning the fine tuning is complete)
        * `bash get_best_model_finetune_subsample_ped.sh`

    * Generate bootstrap results (after previous step is complete)
        * `sbatch bootstrapping_finetune_subsample_ped.sbatch`