import numpy as np
import pandas as pd
import os
import glob
import argparse
import random
from prediction_utils.pytorch_utils.metrics import StandardEvaluator

parser = argparse.ArgumentParser()

parser.add_argument(
    "--data_path",
    type=str,
    default="/share/pi/nigam/projects/sepsis/extraction_200815/",
    help="The root path where data is stored",
)

parser.add_argument(
    "--cohort_path",
    type=str,
    default="/share/pi/nigam/projects/sepsis/extraction_200815/cohort/cohort.parquet",
    help="File name for the file containing label information",
)

parser.add_argument(
    "--features_row_id_map_path",
    type=str,
    default="/share/pi/nigam/projects/sepsis/extraction_200815/merged_features_binary/features_sparse/features_row_id_map.parquet",
)

parser.add_argument(
    "--experiment_name",
    type=str,
    required=True,
    help="The experiment name containing the results",
)

parser.add_argument("--n_boot", type=int, default=1000)


parser.add_argument("--seed", type=int, default=718)


def sample_cohort(df, eval_fold_ids=["eval", "test"]):
    return (
        df.query("fold_id in @eval_fold_ids")
        .groupby(["adult_at_admission", "fold_id"])
        .apply(lambda x: x.sample(frac=1.0, replace=True))
        .reset_index(drop=True)
    )


if __name__ == "__main__":
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    cohort = pd.read_parquet(args.cohort_path)
    features_row_id_map = pd.read_parquet(args.features_row_id_map_path)
    cohort = cohort.merge(features_row_id_map)
    cohort = cohort.rename(columns={"features_row_id": "row_id"})
    cohort_small = cohort[
        ["person_id", "row_id", "adult_at_admission", "fold_id", "early_sepsis"]
    ]

    best_model_config_df_path = os.path.join(
        args.data_path,
        "experiments",
        args.experiment_name,
        "selected_models",
        "config",
        "early_sepsis",
        "best_model_config_df.csv",
    )
    best_model_config_df = pd.read_csv(best_model_config_df_path)
    best_model_config_filename = best_model_config_df.config_filename.values[0]
    print(best_model_config_filename)

    output_files = glob.glob(
        os.path.join(
            args.data_path,
            "experiments",
            args.experiment_name,
            "**",
            best_model_config_filename,
            "**",
            "output_df.parquet",
        ),
        recursive=True,
    )
    output_df_dict = {
        file_name.split("/")[-2]: pd.read_parquet(file_name)
        for file_name in output_files
    }

    evaluator = StandardEvaluator()
    result_df_dict = {}
    for i in range(args.n_boot):
        cohort_boot = sample_cohort(cohort_small)
        for model_id, output_df in output_df_dict.items():
            output_df_resampled = output_df.merge(cohort_boot)
            result_df_dict[(i, model_id)] = evaluator.get_result_df(
                output_df_resampled, strata_vars=["adult_at_admission", "fold_id"]
            ).drop(columns=["performance_overall"])
    result_df = (
        pd.concat(result_df_dict)
        .reset_index(level=-1, drop=True)
        .rename_axis(["boot_id", "val_fold_id"])
        .reset_index()
    )
    result_df_mean = (
        result_df.groupby(["boot_id", "adult_at_admission", "metric", "fold_id"])
        .agg(performance=("performance", "mean"))
        .reset_index()
    )
    result_df_ci = (
        result_df_mean.groupby(["adult_at_admission", "metric", "fold_id"])
        .apply(lambda x: np.quantile(x.performance, [0.025, 0.5, 0.975]))
        .rename("performance")
        .reset_index()
        .assign(
            CI_lower=lambda x: x.performance.str[0],
            CI_med=lambda x: x.performance.str[1],
            CI_upper=lambda x: x.performance.str[2],
        )
        .drop(columns=["performance"])
    )
    result_df_ci_path = os.path.join(
        args.data_path,
        "experiments",
        args.experiment_name,
        "selected_models",
        "performance",
        "early_sepsis",
    )
    os.makedirs(result_df_ci_path, exist_ok=True)
    result_df_ci.to_csv(
        os.path.join(result_df_ci_path, "result_df_ci.csv"), index=False
    )
