import argparse
import pandas as pd
import glob
import os
from prediction_utils.util import df_dict_concat, yaml_read, yaml_write

parser = argparse.ArgumentParser()

parser.add_argument(
    "--data_path",
    type=str,
    default="/share/pi/nigam/projects/sepsis/extraction_200815/",
    help="The root path where data is stored",
)

parser.add_argument(
    "--config_experiment_name",
    type=str,
    default="baseline_config",
    help="The experiment name where the config files are stored",
)

parser.add_argument(
    "--experiment_name",
    type=str,
    required=True,
    help="The experiment name containing the results",
)

if __name__ == "__main__":
    args = parser.parse_args()

    result_files = glob.glob(
        os.path.join(
            args.data_path,
            "experiments",
            args.experiment_name,
            "performance",
            "**",
            "result_df_training_eval.parquet",
        ),
        recursive=True,
    )
    print(args.experiment_name)
    result_df_dict = {
        tuple(file_name.split("/"))[-4:-1]: pd.read_parquet(file_name)
        for file_name in result_files
    }
    result_df = df_dict_concat(result_df_dict, ["task", "config_filename", "fold"])

    # Check whether all results are available
    assert (
        result_df.groupby(["task", "config_filename"])
        .agg(num_folds=("fold", lambda x: len(x.unique())))
        .query("num_folds != 10")
        .shape[0]
    ) == 0

    mean_performance = pd.DataFrame(
        result_df.query('metric == "loss" & phase == "eval"')
        .groupby(["config_filename", "task"])
        .agg(performance=("performance", "mean"))
        .reset_index()
    )
    best_model = (
        mean_performance.groupby("task")
        .agg(performance=("performance", "min"))
        .merge(mean_performance)
    )

    best_model_config_df = best_model[["config_filename", "task"]]
    best_model_performance = result_df.merge(best_model_config_df)

    base_config_path = os.path.join(
        args.data_path, "experiments", args.config_experiment_name, "config"
    )
    
    selected_config_path = os.path.join(
        args.data_path, "experiments", args.experiment_name, "selected_models", "config", 
    )

    # Write to a new directory
    for i, row in best_model_config_df.iterrows():
        the_config = yaml_read(
            os.path.join(base_config_path, row.task, row.config_filename)
        )
        print(the_config)
        the_config["label_col"] = row.task
        os.makedirs(os.path.join(selected_config_path, row.task), exist_ok=True)
        yaml_write(
            the_config, os.path.join(selected_config_path, row.task, "best_config.yaml")
        )
        best_model_config_df.to_csv(
            os.path.join(selected_config_path, row.task, "best_model_config_df.csv"),
            index=False,
        )
