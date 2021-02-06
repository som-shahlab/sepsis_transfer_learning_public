import pandas as pd
import os
import joblib
import configargparse as argparse
import copy
import torch
import numpy as np
from prediction_utils.pytorch_utils.models import FixedWidthModel
from prediction_utils.pytorch_utils.datasets import (
    ArrayLoaderGenerator_Alt,
    ParquetLoaderGenerator,
)
from prediction_utils.util import yaml_write
from prediction_utils.pytorch_utils.metrics import StandardEvaluator

parser = argparse.ArgumentParser(config_file_parser_class=argparse.YAMLConfigFileParser)

parser.add_argument("--config_path", required=False, is_config_file=True)

# Path configuration
parser.add_argument(
    "--data_path",
    type=str,
    default="/share/pi/nigam/projects/sepsis/extraction_200815/",
    help="The root path where data is stored",
)

parser.add_argument(
    "--features_path",
    type=str,
    default="/share/pi/nigam/projects/sepsis/extraction_200815/merged_features_binary/features_sparse/features.gz",
    help="The root path where data is stored",
)

parser.add_argument(
    "--cohort_path",
    type=str,
    default="/share/pi/nigam/projects/sepsis/extraction_200815/cohort/cohort.parquet",
    help="File name for the file containing label information",
)

parser.add_argument(
    "--vocab_path",
    type=str,
    default="/share/pi/nigam/projects/sepsis/extraction_200815/merged_features_binary/vocab/vocab.parquet",
    help="File name for the file containing label information",
)

parser.add_argument(
    "--features_row_id_map_path",
    type=str,
    default="/share/pi/nigam/projects/sepsis/extraction_200815/merged_features_binary/features_sparse/features_row_id_map.parquet",
)

parser.add_argument("--load_checkpoint_path", type=str, default=None)

parser.add_argument("--save_checkpoint_path", type=str, default=None)

# Model Hyperparameters
parser.add_argument(
    "--num_epochs", type=int, default=10, help="The number of epochs of training"
)
parser.add_argument(
    "--iters_per_epoch",
    type=int,
    default=100,
    help="The number of batches to run per epoch",
)

parser.add_argument("--batch_size", type=int, default=256, help="The batch size")

parser.add_argument("--lr", type=float, default=1e-4, help="The learning rate")

parser.add_argument("--gamma", type=float, default=0.95, help="Learning rate decay")

parser.add_argument(
    "--num_hidden", type=int, default=3, help="The number of hidden layers"
)

parser.add_argument(
    "--hidden_dim", type=int, default=128, help="The dimension of the hidden layers"
)

parser.add_argument(
    "--normalize", dest="normalize", action="store_true", help="Use layer normalization"
)

parser.add_argument(
    "--drop_prob", type=float, default=0.75, help="The dropout probability"
)

parser.add_argument(
    "--weight_decay", type=float, default=0.0, help="The value of the weight decay"
)

parser.add_argument(
    "--early_stopping",
    dest="early_stopping",
    action="store_true",
    help="Whether to use early stopping",
)

parser.add_argument("--early_stopping_patience", type=int, default=5)

parser.add_argument(
    "--selection_metric",
    type=str,
    default="loss",
    help="The metric to use for model selection",
)

parser.add_argument("--fold_id", type=str, default="1", help="The fold id")

parser.add_argument(
    "--experiment_name", type=str, default="scratch", help="The name of the experiment"
)

parser.add_argument(
    "--label_col", type=str, default="early_sepsis", help="The label to use"
)

parser.add_argument(
    "--data_mode", type=str, default="array", help="Which mode of source data to use"
)

parser.add_argument("--sparse_mode", type=str, default="csr", help="the sparse mode")
parser.add_argument(
    "--num_workers",
    type=int,
    default=5,
    help="The number of workers to use for data loading during training in parquet mode",
)

parser.add_argument(
    "--save_outputs",
    dest="save_outputs",
    action="store_true",
    help="Whether to save the outputs of evaluation",
)

parser.add_argument(
    "--run_evaluation",
    dest="run_evaluation",
    action="store_true",
    help="Whether to evaluate the model",
)

parser.add_argument(
    "--no_run_evaluation",
    dest="run_evaluation",
    action="store_false",
    help="Whether to evaluate the model",
)

parser.add_argument(
    "--run_evaluation_group",
    dest="run_evaluation",
    action="store_true",
    help="Whether to evaluate the model for each group",
)

parser.add_argument(
    "--no_run_evaluation_group",
    dest="run_evaluation_group",
    action="store_false",
    help="Whether to evaluate the model for each group",
)

parser.add_argument(
    "--run_evaluation_group_standard",
    dest="run_evaluation_group_standard",
    action="store_true",
    help="Whether to evaluate the model",
)
parser.add_argument(
    "--no_run_evaluation_group_standard",
    dest="run_evaluation_group_standard",
    action="store_false",
    help="Whether to evaluate the model",
)

parser.add_argument(
    "--eval_attributes", type=str, nargs="+", required=False, default=None
)

parser.add_argument("--sample_keys", type=str, nargs="*", required=False, default=None)

parser.add_argument("--subset_train_attribute", type=str, required=False, default=None)

parser.add_argument("--subset_train_group", type=str, required=False, default=None)

parser.add_argument(
    "--apply_subset_train_only",
    dest="apply_subset_train_only",
    action="store_true",
    help="Whether to apply the subsetting to only the training set, not the validation set",
)

parser.add_argument(
    "--deterministic",
    dest="deterministic",
    action="store_true",
    help="Whether to use deterministic training",
)

parser.add_argument(
    "--seed", type=int, default=2020, help="The seed",
)

parser.set_defaults(
    normalize=False,
    early_stopping=False,
    run_evaluation=True,
    save_outputs=True,
    run_evaluation_group=True,
    run_evaluation_group_standard=True,
    apply_subset_train_only=False,
    deterministic=True,
)


def get_loader_generator_class(data_mode="parquet"):
    if data_mode == "parquet":
        return ParquetLoaderGenerator
    elif data_mode == "array":
        return ArrayLoaderGenerator_Alt


def read_file(filename, columns=None, **kwargs):
    print(filename)
    load_extension = os.path.splitext(filename)[-1]
    if load_extension == ".parquet":
        return pd.read_parquet(filename, columns=columns, **kwargs)
    elif load_extension == ".csv":
        return pd.read_csv(filename, usecols=columns, **kwargs)


if __name__ == "__main__":
    args = parser.parse_args()

    if args.deterministic:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    config_dict = copy.deepcopy(args.__dict__)

    if args.fold_id == "":
        train_keys = ["train"]
        eval_keys = ["eval", "test"]
        fold_id_test_list = ['test', 'eval']
    else:
        train_keys = ["train", "val"]
        eval_keys = ["val", "eval", "test"]
        fold_id_test_list = ['test', 'eval']

    vocab = read_file(args.vocab_path, engine="pyarrow")
    config_dict["input_dim"] = vocab.col_id.max() + 1

    cohort = read_file(args.cohort_path)

    if args.data_mode == "array":
        features = joblib.load(args.features_path)
        if args.features_row_id_map_path != "":
            row_id_map = read_file(args.features_row_id_map_path, engine="pyarrow")
            cohort = cohort.merge(row_id_map)
            config_dict["row_id_col"] = "features_row_id"
    else:
        features = None

    cohort_eval = cohort.copy()

    if (args.subset_train_attribute is not None) and (
        args.subset_train_group is not None
    ):

        cohort = cohort.query(
            "{} == {}".format(args.subset_train_attribute, args.subset_train_group)
        )
    print("Cohort shape: {}".format(cohort.shape))

    if config_dict.get("config_path") is None:
        result_path_suffix = ""
    else:
        result_path_suffix = os.path.basename(config_dict["config_path"])

    result_path = os.path.join(
        args.data_path,
        "experiments",
        args.experiment_name,
        "performance",
        args.label_col,
        result_path_suffix,
        str(config_dict["fold_id"]),
    )
    print("Result path: {}".format(result_path))
    os.makedirs(result_path, exist_ok=True)

    loader_generator = get_loader_generator_class(data_mode=args.data_mode)(
        features=features,
        cohort=cohort,
        fold_id_test_list=fold_id_test_list,
        **config_dict
    )
    model = FixedWidthModel(**config_dict)
    print(model.config_dict)

    if args.load_checkpoint_path is not None:
        print("Loading checkpoint")
        model.model.load_state_dict(torch.load(args.load_checkpoint_path))

    # Write the resulting config
    yaml_write(config_dict, os.path.join(result_path, "config.yaml"))

    loaders = loader_generator.init_loaders(sample_keys=args.sample_keys)

    result_df = model.train(loaders, phases=train_keys)["performance"]
    del loaders

    if args.save_checkpoint_path is not None:
        os.makedirs(os.path.dirname(args.save_checkpoint_path), exist_ok=True)
        torch.save(model.model.state_dict(), args.save_checkpoint_path)

    # Dump training results to disk
    result_df.to_parquet(
        os.path.join(result_path, "result_df_training.parquet"),
        index=False,
        engine="pyarrow",
    )

    if args.run_evaluation:
        print("Evaluating model")
        loader_generator = get_loader_generator_class(data_mode=args.data_mode)(
            features=features,
            cohort=cohort_eval,
            fold_id_test_list=fold_id_test_list,
            **config_dict
        )
        loaders_predict = loader_generator.init_loaders_predict()
        predict_dict = model.predict(loaders_predict, phases=eval_keys)
        del loaders_predict
        output_df_eval, result_df_eval = (
            predict_dict["outputs"],
            predict_dict["performance"],
        )
        print(result_df_eval)

        # Dump evaluation result to disk
        result_df_eval.to_parquet(
            os.path.join(result_path, "result_df_training_eval.parquet"),
            index=False,
            engine="pyarrow",
        )
        if args.save_outputs:
            output_df_eval.to_parquet(
                os.path.join(result_path, "output_df.parquet"),
                index=False,
                engine="pyarrow",
            )
        if args.run_evaluation_group:
            if args.eval_attributes is None:
                raise ValueError(
                    "If using run_evaluation_group, must specify eval_attributes"
                )
            strata_vars = ["phase", "task", "sensitive_attribute", "attribute"]
            output_df_eval = output_df_eval.assign(task=args.label_col)
            output_df_eval = output_df_eval.merge(
                row_id_map, left_on="row_id", right_on="features_row_id"
            ).merge(cohort_eval)
            output_df_long = output_df_eval.melt(
                id_vars=set(output_df_eval.columns) - set(args.eval_attributes),
                value_vars=args.eval_attributes,
                var_name="attribute",
                value_name="group",
            )
            if args.run_evaluation_group_standard:
                evaluator = StandardEvaluator()
                result_df_group_standard_eval = evaluator.get_result_df(
                    output_df_long, strata_vars=strata_vars,
                )
                print(result_df_group_standard_eval)
                result_df_group_standard_eval.to_parquet(
                    os.path.join(result_path, "result_df_group_standard_eval.parquet"),
                    engine="pyarrow",
                    index=False,
                )
