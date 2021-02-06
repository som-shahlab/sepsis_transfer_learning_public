import numpy as np
import pandas as pd

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--cohort_path", type=str, required=True)
parser.add_argument("--cohort_path_pediatric_subsample", type=str, required=True)
parser.add_argument("--cohort_path_subsample", type=str, required=True)


def generate_subsample_ids(df, n_samples=10, geomspace=False, seed=398):
    df2 = (
        df.sample(frac=1, replace=False, random_state=seed)
        .reset_index(drop=True)
        .rename_axis("row_id")
        .reset_index()
        .assign(row_id=lambda x: x.row_id + 1)
        .assign(
            subsample_id=lambda x: pd.cut(
                x.row_id,
                np.geomspace(x.row_id.min(), x.row_id.max() + 1, n_samples + 1)
                if geomspace
                else n_samples,
                labels=False,
                include_lowest=True,
            )
        )[["person_id", "subsample_id"]]
    )
    df = df.merge(df2)
    for i in range(n_samples):
        df[f"subsample_{i}"] = 1 * (df.subsample_id <= i)
    return df


if __name__ == "__main__":
    args = parser.parse_args()
    cohort = pd.read_parquet(args.cohort_path)

    n_samples = 10
    cohort_ped = (
        cohort.query("adult_at_admission == 0")
        .groupby(["fold_id", "early_sepsis"])
        .apply(
            lambda x: generate_subsample_ids(x, n_samples=n_samples, geomspace=False)
        )
        .reset_index(drop=True)
    )
    cohort_adult = cohort.query("adult_at_admission == 1")
    cohort_subsample = pd.concat([cohort_adult, cohort_ped]).reset_index(drop=True)
    cohort_ped.to_parquet(args.cohort_path_pediatric_subsample)
    cohort_subsample.to_parquet(args.cohort_path_subsample)
