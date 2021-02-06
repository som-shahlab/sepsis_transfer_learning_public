import random
import numpy as np
import argparse
import pandas as pd
from prediction_utils.util import patient_split_cv

parser = argparse.ArgumentParser()
parser.add_argument("--cohort_path_initial", type=str, required=True)
parser.add_argument("--cohort_path_cv", type=str, required=True)
parser.add_argument("--seed", type=int, default=926)

if __name__ == "__main__":
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    cohort = pd.read_parquet(args.cohort_path_initial)
    if "fold_id" in cohort.columns:
        cohort = cohort.drop(columns="fold_id")

    cohort_cv = (
        cohort.groupby(["early_sepsis", "adult_at_admission"])
        .apply(lambda x: patient_split_cv(x))
        .reset_index(drop=True)
    )
    cohort_cv.to_parquet(args.cohort_path_cv, index=False, engine="pyarrow")
