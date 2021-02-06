import os
import argparse
from sepsis.cohort import SepsisCohort
from prediction_utils.util import patient_split

parser = argparse.ArgumentParser()
parser.add_argument(
    "--data_path",
    type=str,
    default="/share/pi/nigam/projects/sepsis/extraction_200815",
)

if __name__ == "__main__":
    args = parser.parse_args()

    config_dict = {
        "dataset_project": "som-rit-phi-starr-prod",
        "rs_dataset_project": "som-nero-phi-nigam-starr",
        "dataset": "starr_omop_cdm5_deid_latest",
        "rs_dataset": "temp_dataset",
    }
    cohort_names = {
        "admission_rollup": "admission_rollup",
        "suspected_infection": "susp_inf_rollup",
        "sepsis_initial": "sepsis_cohort_initial",
        "sepsis_prior": "sepsis_cohort_prior",
        "sepsis_combined": "sepsis_cohort_combined",
        "sepsis_combined_demographics": "sepsis_cohort_combined_demographics",
        "sepsis_with_pediatric": "sepsis_cohort_with_pediatric",
        "admission_sepsis_cohort": "admission_sepsis_cohort",
        "sepsis_cohort_final": "sepsis_cohort_final",
    }
    cohort_names_long = {
        key: "{rs_dataset_project}.{rs_dataset}.{cohort_name}".format(
            cohort_name=value, **config_dict
        )
        for key, value in cohort_names.items()
    }
    config_dict = {**config_dict, **cohort_names_long}

    cohort = SepsisCohort(**config_dict)
    admission_rollup_query = cohort.get_admission_rollup()
    suspected_infection_query = cohort.get_suspected_infection()
    sepsis_cohort_initial_query = cohort.get_sepsis_cohort_initial()
    sepsis_cohort_prior_query = cohort.get_sepsis_cohort_prior()
    sepsis_cohort_combined_query = cohort.get_sepsis_cohort_combined()
    sepsis_cohort_combined_demographics_query = cohort.get_cohort_with_demographics()
    sepsis_cohort_with_pediatric_query = cohort.get_cohort_with_pediatric()
    admission_sepsis_cohort_query = cohort.get_admission_sepsis_cohort()
    sampled_cohort_query = cohort.get_sampled_cohort()
    cohort.db.execute_sql_to_destination_table(
        query=admission_rollup_query, destination=config_dict["admission_rollup"]
    )
    cohort.db.execute_sql_to_destination_table(
        query=suspected_infection_query, destination=config_dict["suspected_infection"]
    )
    cohort.db.execute_sql_to_destination_table(
        query=sepsis_cohort_initial_query, destination=config_dict["sepsis_initial"]
    )
    cohort.db.execute_sql_to_destination_table(
        query=sepsis_cohort_prior_query, destination=config_dict["sepsis_prior"]
    )
    cohort.db.execute_sql_to_destination_table(
        query=sepsis_cohort_combined_query, destination=config_dict["sepsis_combined"]
    )

    cohort.db.execute_sql_to_destination_table(
        query=sepsis_cohort_combined_demographics_query,
        destination=config_dict["sepsis_combined_demographics"],
    )

    cohort.db.execute_sql_to_destination_table(
        query=sepsis_cohort_with_pediatric_query,
        destination=config_dict["sepsis_with_pediatric"],
    )

    cohort.db.execute_sql_to_destination_table(
        query=admission_sepsis_cohort_query,
        destination=config_dict["admission_sepsis_cohort"],
    )

    cohort.db.execute_sql_to_destination_table(
        query=sampled_cohort_query, destination=config_dict["sepsis_cohort_final"]
    )

    cohort_df = cohort.db.read_sql_query(
        """
                SELECT *
                FROM {sepsis_cohort_final}
            """.format_map(
            config_dict
        ),
        use_bqstorage_api=True,
    )
    cohort_path = os.path.join(args.data_path, "cohort")
    os.makedirs(cohort_path, exist_ok=True)
    cohort_df.to_parquet(
        os.path.join(cohort_path, "cohort.parquet"), engine="pyarrow", index=False,
    )
