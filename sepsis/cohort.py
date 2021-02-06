from prediction_utils.cohorts.cohort import BQCohort


class SepsisCohort(BQCohort):
    def get_admission_rollup(self, format_query=True):
        query = """
            WITH 
            visits AS (
                SELECT t1.person_id, visit_occurrence_id, visit_concept_id, visit_start_date, visit_end_date
                FROM {dataset_project}.{dataset}.visit_occurrence t1
                INNER JOIN {dataset_project}.{dataset}.person as t2
                    ON t1.person_id = t2.person_id
                WHERE
                    visit_concept_id in (9201, 262)
                    AND visit_end_date > visit_start_date
                    AND visit_end_date is not NULL
                    AND visit_start_date is not NULL
            ),
            visits_melt AS (
                SELECT person_id, visit_start_date AS endpoint_date, 1 as endpoint_type
                FROM visits
                UNION ALL
                SELECT person_id, visit_end_date AS endpoint_date, -1 as endpoint_type
                FROM visits
            ),
            counts1 AS (
                SELECT *, COUNT(*) * endpoint_type as count
                FROM visits_melt
                GROUP BY person_id, endpoint_date, endpoint_type
            ),
            counts2 AS (
                SELECT person_id, endpoint_date, SUM(count) as count
                FROM counts1
                GROUP BY person_id, endpoint_date
            ),
            counts3 AS (
            SELECT 
                person_id, 
                endpoint_date, 
                SUM(count) OVER(PARTITION BY person_id ORDER BY endpoint_date) as count
            FROM counts2
            ),
            cum_counts AS (
                SELECT *, ROW_NUMBER() OVER(PARTITION BY person_id ORDER BY endpoint_date) as row_number
                FROM counts3
            ),
            discharge_times AS (
                SELECT person_id, endpoint_date, 'discharge_date' as endpoint_type, row_number
                FROM cum_counts
                WHERE count = 0
            ),
            discharge_times_row_shifted AS (
                SELECT person_id, (row_number + 1) as row_number
                FROM discharge_times
            ),
            first_admit_times AS (
                SELECT person_id, endpoint_date, 'admit_date' as endpoint_type
                FROM cum_counts
                WHERE row_number = 1
            ),
            other_admit_times AS (
                SELECT t1.person_id, endpoint_date, 'admit_date' as endpoint_type
                FROM cum_counts t1
                INNER JOIN discharge_times_row_shifted AS t2
                ON t1.person_id=t2.person_id AND t1.row_number=t2.row_number
            ),
            aggregated_endpoints AS (
                SELECT person_id, endpoint_date, endpoint_type
                FROM discharge_times
                UNION ALL
                SELECT person_id, endpoint_date, endpoint_type
                FROM first_admit_times
                UNION ALL
                SELECT person_id, endpoint_date, endpoint_type
                FROM other_admit_times
            ),
            result_long AS (
                SELECT *, ROW_NUMBER() OVER(PARTITION BY person_id, endpoint_type ORDER BY endpoint_date) as row_number
                FROM aggregated_endpoints
            ),
            discharge_times_final AS (
                SELECT person_id, endpoint_date as discharge_date, row_number
                FROM result_long
                WHERE endpoint_type = 'discharge_date'
            ),
            admit_times_final AS (
                SELECT person_id, endpoint_date as admit_date, row_number
                FROM result_long
                WHERE endpoint_type = 'admit_date'
            ),
            result AS (
                SELECT t1.person_id, admit_date, discharge_date, t1.row_number
                FROM admit_times_final t1
                INNER JOIN discharge_times_final as t2
                ON t1.person_id=t2.person_id AND t1.row_number=t2.row_number
            )
            SELECT person_id, admit_date, discharge_date
            FROM result
            ORDER BY person_id, row_number
        """
        if not format_query:
            return query
        else:
            return query.format_map(self.config_dict)

    def get_suspected_infection(self, format_query=True):
        query = """
            WITH 
            blood_culture_list AS (
                SELECT descendant_concept_id AS concept_id
                FROM {dataset_project}.{dataset}.concept_ancestor
                WHERE ancestor_concept_id = 4107893
            ),
            blood_culture_from_measurement_via_ancestor AS (  
                SELECT *
                FROM {dataset_project}.{dataset}.measurement AS measure
                WHERE measure.measurement_concept_id IN (
                    SELECT concept_id
                    FROM blood_culture_list)
            ),
            systemic_abx_list AS (
                SELECT descendant_concept_id AS concept_id
                FROM {dataset_project}.{dataset}.concept_ancestor
                WHERE ancestor_concept_id = 21602796 
            ),
            systemic_abx_from_drug_exposure_via_ancestor AS (
                SELECT *
                FROM {dataset_project}.{dataset}.drug_exposure AS drug
                WHERE drug.drug_concept_id IN (
                    SELECT concept_id
                    FROM systemic_abx_list)
            ),
            systemic_abx_from_drug_exposure_with_name AS (
                SELECT systemic_abx.*, concept.concept_name AS systemic_abx_type  
                FROM systemic_abx_from_drug_exposure_via_ancestor AS systemic_abx
                INNER JOIN {dataset_project}.{dataset}.concept AS concept
                ON systemic_abx.drug_concept_id = concept.concept_id
            ),
            bc_abx AS (
                SELECT 
                    blood_culture.person_id, 
                    blood_culture.measurement_DATETIME as bc_DATETIME,
                    systemic_abx.drug_exposure_start_DATETIME, 
                    systemic_abx.systemic_abx_type
                FROM blood_culture_from_measurement_via_ancestor AS blood_culture
                LEFT JOIN systemic_abx_from_drug_exposure_with_name AS systemic_abx
                ON blood_culture.person_id = systemic_abx.person_id
            ),
            admit_bc_abx AS (
                SELECT admission_rollup.*, bc_abx.bc_DATETIME, bc_abx.drug_exposure_start_DATETIME, bc_abx.systemic_abx_type   
                FROM {admission_rollup} as admission_rollup
                LEFT JOIN bc_abx AS bc_abx
                ON admission_rollup.person_id = bc_abx.person_id
            ),
            susp_inf_allrows AS (
                SELECT 
                    person_id, 
                    bc_DATETIME, 
                    drug_exposure_start_DATETIME, 
                    admit_date, 
                    discharge_date, 
                    systemic_abx_type,
                    datetime_diff(drug_exposure_start_DATETIME, bc_DATETIME, DAY) as days_bc_abx
                FROM admit_bc_abx as admit_bc_abx
                WHERE
                    CAST(bc_DATETIME AS DATE) >= DATE_SUB(admit_date, INTERVAL 1 DAY) AND  CAST(bc_DATETIME AS DATE) <= discharge_date AND
                    CAST(drug_exposure_start_DATETIME AS DATE) >= DATE_SUB(admit_date, INTERVAL 1 DAY) AND  CAST(drug_exposure_start_DATETIME AS DATE) <= discharge_date
                AND
                    CAST(bc_DATETIME AS DATE)<= CAST(DATETIME_ADD(drug_exposure_start_DATETIME, INTERVAL 1 DAY) AS DATE) AND
                    CAST(bc_DATETIME AS DATE)>= CAST(DATETIME_SUB(drug_exposure_start_DATETIME, INTERVAL 3 DAY) AS DATE) 
                ORDER BY person_id, admit_date, bc_DATETIME, drug_exposure_start_DATETIME
            )
            SELECT 
                person_id, admit_date, MIN(discharge_date) AS discharge_date,
                MIN(bc_DATETIME) as min_bc, MIN(drug_exposure_start_DATETIME) as min_systemic_abx,
                LEAST(MIN(bc_DATETIME),MIN(drug_exposure_start_DATETIME)) as index_date
            FROM susp_inf_allrows 
            GROUP BY person_id, admit_date
            ORDER BY person_id, admit_date 
            """

        if not format_query:
            return query
        else:
            return query.format_map(self.config_dict)

    def get_sepsis_cohort_initial(self, format_query=True):
        query = """
            WITH 
            platelet_from_measurement AS (
                SELECT measure.*, concept.concept_name AS measure_type  
                FROM {dataset_project}.{dataset}.measurement AS measure
                INNER JOIN {dataset_project}.{dataset}.concept AS concept
                ON measure.measurement_concept_id = concept.concept_id
                WHERE concept.concept_id=3007461
            ),
            bilirubin_from_measurement AS (
                SELECT measure.*, concept.concept_name AS measure_type  
                FROM {dataset_project}.{dataset}.measurement AS measure
                INNER JOIN {dataset_project}.{dataset}.concept AS concept
                ON measure.measurement_concept_id = concept.concept_id
                WHERE concept.concept_id =3024128
            ),
            ionotrope_list AS (
                SELECT descendant_concept_id AS concept_id
                FROM {dataset_project}.{dataset}.concept_ancestor
                WHERE ancestor_concept_id IN (21600284, 21600287, 21600303, 21600283)
            ),
            ionotrope_from_drug_exposure_via_ancestor AS ( 
                SELECT *
                FROM {dataset_project}.{dataset}.drug_exposure AS drug
                WHERE drug.drug_concept_id IN (
                    SELECT concept_id
                    FROM ionotrope_list
                )
            ),
            ionotrope_from_drug_exposure_with_name AS (
                SELECT ionotrope.*, concept.concept_name AS ionotrope_type  
                FROM ionotrope_from_drug_exposure_via_ancestor AS ionotrope
                INNER JOIN {dataset_project}.{dataset}.concept AS concept
                ON ionotrope.drug_concept_id = concept.concept_id
            ),
            creatinine_from_measurement AS (
                SELECT measure.*, concept.concept_name AS measure_type  
                FROM {dataset_project}.{dataset}.measurement AS measure
                INNER JOIN {dataset_project}.{dataset}.concept AS concept
                ON measure.measurement_concept_id = concept.concept_id
                WHERE concept.concept_id =3051825 OR concept.concept_id=3016723 
            ),
            lactate_from_measurement AS (
                SELECT measure.*, concept.concept_name AS measure_type  
                FROM {dataset_project}.{dataset}.measurement AS measure
                INNER JOIN {dataset_project}.{dataset}.concept AS concept
                ON measure.measurement_concept_id = concept.concept_id
                WHERE concept.concept_id =3020138 OR concept.concept_id=3047181
            ),
            paO2_from_measurement AS (
                SELECT measure.*,concept.concept_name AS measure_type  
                FROM {dataset_project}.{dataset}.measurement AS measure
                INNER JOIN {dataset_project}.{dataset}.concept AS concept
                ON measure.measurement_concept_id = concept.concept_id
                WHERE concept.concept_id=3027801 /*Oxygen [Partial pressure] in Arterial  blood*/ 
            ),
            glascow_coma_scale_from_flowsheet AS (
            SELECT flowsheet.*, CAST(flowsheet.observation_datetime AS DATETIME) AS gcs_date, 
                SAFE_CAST(flowsheet.meas_value AS FLOAT64) AS glascow_coma_scale
            FROM som-nero-phi-nigam-starr.jdposada_explore.flowsheets_mapped AS flowsheet 
            WHERE 
                UPPER(row_disp_name) LIKE "GLASGOW COMA SCALE%" AND 
                SAFE_CAST(flowsheet.meas_value AS FLOAT64) >= 3 AND flowsheet.meas_value IS NOT NULL
            ),
            mean_arterial_pressure_from_flowsheet AS (
                SELECT flowsheet.*  
                FROM som-nero-phi-nigam-starr.jdposada_explore.flowsheets_mapped  AS flowsheet
                WHERE  UPPER(row_disp_name) IN ("MEAN ARTERIAL PRESSURE", "ARTM", "MEAN ARTERIAL PRESSURE (CALCULATED)")
            ),
            mech_vent_from_flowsheet AS (
                SELECT person_id, observation_datetime, UPPER(row_disp_name) AS row_disp_name, meas_value, units
                FROM som-nero-phi-nigam-starr.jdposada_explore.flowsheets AS flowsheet_orig
                WHERE UPPER(row_disp_name) = "VENT MODE" AND meas_value <> 'STANDBY' AND meas_value <> 'MONITOR'
            ),
            # Assess as great as 48 hours before and up to 24 hours after index date
            platelet_window AS (
                SELECT susp_inf_rollup.*, 
                    platelet.measurement_DATETIME AS platelet_date, 
                    platelet.value_as_number,
                    datetime_diff(platelet.measurement_DATETIME, index_date, DAY) as days_plat_index
                --FROM som-nero-phi-nigam-starr.jdposada_explore.susp_inf_rollup AS susp_inf_rollup
                -- FROM susp_inf_rollup
                FROM {suspected_infection} AS susp_inf_rollup
                LEFT JOIN platelet_from_measurement AS platelet USING (person_id)
                WHERE
                    CAST(index_date AS DATE) <= CAST(DATETIME_ADD(measurement_DATETIME, INTERVAL 2 DAY) AS DATE) AND
                    CAST(index_date AS DATE) >= CAST(DATETIME_SUB(measurement_DATETIME, INTERVAL 1 DAY) AS DATE) AND
                    value_as_number IS NOT NULL
            ),
            platelet_rollup AS (
                SELECT person_id, admit_date, MIN(value_as_number) as min_platelet
                FROM platelet_window 
                GROUP BY person_id, admit_date
            ),
            bilirubin_window AS (
                SELECT susp_inf_rollup.*, 
                    bilirubin.measurement_DATETIME AS bilirubin_date, 
                    bilirubin.value_as_number,
                    datetime_diff(bilirubin.measurement_DATETIME, index_date, DAY) as days_bili_index
                --FROM som-nero-phi-nigam-starr.jdposada_explore.susp_inf_rollup AS susp_inf_rollup
                -- FROM susp_inf_rollup
                FROM {suspected_infection} AS susp_inf_rollup
                LEFT JOIN bilirubin_from_measurement AS bilirubin
                USING (person_id)
                WHERE
                    CAST(index_date AS DATE) <= CAST(DATETIME_ADD(measurement_DATETIME, INTERVAL 2 DAY) AS DATE) AND
                    CAST(index_date AS DATE) >= CAST(DATETIME_SUB(measurement_DATETIME, INTERVAL 1 DAY) AS DATE) AND
                    value_as_number IS NOT NULL
            ),
            bilirubin_rollup AS (
                SELECT person_id, admit_date, MAX(value_as_number) as max_bilirubin
                FROM bilirubin_window 
                GROUP BY person_id, admit_date
            ),
            vasopressor_window AS (
                SELECT 
                    susp_inf_rollup.person_id, 
                    susp_inf_rollup.admit_date, 
                    susp_inf_rollup.index_date,
                    vasopressor.drug_exposure_start_DATETIME, 
                    vasopressor.drug_exposure_end_DATETIME,
                    datetime_diff(vasopressor.drug_exposure_start_DATETIME, index_date, DAY) as days_index_vasostart,
                    datetime_diff(vasopressor.drug_exposure_end_DATETIME, index_date, DAY) as days_index_vasoend,
                    (datetime_diff(vasopressor.drug_exposure_end_DATETIME, vasopressor.drug_exposure_start_DATETIME, DAY) +1 ) as days_vasopressor
                --FROM som-nero-phi-nigam-starr.jdposada_explore.susp_inf_rollup AS susp_inf_rollup
                FROM {suspected_infection} AS susp_inf_rollup
                LEFT JOIN ionotrope_from_drug_exposure_with_name AS vasopressor
                USING (person_id)
                WHERE
                    CAST(index_date AS DATE) BETWEEN CAST(drug_exposure_start_DATETIME AS DATE) AND CAST(drug_exposure_end_DATETIME AS DATE) OR
                    CAST(DATETIME_ADD(index_date, INTERVAL 1 DAY) AS DATE) BETWEEN CAST(drug_exposure_start_DATETIME AS DATE) AND CAST (drug_exposure_end_DATETIME AS DATE) OR
                    CAST (DATETIME_SUB(index_date, INTERVAL 1 DAY) AS DATE) BETWEEN CAST (drug_exposure_start_DATETIME AS DATE) AND CAST (drug_exposure_end_DATETIME AS DATE) OR
                    CAST (DATETIME_SUB(index_date, INTERVAL 2 DAY) AS DATE) BETWEEN CAST (drug_exposure_start_DATETIME AS DATE) AND CAST (drug_exposure_end_DATETIME AS DATE)
            ),
            vasopressor_rollup AS (
                SELECT person_id, admit_date, 
                    MAX(datetime_diff(vasopressor.drug_exposure_end_DATETIME, vasopressor.drug_exposure_start_DATETIME, DAY) +1) as max_vaso_days
                FROM vasopressor_window as vasopressor 
                GROUP BY person_id, admit_date
            ),
            creatinine_window AS (
                SELECT susp_inf_rollup.*, creatinine.measurement_DATETIME AS creatinine_date, creatinine.value_as_number,
                    datetime_diff(creatinine.measurement_DATETIME, index_date, DAY) as days_crea_index
                --FROM som-nero-phi-nigam-starr.jdposada_explore.susp_inf_rollup AS susp_inf_rollup
                -- FROM susp_inf_rollup
                FROM {suspected_infection} AS susp_inf_rollup
                LEFT JOIN creatinine_from_measurement AS creatinine
                USING (person_id)
                WHERE
                    CAST(index_date AS DATE) <= CAST(DATETIME_ADD(measurement_DATETIME, INTERVAL 2 DAY) AS DATE) AND
                    CAST(index_date AS DATE) >= CAST(DATETIME_SUB(measurement_DATETIME, INTERVAL 1 DAY) AS DATE) AND
                    value_as_number IS NOT NULL
            ),
            creatinine_rollup AS (
                SELECT person_id, admit_date, MAX(value_as_number) as max_creatinine
                FROM creatinine_window 
                GROUP BY person_id, admit_date
            ),
            lactate_window AS (
                SELECT 
                    susp_inf_rollup.*, 
                    lactate.measurement_DATETIME AS lactate_date, 
                    lactate.value_as_number,
                    datetime_diff(lactate.measurement_DATETIME, index_date, DAY) as days_lact_index
                --FROM som-nero-phi-nigam-starr.jdposada_explore.susp_inf_rollup AS susp_inf_rollup
                -- FROM susp_inf_rollup
                FROM {suspected_infection} AS susp_inf_rollup
                LEFT JOIN lactate_from_measurement AS lactate
                USING (person_id)
                WHERE
                    CAST(index_date AS DATE) <= CAST(DATETIME_ADD(measurement_DATETIME, INTERVAL 2 DAY) AS DATE) AND
                    CAST(index_date AS DATE) >= CAST(DATETIME_SUB(measurement_DATETIME, INTERVAL 1 DAY) AS DATE) AND
                    value_as_number IS NOT NULL
            ),
            lactate_rollup AS (
                SELECT person_id, admit_date, MAX(value_as_number) as max_lactate
                FROM lactate_window 
                GROUP BY person_id, admit_date
            ),
            paO2_window AS (
                SELECT susp_inf_rollup.*, paO2.measurement_DATETIME AS paO2_datetime, paO2.value_as_number as paO2,
                    datetime_diff(paO2.measurement_DATETIME, index_date, DAY) as days_paO2_index
                -- FROM som-nero-phi-nigam-starr.jdposada_explore.susp_inf_rollup AS susp_inf_rollup
                FROM {suspected_infection} AS susp_inf_rollup
                LEFT JOIN paO2_from_measurement AS paO2
                USING (person_id)
                WHERE
                    CAST(index_date AS DATE)<= CAST(DATETIME_ADD(measurement_DATETIME, INTERVAL 2 DAY) AS DATE) AND
                    CAST(index_date AS DATe)>= CAST(DATETIME_SUB(measurement_DATETIME, INTERVAL 1 DAY) AS DATE) AND
                    value_as_number IS NOT NULL
            ),
            fiO2_window AS ( 
                SELECT 
                    susp_inf_rollup.person_id, 
                    min_bc, 
                    min_systemic_abx, 
                    susp_inf_rollup.admit_date, 
                    index_date, 
                    CAST(observation_datetime AS DATETIME) AS fiO2_datetime, 
                    SAFE_CAST(meas_value AS float64) AS fiO2, 
                    datetime_diff(CAST(observation_datetime AS DATETIME), index_date, DAY) as days_fiO2_index
                --FROM som-nero-phi-nigam-starr.jdposada_explore.susp_inf_rollup AS susp_inf_rollup
                -- FROM susp_inf_rollup
                FROM {suspected_infection} AS susp_inf_rollup
                LEFT JOIN som-nero-phi-nigam-starr.jdposada_explore.flowsheets_mapped AS flowsheet
                ON susp_inf_rollup.person_id = CAST(flowsheet.person_id AS INT64) 
                WHERE
                CAST(index_date AS DATE)<= CAST(DATETIME_ADD(CAST(observation_datetime AS DATETIME), INTERVAL 2 DAY) AS DATE) AND
                CAST(index_date As DATE)>= CAST(DATETIME_SUB(CAST(observation_datetime AS DATETIME), INTERVAL 1 DAY) AS DATE) AND
                UPPER(row_disp_name) ="FIO2" AND 
                meas_value IS NOT NULL AND SAFE_CAST(meas_value AS FLOAT64) >=21 AND SAFE_CAST(meas_value AS FLOAT64) <=100
            ),
            paO2_fiO2_window AS (
                SELECT 
                    paO2_window.person_id, 
                    paO2_window.admit_date, 
                    paO2_window.index_date, 
                    fiO2, 
                    fiO2_datetime, 
                    paO2, 
                    paO2_datetime,  
                    paO2/(NULLIF(fiO2, 0))*100 AS paO2fiO2_ratio, 
                    datetime_diff(paO2_datetime, fiO2_datetime, MINUTE) as minutes_fiO2_paO2
                FROM fiO2_window AS fiO2_window
                INNER JOIN paO2_window AS paO2_window
                USING (person_id, index_date)
                WHERE CAST(fiO2_datetime AS DATETIME)<= paO2_datetime 
            ),
            paO2_fiO2_initial_rollup AS (
                SELECT person_id, admit_date, paO2_datetime, MIN(minutes_fiO2_paO2) As minutes_fiO2_paO2 
                FROM paO2_fiO2_window 
                GROUP BY person_id, admit_date, paO2_datetime
            ),
            paO2_fiO2_initial_rollup_join AS (
                SELECT 
                    initial_rollup.person_id, 
                    initial_rollup.admit_date, 
                    initial_rollup.paO2_datetime, 
                    initial_rollup.minutes_fiO2_paO2,
                    index_date, 
                    fiO2, fiO2_datetime, paO2, paO2fiO2_ratio
                FROM paO2_fiO2_initial_rollup AS initial_rollup
                LEFT JOIN paO2_fiO2_window AS combined_window
                USING (person_id, paO2_datetime, minutes_fiO2_paO2)
            ),
            paO2_fiO2_rollup AS (
                SELECT person_id, CAST(admit_date AS DATE) as admit_date, MIN(paO2fiO2_ratio) as min_paO2fiO2_ratio
                FROM paO2_fiO2_initial_rollup_join 
                WHERE minutes_fiO2_paO2 <= 24*60
                GROUP BY person_id, admit_date
            ),
            glasgow_coma_scale_window AS (
                SELECT 
                    susp_inf_rollup.*, 
                    CAST(flowsheet.observation_datetime AS DATETIME) AS gcs_datetime, 
                    CAST(flowsheet.meas_value AS FLOAT64) AS glascow_coma_scale,
                    datetime_diff(CAST(observation_datetime AS DATETIME), index_date, DAY) as days_gcs_index
                --FROM som-nero-phi-nigam-starr.jdposada_explore.susp_inf_rollup AS susp_inf_rollup
                -- FROM susp_inf_rollup
                FROM {suspected_infection} AS susp_inf_rollup
                LEFT JOIN glascow_coma_scale_from_flowsheet as flowsheet
                ON susp_inf_rollup.person_id =  CAST(flowsheet.person_id AS INT64)
                WHERE
                CAST(index_date AS DATE)<= CAST(DATETIME_ADD(CAST(observation_datetime AS DATETIME), INTERVAL 2 DAY) AS DATE) AND
                CAST(index_date AS DATE)>= CAST(DATETIME_SUB(CAST(observation_datetime AS DATETIME), INTERVAL 1 DAY) AS DATE) AND
                CAST(flowsheet.meas_value AS FLOAT64) >= 3 AND flowsheet.meas_value IS NOT NULL
            ),
            glasgow_coma_scale_rollup AS (
                SELECT person_id, admit_date, MIN(glascow_coma_scale) as min_gcs
                FROM glasgow_coma_scale_window 
                GROUP BY person_id, admit_date
            ),
            mean_arterial_pressure_window AS (
                SELECT 
                    susp_inf_rollup.*, 
                    CAST(flowsheet.observation_datetime AS DATETIME) AS map_datetime,
                    CAST(flowsheet.meas_value AS FLOAT64) AS map,
                    datetime_diff(CAST(observation_datetime AS DATETIME), index_date, DAY) as days_map_index
                --FROM som-nero-phi-nigam-starr.jdposada_explore.susp_inf_rollup AS susp_inf_rollup
                -- FROM susp_inf_rollup
                FROM {suspected_infection} AS susp_inf_rollup
                LEFT JOIN mean_arterial_pressure_from_flowsheet AS flowsheet
                    ON susp_inf_rollup.person_id = CAST(flowsheet.person_id AS INT64)
                WHERE
                    CAST(index_date AS DATE)<= CAST(DATETIME_ADD(CAST(observation_datetime AS DATETIME), INTERVAL 2 DAY) AS DATE) AND
                    CAST(index_date AS DATE)>= CAST(DATETIME_SUB(CAST(observation_datetime AS DATETIME), INTERVAL 1 DAY) AS DATE) AND
                    flowsheet.meas_value IS NOT NULL
            ),
            mean_arterial_pressure_rollup AS (
                SELECT person_id, admit_date, MIN(map) as min_map
                FROM mean_arterial_pressure_window 
                WHERE map >=10
                GROUP BY person_id, admit_date
            ),
            mech_vent_window AS (
            SELECT 
                susp_inf_rollup.*, 
                mech_vent.observation_datetime AS mech_vent_datetime, 
                mech_vent.meas_value as vent_mode,
                datetime_diff(mech_vent.observation_datetime, index_date, DAY) as days_mech_vent_index
            --FROM som-nero-phi-nigam-starr.jdposada_explore.susp_inf_rollup AS susp_inf_rollup
            -- FROM susp_inf_rollup
            FROM {suspected_infection} AS susp_inf_rollup
            LEFT JOIN mech_vent_from_flowsheet AS mech_vent USING (person_id)
            WHERE
                CAST(index_date AS DATE)<= CAST(DATETIME_ADD(observation_datetime, INTERVAL 2 DAY) AS DATE) AND
                CAST(index_date AS DATE)>= CAST(DATETIME_SUB(observation_datetime, INTERVAL 1 DAY) AS DATE)
            ORDER BY  person_id, admit_date, index_date, observation_datetime 
            ),
            mech_vent_rollup AS (
                SELECT person_id, admit_date, COUNT(vent_mode) as count_vent_mode
                FROM mech_vent_window 
                GROUP BY person_id, admit_date
            ),
            urine_output_window AS (
                SELECT 
                    susp_inf_rollup.*, 
                    CAST(flowsheet.observation_datetime AS DATETIME) AS urine_datetime, 
                    SAFE_CAST(flowsheet.meas_value AS FLOAT64) AS urine_volume,
                    datetime_diff(CAST(observation_datetime AS DATETIME), index_date, DAY) as days_urine_index
                --FROM som-nero-phi-nigam-starr.jdposada_explore.susp_inf_rollup AS susp_inf_rollup
                -- FROM susp_inf_rollup
                FROM {suspected_infection} AS susp_inf_rollup
                LEFT JOIN som-nero-phi-nigam-starr.jdposada_explore.flowsheets AS flowsheet
                    ON susp_inf_rollup.person_id = CAST(flowsheet.person_id AS INT64) 
                WHERE
                    CAST(index_date AS DATE)<= CAST(DATETIME_ADD(CAST(observation_datetime AS DATETIME), INTERVAL 2 DAY) AS DATE) AND
                    CAST(index_date As DATE)>= CAST(DATETIME_SUB(CAST(observation_datetime AS DATETIME), INTERVAL 1 DAY) AS DATE) AND
                    UPPER(row_disp_name) IN ("URINE OUTPUT (ML)", "URINE") AND meas_value IS NOT NULL AND SAFE_CAST(meas_value AS FLOAT64) >=0
                ORDER BY person_id, admit_date, observation_datetime 
            ),
            admit_time AS (
                SELECT 
                    person_id, 
                    MIN(observation_datetime) AS ext_urine_datetime,
                    EXTRACT(HOUR FROM MIN(observation_datetime)) AS hour, 
                    (24-EXTRACT(HOUR FROM MIN(observation_datetime))) AS adjust_hours
                FROM urine_output_window AS urine
                LEFT JOIN som-nero-phi-nigam-starr.jdposada_explore.flowsheets AS flowsheets_orig  USING (person_id)
                WHERE CAST(admit_date AS DATE) = CAST(observation_datetime AS DATE)  
                AND observation_datetime <> DATETIME_TRUNC(observation_datetime, DAY)
                GROUP BY person_id
            ),
            discharge_time AS (
                SELECT person_id, 
                    MAX(observation_datetime) AS ext_urine_datetime,
                    EXTRACT(HOUR FROM MAX(observation_datetime)) AS adjust_hours
                FROM urine_output_window AS urine
                LEFT JOIN som-nero-phi-nigam-starr.jdposada_explore.flowsheets AS flowsheets_orig  USING (person_id)
                WHERE CAST(admit_date AS DATE) = CAST(observation_datetime AS DATE)  
                AND observation_datetime <> DATETIME_TRUNC(observation_datetime, DAY)
                GROUP BY person_id
            ),
            urine_output_initial_rollup AS (
                (
                    SELECT 
                        person_id, 
                        admit_date, 
                        discharge_date, 
                        CAST(urine_datetime AS DATE) AS urine_date, 
                        SUM(urine_volume) as urine_daily_output_orig, 
                        SUM(urine_volume) as urine_daily_output_adj,
                        ext_urine_datetime, 
                        adjust_hours
                    FROM urine_output_window 
                    LEFT JOIN admit_time USING (person_id)
                    WHERE CAST(urine_datetime AS DATE) <> CAST(admit_date AS DATE) AND CAST(urine_datetime AS DATE) <> CAST (discharge_date AS DATE)
                    GROUP BY person_id, admit_date, discharge_date, CAST (urine_datetime AS DATE), ext_urine_datetime, adjust_hours 
                )
                UNION ALL
                (
                    SELECT 
                        person_id, 
                        admit_date, 
                        discharge_date, 
                        CAST(urine_datetime AS DATE) AS urine_date, 
                        SUM(urine_volume) as urine_daily_output_orig, 
                        (SUM(urine_volume))*24/adjust_hours as urine_daily_output_adj,
                        ext_urine_datetime, adjust_hours
                    FROM urine_output_window 
                    LEFT JOIN admit_time USING (person_id)
                    WHERE CAST(urine_datetime AS DATE) = CAST(admit_date AS DATE) 
                    GROUP BY person_id, admit_date, discharge_date, CAST(urine_datetime AS DATE), ext_urine_datetime, adjust_hours
                )
                UNION ALL
                (
                    SELECT 
                        person_id, 
                        admit_date, 
                        discharge_date, 
                        CAST(urine_datetime AS DATE) AS urine_date, 
                        SUM(urine_volume) as urine_daily_output_orig, (SUM(urine_volume))*24/adjust_hours as urine_daily_output_adj,
                        ext_urine_datetime, 
                        adjust_hours
                    FROM urine_output_window 
                    LEFT JOIN discharge_time USING (person_id)
                    WHERE CAST(urine_datetime AS DATE) = CAST(discharge_date AS DATE) AND adjust_hours <> 0 
                    GROUP BY person_id, admit_date, discharge_date, CAST(urine_datetime AS DATE), ext_urine_datetime, adjust_hours
                )
                UNION ALL # THIS LAST BIT DEALS WITH DISCHARGE AT 0:00:00 HOURS
                (
                    SELECT 
                        person_id, 
                        admit_date, 
                        discharge_date, 
                        CAST(urine_datetime AS DATE) AS urine_date, 
                        SUM(urine_volume) as urine_daily_output_orig, (SUM(urine_volume))*24 as urine_daily_output_adj,
                        ext_urine_datetime, adjust_hours
                    FROM urine_output_window 
                    LEFT JOIN discharge_time USING (person_id)
                    WHERE CAST(urine_datetime AS DATE) = CAST(discharge_date AS DATE) AND adjust_hours = 0 
                    GROUP BY person_id, admit_date, discharge_date, CAST(urine_datetime AS DATE), ext_urine_datetime, adjust_hours
                )
            ),
            urine_output_rollup AS (
                SELECT person_id, admit_date, Min(urine_daily_output_adj) as min_urine_daily
                FROM urine_output_initial_rollup 
                GROUP BY person_id, admit_date
            )
            # Begin sepsis and shock cohort
            SELECT 
                susp_inf_rollup.person_id, 
                CAST(susp_inf_rollup.admit_date AS DATE) AS admit_date, 
                CAST(discharge_date AS DATE) AS discharge_date,
                min_bc, 
                min_systemic_abx, 
                index_date, 
                min_platelet, 
                CASE 
                    WHEN min_platelet IS NULL THEN 0 
                    WHEN min_platelet <20 THEN 4 
                    WHEN min_platelet < 50 THEN 3
                    WHEN min_platelet < 100 THEN 2 
                    WHEN min_platelet < 150 THEN 1 
                    ELSE 0 
                END plat_SOFA,
                max_bilirubin, 
                CASE 
                    WHEN max_bilirubin IS NULL THEN 0 
                    WHEN max_bilirubin >= 12 THEN 4 
                    WHEN max_bilirubin >= 6 THEN 3
                    WHEN max_bilirubin >= 2 THEN 2 
                    WHEN max_bilirubin >= 1.2 THEN 1 
                    ELSE 0
                END bili_SOFA,
                max_creatinine, 
                CASE 
                    WHEN max_creatinine IS NULL THEN 0 
                    WHEN max_creatinine >= 5 THEN 4 
                    WHEN max_creatinine >= 3.5 THEN 3
                    WHEN max_creatinine >= 2 THEN 2 
                    WHEN max_creatinine >= 1.2 THEN 1 
                    ELSE 0 
                END crea_SOFA,
                max_vaso_days, 
                min_map, 
                CASE 
                    WHEN max_vaso_days IS NOT NULL THEN 2
                    WHEN min_map <70 THEN 1 
                    ELSE 0 
                END cv_SOFA,
                min_paO2fiO2_ratio, 
                count_vent_mode, 
                CASE 
                    WHEN (count_vent_mode IS NOT NULL AND min_paO2fiO2_ratio < 100) THEN 4
                    WHEN count_vent_mode IS NOT NULL THEN 3 
                    WHEN min_paO2fiO2_ratio < 300 THEN 2 
                    WHEN min_paO2fiO2_ratio < 400 THEN 1
                    ELSE 0 
                END resp_SOFA,
                min_gcs, 
                CASE 
                    WHEN min_gcs IS NULL THEN 0 
                    WHEN min_gcs < 6 THEN 4 
                    WHEN min_gcs < 10 THEN 3
                    WHEN min_gcs < 13 THEN 2 
                    WHEN min_gcs < 15 THEN 1 
                    ELSE 0 
                END gcs_SOFA,
                min_urine_daily, 
                CASE 
                    WHEN min_urine_daily IS NULL THEN 0 
                    WHEN min_urine_daily < 200 THEN 4 
                    WHEN min_urine_daily < 500 THEN 3 
                    ELSE 0 
                END urine_SOFA,
                CASE 
                    WHEN max_vaso_days IS NOT NULL THEN 1 
                    ELSE 0 
                END vaso_shock,
                CASE 
                    WHEN min_map < 65 THEN 1 
                    ELSE 0 
                END map_shock,
                max_lactate, 
                CASE 
                    WHEN max_lactate > 2 THEN 1 
                    ELSE 0 
                END lact_shock
            --FROM som-nero-phi-nigam-starr.jdposada_explore.susp_inf_rollup AS susp_inf
            -- FROM susp_inf_rollup as susp_inf
            FROM {suspected_infection} AS susp_inf_rollup
            LEFT JOIN platelet_rollup USING (person_id, admit_date)
            LEFT JOIN bilirubin_rollup USING (person_id, admit_date)
            LEFT JOIN creatinine_rollup USING (person_id, admit_date)
            LEFT JOIN vasopressor_rollup USING (person_id, admit_date)
            LEFT JOIN mean_arterial_pressure_rollup USING (person_id, admit_date)
            LEFT JOIN paO2_fiO2_rollup USING (person_id, admit_date)
            LEFT JOIN mech_vent_rollup USING (person_id, admit_date)
            LEFT JOIN glasgow_coma_scale_rollup USING (person_id, admit_date)
            LEFT JOIN urine_output_rollup USING (person_id, admit_date)
            LEFT JOIN lactate_rollup USING (person_id, admit_date)
            ORDER BY person_id, CAST(admit_date AS DATE)
        """

        if not format_query:
            return query
        else:
            return query.format_map(self.config_dict)

    def get_sepsis_cohort_prior(self, format_query=True):
        query = """
        WITH 
        platelet_from_measurement AS (
            SELECT measure.*, concept.concept_name AS measure_type  
            FROM {dataset_project}.{dataset}.measurement AS measure
            INNER JOIN {dataset_project}.{dataset}.concept AS concept
            ON measure.measurement_concept_id = concept.concept_id
            WHERE concept.concept_id=3007461
        ),
        bilirubin_from_measurement AS (
            SELECT measure.*, concept.concept_name AS measure_type  
            FROM {dataset_project}.{dataset}.measurement AS measure
            INNER JOIN {dataset_project}.{dataset}.concept AS concept
            ON measure.measurement_concept_id = concept.concept_id
            WHERE concept.concept_id = 3024128
        ),
        ionotrope_list AS (
            SELECT descendant_concept_id AS concept_id
            FROM {dataset_project}.{dataset}.concept_ancestor
            WHERE ancestor_concept_id IN (21600284, 21600287, 21600303, 21600283)
        ),
        ionotrope_from_drug_exposure_via_ancestor AS ( 
            SELECT *
            FROM {dataset_project}.{dataset}.drug_exposure AS drug
            WHERE drug.drug_concept_id IN (
                SELECT concept_id
                FROM ionotrope_list
            )
        ),
        ionotrope_from_drug_exposure_with_name AS (
            SELECT ionotrope.*, concept.concept_name AS ionotrope_type  
            FROM ionotrope_from_drug_exposure_via_ancestor AS ionotrope
            INNER JOIN {dataset_project}.{dataset}.concept AS concept
            ON ionotrope.drug_concept_id = concept.concept_id
        ),
        creatinine_from_measurement AS (
            SELECT measure.*, concept.concept_name AS measure_type  
            FROM {dataset_project}.{dataset}.measurement AS measure
            INNER JOIN {dataset_project}.{dataset}.concept AS concept
            ON measure.measurement_concept_id = concept.concept_id
            WHERE concept.concept_id =3051825 OR concept.concept_id=3016723 
        ),
        lactate_from_measurement AS (
            SELECT measure.*, concept.concept_name AS measure_type  
            FROM {dataset_project}.{dataset}.measurement AS measure
            INNER JOIN {dataset_project}.{dataset}.concept AS concept
            ON measure.measurement_concept_id = concept.concept_id
            WHERE concept.concept_id =3020138 OR concept.concept_id =3047181
        ),
        paO2_from_measurement AS (
            SELECT measure.*,concept.concept_name AS measure_type  
            FROM {dataset_project}.{dataset}.measurement AS measure
            INNER JOIN {dataset_project}.{dataset}.concept AS concept
            ON measure.measurement_concept_id = concept.concept_id
            WHERE concept.concept_id=3027801 /*Oxygen [Partial pressure] in Arterial  blood*/ 
        ),
        glascow_coma_scale_from_flowsheet AS (
            SELECT 
                flowsheet.*, 
                CAST(flowsheet.observation_datetime AS DATETIME) AS gcs_date, 
                SAFE_CAST(flowsheet.meas_value AS FLOAT64) AS glascow_coma_scale
            FROM som-nero-phi-nigam-starr.jdposada_explore.flowsheets_mapped AS flowsheet 
            WHERE UPPER(row_disp_name) LIKE "GLASGOW COMA SCALE%" AND 
            SAFE_CAST(flowsheet.meas_value AS FLOAT64) >= 3 AND flowsheet.meas_value IS NOT NULL 
        ),
        mean_arterial_pressure_from_flowsheet AS (
            SELECT flowsheet.*  
            FROM som-nero-phi-nigam-starr.jdposada_explore.flowsheets_mapped AS flowsheet
            WHERE  UPPER(row_disp_name) IN ("MEAN ARTERIAL PRESSURE", "ARTM", "MEAN ARTERIAL PRESSURE (CALCULATED)") 
        ),
        mech_vent_from_flowsheet AS (
            SELECT person_id, observation_datetime, UPPER(row_disp_name) AS row_disp_name, meas_value, units
            FROM som-nero-phi-nigam-starr.jdposada_explore.flowsheets AS flowsheets_orig
            WHERE UPPER(row_disp_name) = "VENT MODE" AND meas_value <> 'STANDBY' AND meas_value <> 'MONITOR'
        ),
        # Assess 10 days to 3 days prior to index date
        platelet_window AS (
            SELECT 
                susp_inf_rollup.*, 
                platelet.measurement_DATETIME AS platelet_date, 
                platelet.value_as_number,
                datetime_diff(platelet.measurement_DATETIME, index_date, DAY) as days_plat_index
            -- FROM som-nero-phi-nigam-starr.jdposada_explore.susp_inf_rollup AS susp_inf_rollup
            FROM {suspected_infection} AS susp_inf_rollup
            LEFT JOIN platelet_from_measurement AS platelet
            USING (person_id)
            WHERE
                DATE_SUB(CAST(index_date AS DATE), INTERVAL 2 DAY) > CAST(measurement_DATETIME AS DATE) AND
                DATE_SUB(CAST(index_date AS DATE), INTERVAL 10 DAY) <= CAST(measurement_DATETIME AS DATE) AND
                value_as_number IS NOT NULL
        ),
        platelet_rollup AS (
            SELECT person_id, admit_date, MIN(value_as_number) as min_platelet_prior
            FROM platelet_window 
            GROUP BY person_id, admit_date
        ),
        bilirubin_window AS (
            SELECT 
                susp_inf_rollup.*, 
                bilirubin.measurement_DATETIME AS bilirubin_date, 
                bilirubin.value_as_number,
                datetime_diff(bilirubin.measurement_DATETIME, index_date, DAY) as days_bili_index
            -- FROM som-nero-phi-nigam-starr.jdposada_explore.susp_inf_rollup AS susp_inf_rollup
            FROM {suspected_infection} AS susp_inf_rollup
            LEFT JOIN bilirubin_from_measurement AS bilirubin
            USING (person_id)
            WHERE
                DATE_SUB(CAST(index_date AS DATE), INTERVAL 2 DAY) > CAST(measurement_DATETIME AS DATE) AND
                DATE_SUB(CAST(index_date AS DATE), INTERVAL 10 DAY) <= CAST(measurement_DATETIME AS DATE) AND
                value_as_number IS NOT NULL
        ),
        bilirubin_rollup AS (
            SELECT person_id, admit_date, MAX(value_as_number) as max_bilirubin_prior
            FROM bilirubin_window 
            GROUP BY person_id, admit_date
        ), 
        vasopressor_window AS (
            SELECT 
                susp_inf_rollup.person_id, 
                susp_inf_rollup.admit_date, 
                susp_inf_rollup.index_date,
                vasopressor.drug_exposure_start_DATETIME, 
                vasopressor.drug_exposure_end_DATETIME,
                datetime_diff(vasopressor.drug_exposure_start_DATETIME, index_date, DAY) as days_index_vasostart,
                datetime_diff(vasopressor.drug_exposure_end_DATETIME, index_date, DAY) as days_index_vasoend,
                (datetime_diff(vasopressor.drug_exposure_end_DATETIME, vasopressor.drug_exposure_start_DATETIME, DAY) + 1) as days_vasopressor
            -- FROM som-nero-phi-nigam-starr.jdposada_explore.susp_inf_rollup AS susp_inf_rollup
            FROM {suspected_infection} AS susp_inf_rollup
            LEFT JOIN ionotrope_from_drug_exposure_with_name AS vasopressor
            USING (person_id)
            WHERE
                DATE_SUB(CAST(index_date AS DATE), INTERVAL 3 DAY) BETWEEN CAST (drug_exposure_start_DATETIME AS DATE) AND CAST (drug_exposure_end_DATETIME AS DATE) OR
                DATE_SUB(CAST(index_date AS DATE), INTERVAL 4 DAY) BETWEEN CAST (drug_exposure_start_DATETIME AS DATE) AND CAST (drug_exposure_end_DATETIME AS DATE) OR
                DATE_SUB(CAST(index_date AS DATE), INTERVAL 5 DAY) BETWEEN CAST (drug_exposure_start_DATETIME AS DATE) AND CAST (drug_exposure_end_DATETIME AS DATE) OR
                DATE_SUB(CAST(index_date AS DATE), INTERVAL 6 DAY) BETWEEN CAST (drug_exposure_start_DATETIME AS DATE) AND CAST (drug_exposure_end_DATETIME AS DATE) OR
                DATE_SUB(CAST(index_date AS DATE), INTERVAL 7 DAY) BETWEEN CAST (drug_exposure_start_DATETIME AS DATE) AND CAST (drug_exposure_end_DATETIME AS DATE) OR
                DATE_SUB(CAST(index_date AS DATE), INTERVAL 8 DAY) BETWEEN CAST (drug_exposure_start_DATETIME AS DATE) AND CAST (drug_exposure_end_DATETIME AS DATE) OR
                DATE_SUB(CAST(index_date AS DATE), INTERVAL 9 DAY) BETWEEN CAST (drug_exposure_start_DATETIME AS DATE) AND CAST (drug_exposure_end_DATETIME AS DATE) OR
                DATE_SUB(CAST(index_date AS DATE), INTERVAL 10 DAY) BETWEEN CAST (drug_exposure_start_DATETIME AS DATE) AND CAST (drug_exposure_end_DATETIME AS DATE)
        ),
        vasopressor_rollup AS (
            SELECT 
                person_id, 
                admit_date, 
                MAX(datetime_diff(vasopressor.drug_exposure_end_DATETIME, vasopressor.drug_exposure_start_DATETIME, DAY) + 1)
            as max_vaso_days_prior
            FROM vasopressor_window as vasopressor 
            GROUP BY person_id, admit_date
        ),
        creatinine_window AS (
            SELECT 
                susp_inf_rollup.*, 
                creatinine.measurement_DATETIME AS creatinine_date, 
                creatinine.value_as_number,
                datetime_diff(creatinine.measurement_DATETIME, index_date, DAY) as days_crea_index
            -- FROM som-nero-phi-nigam-starr.jdposada_explore.susp_inf_rollup AS susp_inf_rollup
            FROM {suspected_infection} AS susp_inf_rollup
            LEFT JOIN creatinine_from_measurement AS creatinine
            USING (person_id)
            WHERE
                DATE_SUB(CAST(index_date AS DATE), INTERVAL 2 DAY) > CAST(measurement_DATETIME AS DATE) AND
                DATE_SUB(CAST(index_date AS DATE), INTERVAL 10 DAY) <= CAST(measurement_DATETIME AS DATE) AND
                value_as_number IS NOT NULL
        ),
        creatinine_rollup AS (
            SELECT person_id, admit_date, MAX(value_as_number) as max_creatinine_prior
            FROM creatinine_window 
            GROUP BY person_id, admit_date
        ),
        lactate_window AS (
            SELECT 
                susp_inf_rollup.*, 
                lactate.measurement_DATETIME AS lactate_date, 
                lactate.value_as_number,
                datetime_diff(lactate.measurement_DATETIME, index_date, DAY) as days_lact_index
            -- FROM som-nero-phi-nigam-starr.jdposada_explore.susp_inf_rollup AS susp_inf_rollup
            FROM {suspected_infection} AS susp_inf_rollup
            LEFT JOIN lactate_from_measurement AS lactate
            USING (person_id)
            WHERE
                DATE_SUB(CAST(index_date AS DATE), INTERVAL 2 DAY) > CAST(measurement_DATETIME AS DATE) AND
                DATE_SUB(CAST(index_date AS DATE), INTERVAL 10 DAY) <= CAST(measurement_DATETIME AS DATE) AND
                value_as_number IS NOT NULL
        ),
        lactate_rollup AS (
            SELECT person_id, admit_date, MAX(value_as_number) as max_lactate_prior
            FROM lactate_window 
            GROUP BY person_id, admit_date
        ),
        paO2_window AS (
            SELECT 
                susp_inf_rollup.*, 
                paO2.measurement_DATETIME AS paO2_datetime, 
                paO2.value_as_number as paO2,
                datetime_diff(paO2.measurement_DATETIME, index_date, DAY) as days_paO2_index
            -- FROM som-nero-phi-nigam-starr.jdposada_explore.susp_inf_rollup AS susp_inf_rollup
            FROM {suspected_infection} AS susp_inf_rollup
            LEFT JOIN paO2_from_measurement AS paO2
            USING (person_id)
            WHERE
                DATE_SUB(CAST(index_date AS DATE), INTERVAL 2 DAY) > CAST(measurement_DATETIME AS DATE) AND
                DATE_SUB(CAST(index_date AS DATE), INTERVAL 10 DAY) <= CAST(measurement_DATETIME AS DATE) AND
                value_as_number IS NOT NULL
        ),
        fiO2_window AS ( 
            SELECT 
                susp_inf_rollup.person_id, 
                min_bc, 
                min_systemic_abx, 
                susp_inf_rollup.admit_date, 
                index_date, 
                CAST(observation_datetime AS DATETIME) AS fiO2_datetime, 
                SAFE_CAST(meas_value AS float64) AS fiO2, 
                datetime_diff(CAST(observation_datetime AS DATETIME), index_date, DAY) as days_fiO2_index
            -- FROM som-nero-phi-nigam-starr.jdposada_explore.susp_inf_rollup AS susp_inf_rollup
            FROM {suspected_infection} AS susp_inf_rollup
            LEFT JOIN som-nero-phi-nigam-starr.jdposada_explore.flowsheets AS flowsheet
                ON susp_inf_rollup.person_id = CAST(flowsheet.person_id AS INT64) 
            WHERE
                DATE_SUB(CAST(index_date AS DATE), INTERVAL 2 DAY) > CAST(observation_datetime AS DATE) AND
                DATE_SUB(CAST(index_date AS DATE), INTERVAL 10 DAY) <= CAST(observation_datetime AS DATE) AND
                UPPER(row_disp_name) = "FIO2" AND 
                meas_value IS NOT NULL AND SAFE_CAST(meas_value AS FLOAT64) >=21 AND SAFE_CAST(meas_value AS FLOAT64) <=100
        ),
        paO2_fiO2_window AS (
            SELECT 
                paO2_window.person_id, 
                paO2_window.admit_date, 
                paO2_window.index_date, 
                fiO2, 
                fiO2_datetime, 
                paO2, 
                paO2_datetime,  
                paO2/(NULLIF(fiO2, 0))*100 AS paO2fiO2_ratio, 
                datetime_diff(paO2_datetime, fiO2_datetime, MINUTE) as minutes_fiO2_paO2
            FROM fiO2_window AS fiO2_window
            INNER JOIN paO2_window AS paO2_window
            USING (person_id, index_date)
            WHERE CAST(fiO2_datetime AS DATETIME)<= paO2_datetime 
        ),
        paO2_fiO2_initial_rollup AS (
            SELECT person_id, admit_date, paO2_datetime, MIN(minutes_fiO2_paO2) As minutes_fiO2_paO2
            FROM paO2_fiO2_window 
            GROUP BY person_id, admit_date, paO2_datetime
        ),
        paO2_fiO2_initial_rollup_join AS (
            SELECT initial_rollup.person_id, initial_rollup.admit_date, initial_rollup.paO2_datetime, initial_rollup.minutes_fiO2_paO2,
                index_date, fiO2, fiO2_datetime, paO2, paO2fiO2_ratio 
            FROM paO2_fiO2_initial_rollup AS initial_rollup
            LEFT JOIN paO2_fiO2_window AS combined_window
            USING (person_id, paO2_datetime, minutes_fiO2_paO2)
        ),
        paO2_fiO2_rollup AS (
            SELECT person_id, CAST(admit_date AS DATE) as admit_date, MIN(paO2fiO2_ratio) as min_paO2fiO2_ratio_prior
            FROM paO2_fiO2_initial_rollup_join 
            WHERE minutes_fiO2_paO2 <= 24*60
            GROUP BY person_id, admit_date
        ),
        glasgow_coma_scale_window AS (
            SELECT 
                susp_inf_rollup.*, 
                CAST(flowsheet.observation_datetime AS DATETIME) AS gcs_datetime, 
                CAST(flowsheet.meas_value AS FLOAT64) AS glascow_coma_scale,
                datetime_diff(CAST(observation_datetime AS DATETIME), index_date, DAY) as days_gcs_index
            -- FROM som-nero-phi-nigam-starr.jdposada_explore.susp_inf_rollup AS susp_inf_rollup
            FROM {suspected_infection} AS susp_inf_rollup
            LEFT JOIN glascow_coma_scale_from_flowsheet as flowsheet
            ON susp_inf_rollup.person_id =  CAST(flowsheet.person_id AS INT64)
            WHERE
                DATE_SUB(CAST(index_date AS DATE), INTERVAL 2 DAY) > CAST(CAST(observation_datetime AS DATETIME) AS DATE) AND
                DATE_SUB(CAST(index_date AS DATE), INTERVAL 10 DAY) <= CAST(CAST(observation_datetime AS DATETIME) AS DATE) AND
                CAST(flowsheet.meas_value AS FLOAT64) >= 3 AND flowsheet.meas_value IS NOT NULL
        ),
        glasgow_coma_scale_rollup AS (
            SELECT person_id, admit_date, MIN(glascow_coma_scale) as min_gcs_prior
            FROM glasgow_coma_scale_window 
            GROUP BY person_id, admit_date
        ),
        mean_arterial_pressure_window AS (
            SELECT 
                susp_inf_rollup.*, 
                CAST(flowsheet.observation_datetime AS DATETIME) AS map_datetime, 
                CAST(flowsheet.meas_value AS FLOAT64) AS map,
                datetime_diff(CAST(observation_datetime AS DATETIME), index_date, DAY) as days_map_index,
            -- FROM som-nero-phi-nigam-starr.jdposada_explore.susp_inf_rollup AS susp_inf_rollup
            FROM {suspected_infection} AS susp_inf_rollup
            LEFT JOIN mean_arterial_pressure_from_flowsheet AS flowsheet
            ON susp_inf_rollup.person_id = CAST(flowsheet.person_id AS INT64)
            WHERE
                DATE_SUB(CAST(index_date AS DATE), INTERVAL 2 DAY) > CAST(CAST(observation_datetime AS DATETIME) AS DATE) AND
                DATE_SUB(CAST(index_date AS DATE), INTERVAL 10 DAY) <= CAST(CAST(observation_datetime AS DATETIME) AS DATE) AND
                flowsheet.meas_value IS NOT NULL
        ),
        mean_arterial_pressure_rollup AS (
            SELECT person_id, admit_date, MIN(map) as min_map_prior
            FROM mean_arterial_pressure_window 
            WHERE map >=10
            GROUP BY person_id, admit_date
        ),
        mech_vent_window AS (
            SELECT 
                susp_inf_rollup.*, 
                mech_vent.observation_datetime AS mech_vent_datetime, 
                mech_vent.meas_value as vent_mode,
                datetime_diff(mech_vent.observation_datetime, index_date, DAY) as days_mech_vent_index,
            -- FROM som-nero-phi-nigam-starr.jdposada_explore.susp_inf_rollup AS susp_inf_rollup
            FROM {suspected_infection} AS susp_inf_rollup
            LEFT JOIN mech_vent_from_flowsheet AS mech_vent
            USING (person_id)
            WHERE
                DATE_SUB(CAST(index_date AS DATE), INTERVAL 2 DAY) > CAST(CAST(observation_datetime AS DATETIME) AS DATE) AND
                DATE_SUB(CAST(index_date AS DATE), INTERVAL 10 DAY) <= CAST(CAST(observation_datetime AS DATETIME) AS DATE) 
        ),
        mech_vent_rollup AS (
            SELECT person_id, admit_date, COUNT(vent_mode) as count_vent_mode_prior
            FROM mech_vent_window 
            GROUP BY person_id, admit_date
        ),
        urine_output_window AS (
            SELECT 
                susp_inf_rollup.*, 
                CAST(flowsheet.observation_datetime AS DATETIME) AS urine_datetime, 
                SAFE_CAST(flowsheet.meas_value AS FLOAT64) AS urine_volume,
                datetime_diff(CAST(observation_datetime AS DATETIME), index_date, DAY) as days_urine_index
            -- FROM som-nero-phi-nigam-starr.jdposada_explore.susp_inf_rollup AS susp_inf_rollup
            FROM {suspected_infection} AS susp_inf_rollup
            LEFT JOIN som-nero-phi-nigam-starr.jdposada_explore.flowsheets AS flowsheet
            ON susp_inf_rollup.person_id = CAST(flowsheet.person_id AS INT64) 
            WHERE
                DATE_SUB(CAST(index_date AS DATE), INTERVAL 2 DAY) > CAST(observation_datetime AS DATE) AND
                DATE_SUB(CAST(index_date AS DATE), INTERVAL 10 DAY) <= CAST(observation_datetime AS DATE) AND
                UPPER(row_disp_name) IN ("URINE OUTPUT (ML)", "URINE") AND meas_value IS NOT NULL AND SAFE_CAST(meas_value AS FLOAT64) >=0
            ORDER BY person_id, admit_date, observation_datetime 
        ),
        admit_time AS (
            SELECT 
                person_id, MIN(observation_datetime) AS ext_urine_datetime,
                EXTRACT(HOUR FROM MIN(observation_datetime)) AS hour, (24-EXTRACT(HOUR FROM MIN(observation_datetime))) AS adjust_hours
            FROM urine_output_window AS urine
            LEFT JOIN som-nero-phi-nigam-starr.jdposada_explore.flowsheets AS flowsheets_orig  USING (person_id)
            WHERE CAST(admit_date AS DATE) = CAST(observation_datetime AS DATE)  
            AND observation_datetime <> DATETIME_TRUNC(observation_datetime, DAY)
            GROUP BY person_id
        ),
        discharge_time AS (
            SELECT 
                person_id, 
                MAX(observation_datetime) AS ext_urine_datetime,
                EXTRACT(HOUR FROM MAX(observation_datetime)) AS adjust_hours 
            FROM urine_output_window AS urine
            LEFT JOIN som-nero-phi-nigam-starr.jdposada_explore.flowsheets AS flowsheets_orig  USING (person_id)
            WHERE CAST(admit_date AS DATE) = CAST(observation_datetime AS DATE)  
            AND observation_datetime <> DATETIME_TRUNC(observation_datetime, DAY)
            GROUP BY person_id
        ),
        urine_output_initial_rollup AS (
            (
                SELECT person_id, admit_date, discharge_date, CAST(urine_datetime AS DATE) AS urine_date, 
                    SUM(urine_volume) as urine_daily_output_orig, SUM(urine_volume) as urine_daily_output_adj,
                    ext_urine_datetime, adjust_hours
                FROM urine_output_window 
                LEFT JOIN admit_time USING (person_id)
                WHERE CAST(urine_datetime AS DATE) <> CAST(admit_date AS DATE) AND CAST(urine_datetime AS DATE) <> CAST (discharge_date AS  DATE)
                GROUP BY person_id, admit_date, discharge_date, CAST (urine_datetime AS DATE), ext_urine_datetime, adjust_hours 
            )
            UNION ALL
            (
                SELECT 
                    person_id, admit_date, discharge_date, CAST(urine_datetime AS DATE) AS urine_date, 
                    SUM(urine_volume) as urine_daily_output_orig, (SUM(urine_volume))*24/adjust_hours as urine_daily_output_adj,
                    ext_urine_datetime, adjust_hours
                FROM urine_output_window 
                LEFT JOIN admit_time USING (person_id)
                WHERE CAST(urine_datetime AS DATE) = CAST(admit_date AS DATE) 
                GROUP BY person_id, admit_date, discharge_date, CAST(urine_datetime AS DATE), ext_urine_datetime, adjust_hours
            )
            UNION ALL
            (
                SELECT person_id, admit_date, discharge_date, CAST(urine_datetime AS DATE) AS urine_date, 
                    SUM(urine_volume) as urine_daily_output_orig, (SUM(urine_volume))*24/adjust_hours as urine_daily_output_adj,
                    ext_urine_datetime, adjust_hours
                FROM urine_output_window 
                LEFT JOIN discharge_time USING (person_id)
                WHERE CAST(urine_datetime AS DATE) = CAST(discharge_date AS DATE) AND adjust_hours <> 0 
                GROUP BY person_id, admit_date, discharge_date, CAST(urine_datetime AS DATE), ext_urine_datetime, adjust_hours
            )
            UNION ALL # THIS LAST BIT DEALS WITH DISCHARGE AT 0:00:00 HOURS
            (
                SELECT person_id, admit_date, discharge_date, CAST(urine_datetime AS DATE) AS urine_date, 
                    SUM(urine_volume) as urine_daily_output_orig, (SUM(urine_volume))*24 as urine_daily_output_adj,
                    ext_urine_datetime, adjust_hours
                FROM urine_output_window 
                LEFT JOIN discharge_time USING (person_id)
                WHERE CAST(urine_datetime AS DATE) = CAST(discharge_date AS DATE) AND adjust_hours = 0 
                GROUP BY person_id, admit_date, discharge_date, CAST(urine_datetime AS DATE), ext_urine_datetime, adjust_hours
            )
        ),
        urine_output_rollup AS (
            SELECT person_id, admit_date, Min(urine_daily_output_adj) as min_urine_daily_prior
            FROM urine_output_initial_rollup 
            GROUP BY person_id, admit_date
        )
        # Begin sepsis and shock cohort
        SELECT 
            susp_inf_rollup.person_id, 
            CAST(susp_inf_rollup.admit_date AS DATE) AS admit_date, 
            CAST(discharge_date AS DATE) AS discharge_date,
            min_bc, 
            min_systemic_abx, 
            index_date, 
            min_platelet_prior, 
            CASE 
                WHEN min_platelet_prior IS NULL THEN 0 
                WHEN min_platelet_prior <20 THEN 4 
                WHEN min_platelet_prior < 50 THEN 3
                WHEN min_platelet_prior < 100 THEN 2 
                WHEN min_platelet_prior < 150 THEN 1 
                ELSE 0 
            END plat_SOFA_prior,
            max_bilirubin_prior, 
            CASE 
                WHEN max_bilirubin_prior IS NULL THEN 0 
                WHEN max_bilirubin_prior >= 12 THEN 4 
                WHEN max_bilirubin_prior >= 6 THEN 3
                WHEN max_bilirubin_prior >= 2 THEN 2 
                WHEN max_bilirubin_prior >= 1.2 THEN 1 
                ELSE 0 
            END bili_SOFA_prior,
            max_creatinine_prior, 
            CASE 
                WHEN max_creatinine_prior IS NULL THEN 0 
                WHEN max_creatinine_prior >= 5 THEN 4 
                WHEN max_creatinine_prior >= 3.5 THEN 3
                WHEN max_creatinine_prior >= 2 THEN 2 
                WHEN max_creatinine_prior >= 1.2 THEN 1 
                ELSE 0 
            END crea_SOFA_prior,
            max_vaso_days_prior, 
            min_map_prior, 
            CASE 
                WHEN max_vaso_days_prior IS NOT NULL THEN 2
                WHEN min_map_prior <70 THEN 1 
                ELSE 0 
            END cv_SOFA_prior,
            min_paO2fiO2_ratio_prior, 
            count_vent_mode_prior, 
            CASE 
                WHEN (count_vent_mode_prior IS NOT NULL AND min_paO2fiO2_ratio_prior < 100) THEN 4
                WHEN count_vent_mode_prior IS NOT NULL THEN 3 
                WHEN min_paO2fiO2_ratio_prior < 300 THEN 2 
                WHEN min_paO2fiO2_ratio_prior < 400 THEN 1
                ELSE 0 
            END resp_SOFA_prior,
            min_gcs_prior, 
            CASE 
                WHEN min_gcs_prior IS NULL THEN 0 
                WHEN min_gcs_prior < 6 THEN 4 
                WHEN min_gcs_prior < 10 THEN 3
                WHEN min_gcs_prior < 13 THEN 2 
                WHEN min_gcs_prior < 15 THEN 1 
                ELSE 0 
            END gcs_SOFA_prior,
            min_urine_daily_prior, 
            CASE 
                WHEN min_urine_daily_prior IS NULL THEN 0 
                WHEN min_urine_daily_prior < 200 THEN 4 
                WHEN min_urine_daily_prior < 500 THEN 3 
                ELSE 0 
            END urine_SOFA_prior,
            CASE 
                WHEN max_vaso_days_prior IS NOT NULL THEN 1 
                ELSE 0 
            END vaso_shock_prior,
            CASE 
                WHEN min_map_prior < 65 THEN 1 
                ELSE 0 
            END map_shock_prior,
            max_lactate_prior, 
            CASE 
                WHEN max_lactate_prior > 2 THEN 1 
                ELSE 0 
            END lact_shock_prior, 
        -- FROM som-nero-phi-nigam-starr.jdposada_explore.susp_inf_rollup AS susp_inf
        FROM {suspected_infection} AS susp_inf_rollup
        LEFT JOIN platelet_rollup USING (person_id, admit_date)
        LEFT JOIN bilirubin_rollup USING (person_id, admit_date)
        LEFT JOIN creatinine_rollup USING (person_id, admit_date)
        LEFT JOIN vasopressor_rollup USING (person_id, admit_date)
        LEFT JOIN mean_arterial_pressure_rollup USING (person_id, admit_date)
        LEFT JOIN paO2_fiO2_rollup USING (person_id, admit_date)
        LEFT JOIN mech_vent_rollup USING (person_id, admit_date)
        LEFT JOIN glasgow_coma_scale_rollup USING (person_id, admit_date)
        LEFT JOIN urine_output_rollup USING (person_id, admit_date)
        LEFT JOIN lactate_rollup USING (person_id, admit_date)
        ORDER BY person_id, CAST(admit_date AS DATE)
        """
        if not format_query:
            return query
        else:
            return query.format_map(self.config_dict)

    def get_sepsis_cohort_combined(self, format_query=True):
        query = """
        SELECT 
            cohort.person_id, cohort.admit_date, cohort.discharge_date, cohort.min_bc, cohort.min_systemic_abx, cohort.index_date, 
            cohort.min_platelet, cohort.max_bilirubin, cohort.max_creatinine, cohort.max_vaso_days, 
            cohort.min_map, cohort.min_paO2fiO2_ratio, cohort.count_vent_mode, 
            cohort.min_gcs, cohort.min_urine_daily, cohort.max_lactate,
            prior.min_platelet_prior, prior.max_bilirubin_prior, prior.max_creatinine_prior, prior.max_vaso_days_prior, 
            prior.min_map_prior, prior.min_paO2fiO2_ratio_prior, prior.count_vent_mode_prior, 
            prior.min_gcs_prior, prior.min_urine_daily_prior, prior.max_lactate_prior,
            plat_SOFA, bili_SOFA, crea_SOFA, cv_SOFA, resp_SOFA, gcs_SOFA, urine_SOFA, 
            (plat_SOFA + bili_SOFA + crea_SOFA + cv_SOFA + resp_SOFA + gcs_SOFA + urine_SOFA) AS SOFA_score_current,
            plat_SOFA_prior, bili_SOFA_prior, crea_SOFA_prior, cv_SOFA_prior, resp_SOFA_prior, gcs_SOFA_prior, urine_SOFA_prior,  
            (plat_SOFA_prior + bili_SOFA_prior + crea_SOFA_prior + cv_SOFA_prior + 
            resp_SOFA_prior + gcs_SOFA_prior + urine_SOFA_prior) AS SOFA_score_prior,  
            ((plat_SOFA + bili_SOFA + crea_SOFA + cv_SOFA + resp_SOFA + gcs_SOFA + urine_SOFA) - 
            (plat_SOFA_prior + bili_SOFA_prior + crea_SOFA_prior + cv_SOFA_prior + 
            resp_SOFA_prior + gcs_SOFA_prior + urine_SOFA_prior)) AS SOFA_score_diff,
            CASE 
                WHEN (plat_SOFA + bili_SOFA + crea_SOFA + cv_SOFA + resp_SOFA + gcs_SOFA + urine_SOFA) >=2 THEN "Yes" 
                ELSE "No" 
            END SOFA_current,
            CASE 
                WHEN (plat_SOFA_prior + bili_SOFA_prior + crea_SOFA_prior + cv_SOFA_prior + 
                    resp_SOFA_prior + gcs_SOFA_prior + urine_SOFA_prior) >=2 THEN "Yes" 
                ELSE "No" 
            END SOFA_prior,
            CASE 
                WHEN ((plat_SOFA + bili_SOFA + crea_SOFA + cv_SOFA + resp_SOFA + gcs_SOFA + urine_SOFA) - 
                    (plat_SOFA_prior + bili_SOFA_prior + crea_SOFA_prior + cv_SOFA_prior + 
                    resp_SOFA_prior + gcs_SOFA_prior + urine_SOFA_prior)) >=2 THEN "Yes" 
                ELSE "No" 
            END SOFA_diff,
            vaso_shock, map_shock, lact_shock, (vaso_shock + map_shock + lact_shock) AS shock_score,
            vaso_shock_prior, map_shock_prior, lact_shock_prior , (vaso_shock_prior + map_shock_prior + lact_shock_prior ) as shock_score_prior,
            # Modification for peds to remove mean arterial pressure
            CASE WHEN (vaso_shock + lact_shock) = 2 THEN "Yes" ELSE "No" END shock,  
            CASE WHEN (vaso_shock_prior + lact_shock_prior) = 2 THEN "Yes" ELSE "No" END shock_prior,  
            CASE WHEN ((vaso_shock + lact_shock) - (vaso_shock_prior + lact_shock_prior)) = 2 THEN "Yes" ELSE "No" END shock_diff, 

            CASE WHEN plat_SOFA >= 2 THEN 1 ELSE 0 END plat_SOFA_GT2,
            CASE WHEN bili_SOFA >= 2 THEN 1 ELSE 0 END bili_SOFA_GT2,
            CASE WHEN crea_SOFA >= 2 THEN 1 ELSE 0 END crea_SOFA_GT2,
            CASE WHEN cv_SOFA >= 2 THEN 1 ELSE 0 END cv_SOFA_GT2,
            CASE WHEN resp_SOFA >= 2 THEN 1 ELSE 0 END resp_SOFA_GT2,
            CASE WHEN gcs_SOFA >= 2 THEN 1 ELSE 0 END gcs_SOFA_GT2,
            CASE WHEN urine_SOFA >= 2 THEN 1 ELSE 0 END urine_SOFA_GT2,

            CASE WHEN plat_SOFA - plat_SOFA_prior >= 2 THEN 1 ELSE 0 END plat_SOFA_GT2_diff,
            CASE WHEN bili_SOFA -bili_SOFA_prior >= 2 THEN 1 ELSE 0 END bili_SOFA_GT2_diff,
            CASE WHEN crea_SOFA - crea_SOFA_prior >= 2 THEN 1 ELSE 0 END crea_SOFA_GT2_diff,
            CASE WHEN cv_SOFA - cv_SOFA_prior >= 2 THEN 1 ELSE 0 END cv_SOFA_GT2_diff,
            CASE WHEN resp_SOFA - resp_SOFA_prior >= 2 THEN 1 ELSE 0 END resp_SOFA_GT2_diff,
            CASE WHEN gcs_SOFA - gcs_SOFA_prior >= 2 THEN 1 ELSE 0 END gcs_SOFA_GT2_diff,
            CASE WHEN urine_SOFA - urine_SOFA_prior >= 2 THEN 1 ELSE 0 END urine_SOFA_GT2_diff,

            CASE WHEN vaso_shock - vaso_shock_prior = 1 THEN 1 ELSE 0 END vaso_shock_diff,
            CASE WHEN map_shock - map_shock_prior = 1 THEN 1 ELSE 0 END map_shock_diff,
            CASE WHEN lact_shock - lact_shock_prior = 1 THEN 1 ELSE 0 END lact_shock_diff
            FROM 
                {sepsis_initial} as cohort
            -- `som-nero-phi-nigam-starr.jdposada_explore.sepsis_cohort` AS cohort
            LEFT JOIN
                {sepsis_prior} as prior USING (person_id, admit_date)
            -- `som-nero-phi-nigam-starr.jdposada_explore.preexisting_SOFA` AS prior USING (person_id, admit_date)
            ORDER BY
                person_id, CAST(admit_date AS DATE)
        """
        if not format_query:
            return query
        else:
            return query.format_map(self.config_dict)

    def get_cohort_with_demographics(self, format_query=True):
        query = """
            WITH gender_name AS (
                SELECT person.*, concept.concept_name AS gender_name,
                FROM `{dataset_project}.{dataset}.person` AS person 
                LEFT JOIN `{dataset_project}.{dataset}.concept` AS concept ON person.gender_concept_id = concept.concept_id
            ),
            race_name AS (
                SELECT person.*, concept.concept_name AS race_name,
                FROM `{dataset_project}.{dataset}.person` AS person
                LEFT JOIN `{dataset_project}.{dataset}.concept` AS concept ON person.race_concept_id = concept.concept_id
            ),
            ethnicity_name AS (
                SELECT person.*, concept.concept_name AS ethnicity_name,
                FROM `{dataset_project}.{dataset}.person` AS person 
                LEFT JOIN `{dataset_project}.{dataset}.concept` AS concept ON person.ethnicity_concept_id = concept.concept_id
            )
            SELECT 
                cohort.*, person.birth_DATETIME, gender_name.gender_name, race_name.race_name, ethnicity_name.ethnicity_name, 
                DATE_DIFF(CAST(admit_date AS DATE), CAST(person.birth_DATETIME AS DATE), YEAR) AS age_in_years,
            CASE 
                WHEN DATE_DIFF(CAST(admit_date AS DATE), CAST(person.birth_DATETIME AS DATE), YEAR) <= 18 THEN 0
                ELSE 1 
            END adult_at_admission
            FROM {sepsis_combined} AS cohort
            -- FROM `som-nero-phi-nigam-starr.jdposada_explore.sepsis_shock_cohort_consider_prior` AS cohort
            LEFT JOIN gender_name USING (person_id) 
            LEFT JOIN race_name USING (person_id)
            LEFT JOIN ethnicity_name USING (person_id)
            LEFT JOIN `{dataset_project}.{dataset}.person` AS person USING (person_id)
            ORDER BY person_id, CAST(admit_date AS DATE)
        """
        if not format_query:
            return query
        else:
            return query.format_map(self.config_dict)

    def get_cohort_with_pediatric(self, format_query=True):
        query="""
            WITH 
            crea_withpeds_prior AS (
                (
                    SELECT person_id, admit_date, 
                        CASE 
                            WHEN max_creatinine_prior IS NULL THEN 0 
                            WHEN max_creatinine_prior >= 1.6 THEN 4 
                            WHEN max_creatinine_prior >= 1.2 THEN 3
                            WHEN max_creatinine_prior >= 1.0 THEN 2 
                            WHEN max_creatinine_prior >= 0.8 THEN 1 
                            ELSE 0 
                        END crea_SOFA_prior_update
                    FROM `som-nero-phi-nigam-starr.jdposada_explore.sepsis_shock_cohort_with_person`
                    WHERE age_in_years*12 < 1
                )
                UNION ALL 
                (
                    SELECT person_id, admit_date, 
                    CASE WHEN max_creatinine_prior IS NULL THEN 0 WHEN max_creatinine_prior >= 1.2 THEN 4 WHEN max_creatinine_prior >= 0.8 THEN 3
                    WHEN max_creatinine_prior >= 0.5 THEN 2 WHEN max_creatinine_prior >= 0.3 THEN 1 ELSE 0 END crea_SOFA_prior_update
                    FROM `som-nero-phi-nigam-starr.jdposada_explore.sepsis_shock_cohort_with_person`
                    WHERE age_in_years*12 >= 1 AND age_in_years*12 < 12
                )
                UNION ALL 
                (
                    SELECT  person_id, admit_date, 
                    CASE WHEN max_creatinine_prior IS NULL THEN 0 WHEN max_creatinine_prior >= 1.5 THEN 4 WHEN max_creatinine_prior >= 1.1 THEN 3
                    WHEN max_creatinine_prior >= 0.6 THEN 2 WHEN max_creatinine_prior >= 0.4 THEN 1 ELSE 0 END crea_SOFA_prior_update
                    FROM {sepsis_combined_demographics}
                    WHERE age_in_years*12 >= 12 AND age_in_years*12 < 24
                )
                UNION ALL 
                (
                    SELECT  person_id, admit_date, 
                    CASE WHEN max_creatinine_prior IS NULL THEN 0 WHEN max_creatinine_prior >= 2.3 THEN 4 WHEN max_creatinine_prior >= 1.6 THEN 3
                    WHEN max_creatinine_prior >= 0.9 THEN 2 WHEN max_creatinine_prior >= 0.6 THEN 1 ELSE 0 END crea_SOFA_prior_update
                    FROM {sepsis_combined_demographics}
                    WHERE age_in_years*12 >= 24 AND age_in_years*12 < 60
                )
                UNION ALL 
                (
                    SELECT  person_id, admit_date, 
                    CASE WHEN max_creatinine_prior IS NULL THEN 0 WHEN max_creatinine_prior >= 2.6 THEN 4 WHEN max_creatinine_prior >= 1.8 THEN 3
                    WHEN max_creatinine_prior >= 1.1 THEN 2 WHEN max_creatinine_prior >= 0.7 THEN 1 ELSE 0 END crea_SOFA_prior_update
                    FROM {sepsis_combined_demographics}
                    WHERE age_in_years*12 >= 60 AND age_in_years*12 < 144
                )
                UNION ALL 
                (
                    SELECT  person_id, admit_date, 
                    CASE WHEN max_creatinine_prior IS NULL THEN 0 WHEN max_creatinine_prior >= 4.2 THEN 4 WHEN max_creatinine_prior >= 2.9 THEN 3
                    WHEN max_creatinine_prior >= 1.7 THEN 2 WHEN max_creatinine_prior >= 1.0 THEN 1 ELSE 0 END crea_SOFA_prior_update
                    FROM {sepsis_combined_demographics}
                    WHERE age_in_years*12 >= 144 AND age_in_years*12 < 216
                )
                UNION ALL 
                (
                    SELECT  person_id, admit_date, crea_SOFA_prior AS crea_SOFA_prior_update
                    FROM {sepsis_combined_demographics}
                    WHERE age_in_years*12 >= 216
                )
            ),
            crea_withpeds AS (
                SELECT  person_id, admit_date, 
                CASE WHEN max_creatinine IS NULL THEN 0 WHEN max_creatinine >= 1.6 THEN 4 WHEN max_creatinine >= 1.2 THEN 3
                WHEN max_creatinine >= 1.0 THEN 2 WHEN max_creatinine >= 0.8 THEN 1 ELSE 0 END crea_SOFA_update
                FROM {sepsis_combined_demographics}
                WHERE age_in_years*12 < 1
                UNION ALL 
                SELECT  person_id, admit_date, 
                CASE WHEN max_creatinine IS NULL THEN 0 WHEN max_creatinine >= 1.2 THEN 4 WHEN max_creatinine >= 0.8 THEN 3
                WHEN max_creatinine >= 0.5 THEN 2 WHEN max_creatinine >= 0.3 THEN 1 ELSE 0 END crea_SOFA_update
                FROM {sepsis_combined_demographics}
                WHERE age_in_years*12 >= 1 AND age_in_years*12 < 12
                UNION ALL 
                SELECT  person_id, admit_date, 
                CASE WHEN max_creatinine IS NULL THEN 0 WHEN max_creatinine >= 1.5 THEN 4 WHEN max_creatinine >= 1.1 THEN 3
                WHEN max_creatinine >= 0.6 THEN 2 WHEN max_creatinine >= 0.4 THEN 1 ELSE 0 END crea_SOFA_update
                FROM {sepsis_combined_demographics}
                WHERE age_in_years*12 >= 12 AND age_in_years*12 < 24
                UNION ALL 
                SELECT  person_id, admit_date, 
                CASE WHEN max_creatinine IS NULL THEN 0 WHEN max_creatinine >= 2.3 THEN 4 WHEN max_creatinine >= 1.6 THEN 3
                WHEN max_creatinine >= 0.9 THEN 2 WHEN max_creatinine >= 0.6 THEN 1 ELSE 0 END crea_SOFA_update
                FROM {sepsis_combined_demographics}
                WHERE age_in_years*12 >= 24 AND age_in_years*12 < 60
                UNION ALL 
                SELECT  person_id, admit_date, 
                CASE WHEN max_creatinine IS NULL THEN 0 WHEN max_creatinine >= 2.6 THEN 4 WHEN max_creatinine >= 1.8 THEN 3
                WHEN max_creatinine >= 1.1 THEN 2 WHEN max_creatinine >= 0.7 THEN 1 ELSE 0 END crea_SOFA_update
                FROM {sepsis_combined_demographics}
                WHERE age_in_years*12 >= 60 AND age_in_years*12 < 144
                UNION ALL 
                SELECT  person_id, admit_date, 
                CASE WHEN max_creatinine IS NULL THEN 0 WHEN max_creatinine >= 4.2 THEN 4 WHEN max_creatinine >= 2.9 THEN 3
                WHEN max_creatinine >= 1.7 THEN 2 WHEN max_creatinine >= 1.0 THEN 1 ELSE 0 END crea_SOFA_update
                FROM {sepsis_combined_demographics}
                WHERE age_in_years*12 >= 144 AND age_in_years*12 < 216
                UNION ALL 
                SELECT  person_id, admit_date, crea_SOFA AS crea_SOFA_update
                FROM {sepsis_combined_demographics}
                WHERE age_in_years*12 >= 216
            ),
            cv_withpeds_prior AS (
                SELECT person_id, admit_date,
                CASE WHEN max_vaso_days_prior IS NOT NULL THEN 2 WHEN min_map_prior <46 THEN 1 ELSE 0 END cv_SOFA_prior_update
                FROM {sepsis_combined_demographics}
                WHERE age_in_years*12 < 1
                UNION ALL
                SELECT person_id, admit_date,
                CASE WHEN max_vaso_days_prior IS NOT NULL THEN 2 WHEN min_map_prior <55 THEN 1 ELSE 0 END cv_SOFA_prior_update
                FROM {sepsis_combined_demographics}
                WHERE age_in_years*12 >= 1 AND age_in_years*12 < 12  
                UNION ALL
                SELECT person_id, admit_date,
                CASE WHEN max_vaso_days_prior IS NOT NULL THEN 2 WHEN min_map_prior <60 THEN 1 ELSE 0 END cv_SOFA_prior_update
                FROM {sepsis_combined_demographics}
                WHERE age_in_years*12 >= 12 AND age_in_years*12 < 24
                UNION ALL
                SELECT person_id, admit_date,
                CASE WHEN max_vaso_days_prior IS NOT NULL THEN 2 WHEN min_map_prior <62 THEN 1 ELSE 0 END cv_SOFA_prior_update
                FROM {sepsis_combined_demographics}
                WHERE age_in_years*12 >= 24 AND age_in_years*12 < 60
                UNION ALL
                SELECT person_id, admit_date,
                CASE WHEN max_vaso_days_prior IS NOT NULL THEN 2 WHEN min_map_prior <65 THEN 1 ELSE 0 END cv_SOFA_prior_update
                FROM {sepsis_combined_demographics}
                WHERE age_in_years*12 >= 60 AND age_in_years*12 < 144
                UNION ALL
                SELECT person_id, admit_date,
                CASE WHEN max_vaso_days_prior IS NOT NULL THEN 2 WHEN min_map_prior <67 THEN 1 ELSE 0 END cv_SOFA_prior_update
                FROM {sepsis_combined_demographics}
                WHERE age_in_years*12 >= 144 AND age_in_years*12 < 216
                UNION ALL 
                SELECT  person_id, admit_date, cv_SOFA_prior AS cv_SOFA_prior_update
                FROM {sepsis_combined_demographics}
                WHERE age_in_years*12 >= 216
            ),
            cv_withpeds AS (
                SELECT person_id, admit_date,
                CASE WHEN max_vaso_days IS NOT NULL THEN 2 WHEN min_map <46 THEN 1 ELSE 0 END cv_SOFA_update
                FROM {sepsis_combined_demographics}
                WHERE age_in_years*12 < 1
                UNION ALL
                SELECT person_id, admit_date,
                CASE WHEN max_vaso_days IS NOT NULL THEN 2 WHEN min_map <55 THEN 1 ELSE 0 END cv_SOFA_update
                FROM {sepsis_combined_demographics}
                WHERE age_in_years*12 >= 1 AND age_in_years*12 < 12  
                UNION ALL
                SELECT person_id, admit_date,
                CASE WHEN max_vaso_days IS NOT NULL THEN 2 WHEN min_map <60 THEN 1 ELSE 0 END cv_SOFA_update
                FROM {sepsis_combined_demographics}
                WHERE age_in_years*12 >= 12 AND age_in_years*12 < 24
                UNION ALL
                SELECT person_id, admit_date,
                CASE WHEN max_vaso_days IS NOT NULL THEN 2 WHEN min_map <62 THEN 1 ELSE 0 END cv_SOFA_update
                FROM {sepsis_combined_demographics}
                WHERE age_in_years*12 >= 24 AND age_in_years*12 < 60
                UNION ALL
                SELECT person_id, admit_date,
                CASE WHEN max_vaso_days IS NOT NULL THEN 2 WHEN min_map <65 THEN 1 ELSE 0 END cv_SOFA_update
                FROM {sepsis_combined_demographics}
                WHERE age_in_years*12 >= 60 AND age_in_years*12 < 144
                UNION ALL
                SELECT person_id, admit_date,
                CASE WHEN max_vaso_days IS NOT NULL THEN 2 WHEN min_map <67 THEN 1 ELSE 0 END cv_SOFA_update
                FROM {sepsis_combined_demographics}
                WHERE age_in_years*12 >= 144 AND age_in_years*12 < 216
                UNION ALL 
                SELECT  person_id, admit_date, cv_SOFA AS cv_SOFA_update
                FROM {sepsis_combined_demographics}
                WHERE age_in_years*12 >= 216
            )
            SELECT *,

            # Modification to include pediatrics by removing urine output
            CASE WHEN (plat_SOFA + bili_SOFA + crea_SOFA_update + cv_SOFA_update + resp_SOFA + gcs_SOFA) >=2 THEN "Yes" 
            ELSE "No" END SOFA_current_update,
            CASE WHEN (plat_SOFA_prior + bili_SOFA_prior + crea_SOFA_prior_update + cv_SOFA_prior_update + 
            resp_SOFA_prior + gcs_SOFA_prior) >=2 THEN "Yes" ELSE "No" END SOFA_prior_update,
            CASE WHEN ((plat_SOFA + bili_SOFA + crea_SOFA_update + cv_SOFA_update + resp_SOFA + gcs_SOFA) - 
            (plat_SOFA_prior + bili_SOFA_prior + crea_SOFA_prior_update + cv_SOFA_prior_update + 
            resp_SOFA_prior + gcs_SOFA_prior)) >=2 THEN "Yes" ELSE "No" END SOFA_diff_update,

            (plat_SOFA + bili_SOFA + crea_SOFA_update + cv_SOFA_update + resp_SOFA + gcs_SOFA) AS SOFA_score_current_update,
            (plat_SOFA_prior + bili_SOFA_prior + crea_SOFA_prior_update + cv_SOFA_prior_update + 
            resp_SOFA_prior + gcs_SOFA_prior) AS SOFA_score_prior_update,  
            ((plat_SOFA + bili_SOFA + crea_SOFA_update + cv_SOFA_update + resp_SOFA + gcs_SOFA) - 
            (plat_SOFA_prior + bili_SOFA_prior + crea_SOFA_prior_update + cv_SOFA_prior_update + 
            resp_SOFA_prior + gcs_SOFA_prior)) AS SOFA_score_diff_update,

            CASE WHEN crea_SOFA_update >= 2 THEN 1 ELSE 0 END crea_SOFA_GT2_update,
            CASE WHEN cv_SOFA_update >= 2 THEN 1 ELSE 0 END cv_SOFA_GT2_update,
            CASE WHEN crea_SOFA_update - crea_SOFA_prior_update >= 2 THEN 1 ELSE 0 END crea_SOFA_GT2_diff_update,
            CASE WHEN cv_SOFA_update - cv_SOFA_prior_update >= 2 THEN 1 ELSE 0 END cv_SOFA_GT2_diff_update,

            FROM {sepsis_combined_demographics}
            LEFT JOIN crea_withpeds_prior USING (person_id, admit_date)
            LEFT JOIN crea_withpeds USING (person_id, admit_date)
            LEFT JOIN cv_withpeds_prior USING (person_id, admit_date)
            LEFT JOIN cv_withpeds USING (person_id, admit_date)
        """
        if not format_query:
            return query
        else:
            return query.format_map(self.config_dict)

    def get_admission_sepsis_cohort(self, format_query=True):
        query = """
            SELECT 
                admit.person_id, admit.admit_date, admit.discharge_date,
                sepsis.min_bc, sepsis.min_systemic_abx, sepsis.index_date AS infection_date,
                sepsis.SOFA_diff_update, sepsis.SOFA_current_update, sepsis.SOFA_prior_update, 
                sepsis.SOFA_score_diff_update, sepsis.SOFA_score_current_update, sepsis.SOFA_score_prior_update,
                sepsis.shock_diff, sepsis.shock, sepsis.shock_prior,
                DATE_DIFF(CAST(admit.admit_date AS DATE), CAST(index_date AS DATE), DAY) AS days_admit_index,
                CASE 
                    WHEN index_date IS NULL THEN 0 
                    WHEN  DATE_DIFF(CAST(admit.admit_date AS DATE), CAST(index_date AS DATE), DAY) >= -3 THEN 1
                    ELSE 0 
                END early_infection, 
                CASE 
                    WHEN index_date IS NULL THEN 0 
                    WHEN  (DATE_DIFF(CAST(admit.admit_date AS DATE), CAST(index_date AS DATE), DAY) >= -3 AND SOFA_diff_update = "Yes") THEN 1
                ELSE 0 END early_sepsis,
                person.birth_DATETIME, DATE_DIFF(CAST(admit.admit_date AS DATE), CAST(person.birth_DATETIME AS DATE), YEAR) AS age_in_years,
                CASE 
                    WHEN DATE_DIFF(CAST(admit.admit_date AS DATE), CAST(person.birth_DATETIME AS DATE), YEAR) <= 18 THEN 0 
                    ELSE 1 
                END adult_at_admission
            FROM {admission_rollup}  AS admit
            LEFT JOIN {sepsis_with_pediatric} AS sepsis
                ON (admit.person_id = sepsis.person_id AND CAST(admit.admit_date AS DATE) = CAST (sepsis.admit_DATE AS DATE))
            LEFT JOIN `{dataset_project}.{dataset}.person` AS person ON admit.person_id = person.person_id
            ORDER BY person_id, CAST(admit.admit_date AS DATE), index_date
        """
        if not format_query:
            return query
        else:
            return query.format_map(self.config_dict)

    def get_sampled_cohort(self, format_query=True):
        query = """
            SELECT * EXCEPT (rnd, pos), 
            FARM_FINGERPRINT(GENERATE_UUID()) as prediction_id
            FROM (
                SELECT *, ROW_NUMBER() OVER(PARTITION BY person_id ORDER BY rnd) AS pos
                FROM (
                    SELECT 
                        *,
                        FARM_FINGERPRINT(CONCAT(CAST(person_id AS STRING), CAST(admit_date AS STRING), CAST(discharge_date AS STRING))) as rnd
                    FROM {admission_sepsis_cohort}
                )
            )
            WHERE pos = 1
            ORDER BY person_id, admit_date
        """
        if not format_query:
            return query
        else:
            return query.format_map(self.config_dict)
