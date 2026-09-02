-- =======================================================================
-- 01_export_clinical_events.sql
-- Export patient clinical event sequences from OMOP CDM for GNN/TRANS
--
-- Produces 4 files (run each COPY block separately in psql):
--   clinical_visits.csv       -- one row per visit per patient
--   clinical_conditions.csv   -- diagnosis codes per visit
--   clinical_procedures.csv   -- procedure codes per visit
--   clinical_drugs.csv        -- drug codes per visit
--   clinical_measurements.csv -- lab results per visit (numeric summary)
--
-- Cohort: all patients with >= 2 inpatient visits
-- =======================================================================

-- -----------------------------------------------------------------------
-- Step 1: Build cohort (patients with >= 2 visits, matching TRANS criteria)
-- -----------------------------------------------------------------------
DROP TABLE IF EXISTS export_cohort;
CREATE TEMP TABLE export_cohort AS
SELECT
    p.person_id,
    p.person_source_value                          AS subject_id,
    p.year_of_birth,
    p.gender_concept_id,
    p.race_concept_id,
    p.ethnicity_concept_id,
    COUNT(DISTINCT vo.visit_occurrence_id)         AS visit_count
FROM omopcdm.person p
JOIN omopcdm.visit_occurrence vo
    ON vo.person_id = p.person_id
    AND vo.visit_concept_id IN (
        9201,   -- Inpatient Visit
        9203,   -- Emergency Room and Inpatient Visit
        262     -- Emergency Room and Inpatient Visit (alt)
    )
GROUP BY p.person_id, p.person_source_value, p.year_of_birth,
         p.gender_concept_id, p.race_concept_id, p.ethnicity_concept_id
HAVING COUNT(DISTINCT vo.visit_occurrence_id) >= 2;

SELECT COUNT(*) AS cohort_size FROM export_cohort;

-- -----------------------------------------------------------------------
-- Export 1: clinical_visits.csv
-- One row per visit — temporal backbone for patient trajectories
-- -----------------------------------------------------------------------
COPY (
    SELECT
        ec.subject_id,
        ec.person_id,
        vo.visit_occurrence_id                                  AS hadm_id,
        vo.visit_source_value                                   AS hadm_source_id,
        vo.visit_start_date                                     AS admittime,
        vo.visit_end_date                                       AS dischtime,
        vo.visit_start_datetime,
        vo.visit_end_datetime,
        EXTRACT(EPOCH FROM (vo.visit_end_datetime - vo.visit_start_datetime))/3600 AS los_hours,
        vo.visit_concept_id,
        vc.concept_name                                         AS visit_type,
        vo.admitted_from_concept_id,
        vo.discharged_to_concept_id,
        ec.year_of_birth,
        ec.gender_concept_id,
        ec.race_concept_id,
        ec.ethnicity_concept_id,
        -- Death label: did patient die during or within 30 days of visit?
        CASE WHEN d.death_date IS NOT NULL
              AND d.death_date <= vo.visit_end_date + INTERVAL '30 days'
             THEN 1 ELSE 0 END                                  AS mortality_label,
        -- Readmission label: was there another visit within 30 days?
        CASE WHEN next_vo.visit_occurrence_id IS NOT NULL
             THEN 1 ELSE 0 END                                  AS readmission_30d_label,
        -- Visit index within patient (for temporal ordering)
        ROW_NUMBER() OVER (
            PARTITION BY ec.person_id
            ORDER BY vo.visit_start_datetime
        )                                                       AS visit_index
    FROM export_cohort ec
    JOIN omopcdm.visit_occurrence vo ON vo.person_id = ec.person_id
    LEFT JOIN omopcdm.concept vc ON vc.concept_id = vo.visit_concept_id
    LEFT JOIN omopcdm.death d ON d.person_id = ec.person_id
    LEFT JOIN omopcdm.visit_occurrence next_vo
        ON next_vo.person_id = ec.person_id
        AND next_vo.visit_occurrence_id != vo.visit_occurrence_id
        AND next_vo.visit_start_date > vo.visit_end_date
        AND next_vo.visit_start_date <= vo.visit_end_date + INTERVAL '30 days'
    WHERE vo.visit_concept_id IN (9201, 9203, 262)
    ORDER BY ec.subject_id, vo.visit_start_datetime
) TO 'C:/omop_export/clinical_visits.csv' CSV HEADER;

-- -----------------------------------------------------------------------
-- Export 2: clinical_conditions.csv
-- Diagnosis codes per visit — maps to TRANS 'conditions' / 'cond_hist'
-- -----------------------------------------------------------------------
COPY (
    SELECT
        ec.subject_id,
        ec.person_id,
        co.visit_occurrence_id                                  AS hadm_id,
        co.condition_start_date,
        co.condition_concept_id,
        c.concept_code                                          AS icd_code,
        c.vocabulary_id                                         AS icd_version,
        c.concept_name                                          AS condition_name,
        c.domain_id,
        -- CCS-equivalent: use condition_source_value for raw ICD
        co.condition_source_value                               AS icd_source_code,
        co.condition_source_concept_id,
        co.condition_type_concept_id,
        co.condition_status_concept_id
    FROM export_cohort ec
    JOIN omopcdm.condition_occurrence co ON co.person_id = ec.person_id
    JOIN omopcdm.visit_occurrence vo
        ON vo.visit_occurrence_id = co.visit_occurrence_id
        AND vo.visit_concept_id IN (9201, 9203, 262)
    LEFT JOIN omopcdm.concept c ON c.concept_id = co.condition_concept_id
    ORDER BY ec.subject_id, co.visit_occurrence_id, co.condition_start_date
) TO 'C:/omop_export/clinical_conditions.csv' CSV HEADER;

-- -----------------------------------------------------------------------
-- Export 3: clinical_procedures.csv
-- Procedure codes per visit — maps to TRANS 'procedures'
-- -----------------------------------------------------------------------
COPY (
    SELECT
        ec.subject_id,
        ec.person_id,
        po.visit_occurrence_id                                  AS hadm_id,
        po.procedure_date,
        po.procedure_concept_id,
        c.concept_code                                          AS procedure_code,
        c.vocabulary_id                                         AS procedure_vocab,
        c.concept_name                                          AS procedure_name,
        po.procedure_source_value                               AS procedure_source_code,
        po.procedure_source_concept_id,
        po.procedure_type_concept_id,
        po.modifier_concept_id
    FROM export_cohort ec
    JOIN omopcdm.procedure_occurrence po ON po.person_id = ec.person_id
    JOIN omopcdm.visit_occurrence vo
        ON vo.visit_occurrence_id = po.visit_occurrence_id
        AND vo.visit_concept_id IN (9201, 9203, 262)
    LEFT JOIN omopcdm.concept c ON c.concept_id = po.procedure_concept_id
    ORDER BY ec.subject_id, po.visit_occurrence_id, po.procedure_date
) TO 'C:/omop_export/clinical_procedures.csv' CSV HEADER;

-- -----------------------------------------------------------------------
-- Export 4: clinical_drugs.csv
-- Drug exposures per visit — maps to TRANS 'drugs' (NDC/RxNorm → ATC)
-- -----------------------------------------------------------------------
COPY (
    SELECT
        ec.subject_id,
        ec.person_id,
        de.visit_occurrence_id                                  AS hadm_id,
        de.drug_exposure_start_date,
        de.drug_exposure_end_date,
        de.drug_concept_id,
        c.concept_code                                          AS drug_code,
        c.vocabulary_id                                         AS drug_vocab,
        c.concept_name                                          AS drug_name,
        de.drug_source_value                                    AS ndc_source_code,
        de.drug_source_concept_id,
        de.drug_type_concept_id,
        de.route_concept_id,
        de.quantity,
        de.days_supply
    FROM export_cohort ec
    JOIN omopcdm.drug_exposure de ON de.person_id = ec.person_id
    JOIN omopcdm.visit_occurrence vo
        ON vo.visit_occurrence_id = de.visit_occurrence_id
        AND vo.visit_concept_id IN (9201, 9203, 262)
    LEFT JOIN omopcdm.concept c ON c.concept_id = de.drug_concept_id
    ORDER BY ec.subject_id, de.visit_occurrence_id, de.drug_exposure_start_date
) TO 'C:/omop_export/clinical_drugs.csv' CSV HEADER;

-- -----------------------------------------------------------------------
-- Export 5: clinical_measurements.csv
-- Lab results per visit — numeric summary statistics
-- (Not in TRANS baseline but valuable for your extension)
-- -----------------------------------------------------------------------
COPY (
    SELECT
        ec.subject_id,
        ec.person_id,
        m.visit_occurrence_id                                   AS hadm_id,
        m.measurement_date,
        m.measurement_concept_id,
        c.concept_code                                          AS lab_code,
        c.concept_name                                          AS lab_name,
        m.value_as_number,
        m.unit_concept_id,
        uc.concept_name                                         AS unit_name,
        m.value_source_value,
        m.measurement_source_value                              AS lab_source_code,
        m.range_low,
        m.range_high,
        -- Abnormality flag
        CASE
            WHEN m.value_as_number IS NOT NULL
             AND m.range_low IS NOT NULL
             AND m.range_high IS NOT NULL
            THEN CASE
                WHEN m.value_as_number < m.range_low  THEN 'LOW'
                WHEN m.value_as_number > m.range_high THEN 'HIGH'
                ELSE 'NORMAL'
            END
            ELSE NULL
        END                                                     AS abnormal_flag
    FROM export_cohort ec
    JOIN omopcdm.measurement m ON m.person_id = ec.person_id
    JOIN omopcdm.visit_occurrence vo
        ON vo.visit_occurrence_id = m.visit_occurrence_id
        AND vo.visit_concept_id IN (9201, 9203, 262)
    LEFT JOIN omopcdm.concept c  ON c.concept_id  = m.measurement_concept_id
    LEFT JOIN omopcdm.concept uc ON uc.concept_id = m.unit_concept_id
    WHERE m.value_as_number IS NOT NULL
    ORDER BY ec.subject_id, m.visit_occurrence_id, m.measurement_date
) TO 'C:/omop_export/clinical_measurements.csv' CSV HEADER;
