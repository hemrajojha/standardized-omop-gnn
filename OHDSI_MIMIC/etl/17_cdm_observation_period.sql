-- =======================================================================
-- 17_cdm_observation_period.sql
-- Derive observation periods from all populated event tables.
-- One row per person covering the full span of their records.
--
-- Adapted from: https://github.com/OHDSI/MIMIC/blob/main/etl/etl/cdm_observation_period.sql
-- =======================================================================

DROP TABLE IF EXISTS staging.tmp_observation_period_clean;
CREATE TABLE staging.tmp_observation_period_clean AS

SELECT person_id, CAST(visit_start_date AS DATE) AS start_date, CAST(visit_end_date AS DATE) AS end_date, unit_id
FROM staging.cdm_visit_occurrence

UNION ALL
SELECT person_id, condition_start_date, condition_end_date, unit_id
FROM staging.cdm_condition_occurrence

UNION ALL
SELECT person_id, procedure_date, procedure_date, unit_id
FROM staging.cdm_procedure_occurrence

UNION ALL
SELECT person_id, drug_exposure_start_date, drug_exposure_end_date, unit_id
FROM staging.cdm_drug_exposure

UNION ALL
SELECT person_id, measurement_date, measurement_date, unit_id
FROM staging.cdm_measurement

UNION ALL
SELECT person_id, death_date, death_date, unit_id
FROM staging.cdm_death
;

-- -----------------------------------------------------------------------
-- cdm_observation_period
-- -----------------------------------------------------------------------
DROP TABLE IF EXISTS staging.cdm_observation_period;
CREATE TABLE staging.cdm_observation_period (
    observation_period_id           BIGINT  NOT NULL,
    person_id                       BIGINT  NOT NULL,
    observation_period_start_date   DATE    NOT NULL,
    observation_period_end_date     DATE    NOT NULL,
    period_type_concept_id          BIGINT  NOT NULL,
    -- tracking cols
    unit_id         TEXT,
    load_table_id   TEXT,
    load_row_id     BIGINT,
    trace_id        TEXT
);

INSERT INTO staging.cdm_observation_period
SELECT
    staging.omop_id()           AS observation_period_id,
    src.person_id               AS person_id,
    MIN(src.start_date)         AS observation_period_start_date,
    MAX(COALESCE(src.end_date, src.start_date)) AS observation_period_end_date,
    32828                       AS period_type_concept_id,  -- EHR episode record
    'observation_period'        AS unit_id,
    'event tables'              AS load_table_id,
    0                           AS load_row_id,
    NULL::TEXT                  AS trace_id
FROM
    staging.tmp_observation_period_clean src
WHERE
    src.start_date IS NOT NULL
GROUP BY
    src.person_id
;

DROP TABLE IF EXISTS staging.tmp_observation_period_clean;

CREATE INDEX IF NOT EXISTS idx_cdm_op_person ON staging.cdm_observation_period (person_id);
