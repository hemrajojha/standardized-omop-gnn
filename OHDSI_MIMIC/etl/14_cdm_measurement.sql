-- =======================================================================
-- 14_cdm_measurement.sql
-- Populate staging.cdm_measurement from lk_meas_labevents_mapped.
--
-- Adapted from: https://github.com/OHDSI/MIMIC/blob/main/etl/etl/cdm_measurement.sql
-- =======================================================================

DROP TABLE IF EXISTS staging.cdm_measurement;
CREATE TABLE staging.cdm_measurement (
    measurement_id                  BIGINT  NOT NULL,
    person_id                       BIGINT  NOT NULL,
    measurement_concept_id          BIGINT  NOT NULL,
    measurement_date                DATE    NOT NULL,
    measurement_datetime            TIMESTAMP,
    measurement_time                TEXT,
    measurement_type_concept_id     BIGINT  NOT NULL,
    operator_concept_id             BIGINT,
    value_as_number                 FLOAT,
    value_as_concept_id             BIGINT,
    unit_concept_id                 BIGINT,
    range_low                       FLOAT,
    range_high                      FLOAT,
    provider_id                     BIGINT,
    visit_occurrence_id             BIGINT,
    visit_detail_id                 BIGINT,
    measurement_source_value        TEXT,
    measurement_source_concept_id   BIGINT,
    unit_source_value               TEXT,
    value_source_value              TEXT,
    -- tracking cols
    unit_id         TEXT,
    load_table_id   TEXT,
    load_row_id     BIGINT,
    trace_id        TEXT
);

-- Rule 1: Labs from labevents
INSERT INTO staging.cdm_measurement
SELECT
    src.measurement_id                          AS measurement_id,
    per.person_id                               AS person_id,
    COALESCE(src.target_concept_id, 0)          AS measurement_concept_id,
    CAST(src.start_datetime AS DATE)            AS measurement_date,
    src.start_datetime                          AS measurement_datetime,
    NULL::TEXT                                  AS measurement_time,
    32856                                       AS measurement_type_concept_id,  -- Lab
    src.operator_concept_id                     AS operator_concept_id,
    CAST(src.value_as_number AS FLOAT)          AS value_as_number,
    NULL::BIGINT                                AS value_as_concept_id,
    src.unit_concept_id                         AS unit_concept_id,
    src.range_low                               AS range_low,
    src.range_high                              AS range_high,
    NULL::BIGINT                                AS provider_id,
    vis.visit_occurrence_id                     AS visit_occurrence_id,
    NULL::BIGINT                                AS visit_detail_id,
    src.source_code                             AS measurement_source_value,
    COALESCE(src.source_concept_id, 0)          AS measurement_source_concept_id,
    src.unit_source_value                       AS unit_source_value,
    src.value_source_value                      AS value_source_value,
    'measurement.' || src.unit_id               AS unit_id,
    src.load_table_id                           AS load_table_id,
    src.load_row_id                             AS load_row_id,
    src.trace_id                                AS trace_id
FROM
    staging.lk_meas_labevents_mapped src
INNER JOIN
    staging.cdm_person per
        ON src.subject_id::TEXT = per.person_source_value
LEFT JOIN
    staging.cdm_visit_occurrence vis
        ON vis.visit_source_value =
            src.subject_id::TEXT || '|' || src.hadm_id::TEXT
WHERE
    src.target_domain_id = 'Measurement'
   OR src.target_domain_id IS NULL
;

CREATE INDEX IF NOT EXISTS idx_cdm_meas_person ON staging.cdm_measurement (person_id);
CREATE INDEX IF NOT EXISTS idx_cdm_meas_visit  ON staging.cdm_measurement (visit_occurrence_id);
