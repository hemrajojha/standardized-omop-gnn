-- =======================================================================
-- 12_cdm_condition_occurrence.sql
-- Populate staging.cdm_condition_occurrence from lk_diagnoses_icd_mapped.
--
-- Adapted from: https://github.com/OHDSI/MIMIC/blob/main/etl/etl/cdm_condition_occurrence.sql
-- =======================================================================

DROP TABLE IF EXISTS staging.cdm_condition_occurrence;
CREATE TABLE staging.cdm_condition_occurrence (
    condition_occurrence_id         BIGINT  NOT NULL,
    person_id                       BIGINT  NOT NULL,
    condition_concept_id            BIGINT  NOT NULL,
    condition_start_date            DATE    NOT NULL,
    condition_start_datetime        TIMESTAMP,
    condition_end_date              DATE,
    condition_end_datetime          TIMESTAMP,
    condition_type_concept_id       BIGINT  NOT NULL,
    condition_status_concept_id     BIGINT,
    stop_reason                     TEXT,
    provider_id                     BIGINT,
    visit_occurrence_id             BIGINT,
    visit_detail_id                 BIGINT,
    condition_source_value          TEXT,
    condition_source_concept_id     BIGINT,
    condition_status_source_value   TEXT,
    -- tracking cols
    unit_id         TEXT,
    load_table_id   TEXT,
    load_row_id     BIGINT,
    trace_id        TEXT
);

-- Rule 1: ICD diagnoses
INSERT INTO staging.cdm_condition_occurrence
SELECT
    staging.omop_id()                           AS condition_occurrence_id,
    per.person_id                               AS person_id,
    COALESCE(src.target_concept_id, 0)          AS condition_concept_id,
    CAST(src.start_datetime AS DATE)            AS condition_start_date,
    src.start_datetime                          AS condition_start_datetime,
    CAST(src.end_datetime AS DATE)              AS condition_end_date,
    src.end_datetime                            AS condition_end_datetime,
    src.type_concept_id                         AS condition_type_concept_id,
    NULL::BIGINT                                AS condition_status_concept_id,
    NULL::TEXT                                  AS stop_reason,
    NULL::BIGINT                                AS provider_id,
    vis.visit_occurrence_id                     AS visit_occurrence_id,
    NULL::BIGINT                                AS visit_detail_id,
    src.source_code                             AS condition_source_value,
    COALESCE(src.source_concept_id, 0)          AS condition_source_concept_id,
    NULL::TEXT                                  AS condition_status_source_value,
    'condition.' || src.unit_id                 AS unit_id,
    src.load_table_id                           AS load_table_id,
    src.load_row_id                             AS load_row_id,
    src.trace_id                                AS trace_id
FROM
    staging.lk_diagnoses_icd_mapped src
INNER JOIN
    staging.cdm_person per
        ON src.subject_id::TEXT = per.person_source_value
INNER JOIN
    staging.cdm_visit_occurrence vis
        ON vis.visit_source_value =
            src.subject_id::TEXT || '|' || src.hadm_id::TEXT
WHERE
    src.target_domain_id = 'Condition'
;

CREATE INDEX IF NOT EXISTS idx_cdm_co_person ON staging.cdm_condition_occurrence (person_id);
CREATE INDEX IF NOT EXISTS idx_cdm_co_visit  ON staging.cdm_condition_occurrence (visit_occurrence_id);
