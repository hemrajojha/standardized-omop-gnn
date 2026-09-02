-- =======================================================================
-- 08_cdm_visit_occurrence.sql
-- Populate staging.cdm_visit_occurrence from lk_visit_clean (admissions).
--
-- Adapted from: https://github.com/OHDSI/MIMIC/blob/main/etl/etl/cdm_visit_occurrence.sql
-- =======================================================================

DROP TABLE IF EXISTS staging.cdm_visit_occurrence;
CREATE TABLE staging.cdm_visit_occurrence (
    visit_occurrence_id             BIGINT      NOT NULL,
    person_id                       BIGINT      NOT NULL,
    visit_concept_id                BIGINT      NOT NULL,
    visit_start_date                DATE        NOT NULL,
    visit_start_datetime            TIMESTAMP,
    visit_end_date                  DATE        NOT NULL,
    visit_end_datetime              TIMESTAMP,
    visit_type_concept_id           BIGINT      NOT NULL,
    provider_id                     BIGINT,
    care_site_id                    BIGINT,
    visit_source_value              TEXT,
    visit_source_concept_id         BIGINT,
    admitted_from_concept_id        BIGINT,
    admitted_from_source_value      TEXT,
    discharge_to_concept_id         BIGINT,
    discharge_to_source_value       TEXT,
    preceding_visit_occurrence_id   BIGINT,
    -- tracking cols
    unit_id         TEXT,
    load_table_id   TEXT,
    load_row_id     BIGINT,
    trace_id        TEXT
);

INSERT INTO staging.cdm_visit_occurrence
SELECT
    src.visit_occurrence_id                         AS visit_occurrence_id,
    per.person_id                                   AS person_id,
    COALESCE(lat.target_concept_id, 9201)           AS visit_concept_id,   -- default: Inpatient Visit
    CAST(src.start_datetime AS DATE)                AS visit_start_date,
    src.start_datetime                              AS visit_start_datetime,
    CAST(src.end_datetime AS DATE)                  AS visit_end_date,
    src.end_datetime                                AS visit_end_datetime,
    32817                                           AS visit_type_concept_id,  -- EHR
    NULL::BIGINT                                    AS provider_id,
    cs.care_site_id                                 AS care_site_id,
    src.source_value                                AS visit_source_value,
    COALESCE(lat.source_concept_id, 0)              AS visit_source_concept_id,
    CASE WHEN src.admission_location IS NOT NULL
        THEN COALESCE(la.target_concept_id, 0)
        ELSE NULL
    END                                             AS admitted_from_concept_id,
    src.admission_location                          AS admitted_from_source_value,
    CASE WHEN src.discharge_location IS NOT NULL
        THEN COALESCE(ld.target_concept_id, 0)
        ELSE NULL
    END                                             AS discharge_to_concept_id,
    src.discharge_location                          AS discharge_to_source_value,
    LAG(src.visit_occurrence_id) OVER (
        PARTITION BY src.subject_id
        ORDER BY src.start_datetime
    )                                               AS preceding_visit_occurrence_id,
    'visit.' || src.unit_id                         AS unit_id,
    src.load_table_id                               AS load_table_id,
    src.load_row_id                                 AS load_row_id,
    src.trace_id                                    AS trace_id
FROM
    staging.lk_visit_clean src
INNER JOIN
    staging.cdm_person per
        ON src.subject_id::TEXT = per.person_source_value
LEFT JOIN
    staging.lk_visit_concept lat
        ON lat.source_code = src.admission_type
LEFT JOIN
    staging.lk_visit_concept la
        ON la.source_code = src.admission_location
LEFT JOIN
    staging.lk_visit_concept ld
        ON ld.source_code = src.discharge_location
LEFT JOIN
    staging.cdm_care_site cs
        ON cs.care_site_name = 'BIDMC'
;

CREATE INDEX IF NOT EXISTS idx_cdm_vo_person ON staging.cdm_visit_occurrence (person_id);
CREATE INDEX IF NOT EXISTS idx_cdm_vo_src    ON staging.cdm_visit_occurrence (visit_source_value);
CREATE INDEX IF NOT EXISTS idx_cdm_vo_id     ON staging.cdm_visit_occurrence (visit_occurrence_id);
