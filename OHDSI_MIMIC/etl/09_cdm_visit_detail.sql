-- =======================================================================
-- 09_cdm_visit_detail.sql
-- Populate staging.cdm_visit_detail from icustays.
-- Each ICU stay becomes a visit_detail record linked to its parent
-- visit_occurrence (from admissions).
--
-- Adapted from: https://github.com/OHDSI/MIMIC/blob/main/etl/etl/cdm_visit_detail.sql
-- =======================================================================

DROP TABLE IF EXISTS staging.cdm_visit_detail;
CREATE TABLE staging.cdm_visit_detail (
    visit_detail_id                 BIGINT  NOT NULL,
    person_id                       BIGINT  NOT NULL,
    visit_detail_concept_id         BIGINT  NOT NULL,
    visit_detail_start_date         DATE    NOT NULL,
    visit_detail_start_datetime     TIMESTAMP,
    visit_detail_end_date           DATE    NOT NULL,
    visit_detail_end_datetime       TIMESTAMP,
    visit_detail_type_concept_id    BIGINT  NOT NULL,
    provider_id                     BIGINT,
    care_site_id                    BIGINT,
    admitted_from_concept_id        BIGINT,
    discharge_to_concept_id         BIGINT,
    preceding_visit_detail_id       BIGINT,
    visit_detail_source_value       TEXT,
    visit_detail_source_concept_id  BIGINT,
    admitted_from_source_value      TEXT,
    discharge_to_source_value       TEXT,
    visit_detail_parent_id          BIGINT,
    visit_occurrence_id             BIGINT  NOT NULL,
    -- tracking cols
    unit_id         TEXT,
    load_table_id   TEXT,
    load_row_id     BIGINT,
    trace_id        TEXT
);

INSERT INTO staging.cdm_visit_detail
SELECT
    staging.omop_id()                               AS visit_detail_id,
    per.person_id                                   AS person_id,
    32037                                           AS visit_detail_concept_id, -- Intensive Care
    CAST(icu.intime AS DATE)                        AS visit_detail_start_date,
    icu.intime                                      AS visit_detail_start_datetime,
    CAST(COALESCE(icu.outtime, icu.intime) AS DATE) AS visit_detail_end_date,
    COALESCE(icu.outtime, icu.intime)               AS visit_detail_end_datetime,
    32817                                           AS visit_detail_type_concept_id,  -- EHR
    NULL::BIGINT                                    AS provider_id,
    cs.care_site_id                                 AS care_site_id,
    NULL::BIGINT                                    AS admitted_from_concept_id,
    NULL::BIGINT                                    AS discharge_to_concept_id,
    NULL::BIGINT                                    AS preceding_visit_detail_id,
    icu.subject_id::TEXT || '|' || icu.stay_id::TEXT AS visit_detail_source_value,
    0                                               AS visit_detail_source_concept_id,
    NULL::TEXT                                      AS admitted_from_source_value,
    NULL::TEXT                                      AS discharge_to_source_value,
    NULL::BIGINT                                    AS visit_detail_parent_id,
    vo.visit_occurrence_id                          AS visit_occurrence_id,
    'visit_detail.icustays'                         AS unit_id,
    'icustays'                                      AS load_table_id,
    icu.stay_id::BIGINT                             AS load_row_id,
    'icustays|' || icu.subject_id::TEXT || '|' || icu.stay_id::TEXT AS trace_id
FROM
    mimiciv_icu.icustays icu
INNER JOIN
    staging.cdm_person per
        ON icu.subject_id::TEXT = per.person_source_value
INNER JOIN
    staging.cdm_visit_occurrence vo
        ON vo.visit_source_value =
            icu.subject_id::TEXT || '|' || icu.hadm_id::TEXT
LEFT JOIN
    staging.cdm_care_site cs
        ON cs.care_site_name = icu.first_careunit
;

CREATE INDEX IF NOT EXISTS idx_cdm_vd_vo ON staging.cdm_visit_detail (visit_occurrence_id);
CREATE INDEX IF NOT EXISTS idx_cdm_vd_person ON staging.cdm_visit_detail (person_id);
