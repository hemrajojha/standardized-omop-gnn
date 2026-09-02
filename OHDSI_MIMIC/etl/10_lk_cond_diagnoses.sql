-- =======================================================================
-- 10_lk_cond_diagnoses.sql
-- Build intermediate lookup tables for condition_occurrence.
-- Source: src_diagnoses_icd mapped to OMOP standard concepts via ICD9CM/ICD10CM.
--
-- Adapted from: https://github.com/OHDSI/MIMIC/blob/main/etl/etl/lk_cond_diagnoses.sql
-- =======================================================================

-- -----------------------------------------------------------------------
-- lk_diagnoses_icd_clean
-- -----------------------------------------------------------------------
DROP TABLE IF EXISTS staging.lk_diagnoses_icd_clean;
CREATE TABLE staging.lk_diagnoses_icd_clean AS
SELECT
    src.subject_id                              AS subject_id,
    src.hadm_id                                 AS hadm_id,
    CASE WHEN src.seq_num > 20 THEN 20 ELSE src.seq_num END AS seq_num,
    COALESCE(adm.edregtime, adm.admittime)      AS start_datetime,
    adm.dischtime                               AS end_datetime,
    src.icd_code                                AS source_code,
    CASE
        WHEN src.icd_version = 9  THEN 'ICD9CM'
        WHEN src.icd_version = 10 THEN 'ICD10CM'
        ELSE NULL
    END                                         AS source_vocabulary_id,
    'diagnoses_icd'         AS unit_id,
    src.load_table_id       AS load_table_id,
    src.load_row_id         AS load_row_id,
    src.trace_id            AS trace_id
FROM
    staging.src_diagnoses_icd src
INNER JOIN
    staging.src_admissions adm
        ON  src.hadm_id = adm.hadm_id
;

-- -----------------------------------------------------------------------
-- lk_diagnoses_icd_mapped
-- -----------------------------------------------------------------------
DROP TABLE IF EXISTS staging.lk_diagnoses_icd_mapped;
CREATE TABLE staging.lk_diagnoses_icd_mapped AS
SELECT
    src.subject_id                          AS subject_id,
    src.hadm_id                             AS hadm_id,
    src.seq_num                             AS seq_num,
    src.start_datetime                      AS start_datetime,
    src.end_datetime                        AS end_datetime,
    32821                                   AS type_concept_id,  -- EHR billing record
    src.source_code                         AS source_code,
    src.source_vocabulary_id                AS source_vocabulary_id,
    COALESCE(vc.concept_id, 0)              AS source_concept_id,
    vc.domain_id                            AS source_domain_id,
    COALESCE(vc2.concept_id, 0)             AS target_concept_id,
    COALESCE(vc2.domain_id, 'Condition')    AS target_domain_id,
    'cond.' || src.unit_id                  AS unit_id,
    src.load_table_id                       AS load_table_id,
    src.load_row_id                         AS load_row_id,
    src.trace_id                            AS trace_id
FROM
    staging.lk_diagnoses_icd_clean src
LEFT JOIN
    vocab.concept vc
        ON REPLACE(vc.concept_code, '.', '') = REPLACE(TRIM(src.source_code), '.', '')
        AND vc.vocabulary_id = src.source_vocabulary_id
LEFT JOIN
    vocab.concept_relationship vcr
        ON  vc.concept_id = vcr.concept_id_1
        AND vcr.relationship_id = 'Maps to'
LEFT JOIN
    vocab.concept vc2
        ON vc2.concept_id = vcr.concept_id_2
        AND vc2.standard_concept = 'S'
        AND vc2.invalid_reason IS NULL
;

CREATE INDEX IF NOT EXISTS idx_lk_diag_subj ON staging.lk_diagnoses_icd_mapped (subject_id);
CREATE INDEX IF NOT EXISTS idx_lk_diag_hadm ON staging.lk_diagnoses_icd_mapped (hadm_id);
