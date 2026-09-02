-- =======================================================================
-- 11_lk_procedure.sql
-- Build lookup tables for procedure_occurrence.
-- Source: src_procedures_icd (ICD9Proc / ICD10PCS).
--
-- Adapted from: https://github.com/OHDSI/MIMIC/blob/main/etl/etl/lk_procedure.sql
-- =======================================================================

-- -----------------------------------------------------------------------
-- lk_procedures_icd_clean
-- -----------------------------------------------------------------------
DROP TABLE IF EXISTS staging.lk_procedures_icd_clean;
CREATE TABLE staging.lk_procedures_icd_clean AS
SELECT
    src.subject_id                              AS subject_id,
    src.hadm_id                                 AS hadm_id,
    adm.dischtime                               AS start_datetime,
    src.icd_code                                AS icd_code,
    src.icd_version                             AS icd_version,
    CASE
        WHEN src.icd_version = 9  THEN 'ICD9Proc'
        WHEN src.icd_version = 10 THEN 'ICD10PCS'
        ELSE 'Unknown'
    END                                         AS source_vocabulary_id,
    REPLACE(src.icd_code, '.', '')              AS source_code,
    src.load_table_id                           AS load_table_id,
    src.load_row_id                             AS load_row_id,
    src.trace_id                                AS trace_id
FROM
    staging.src_procedures_icd src
INNER JOIN
    staging.src_admissions adm
        ON src.hadm_id = adm.hadm_id
;

-- -----------------------------------------------------------------------
-- lk_procedure_mapped
-- -----------------------------------------------------------------------
DROP TABLE IF EXISTS staging.lk_procedure_mapped;
CREATE TABLE staging.lk_procedure_mapped AS
SELECT
    src.subject_id                          AS subject_id,
    src.hadm_id                             AS hadm_id,
    src.start_datetime                      AS start_datetime,
    32821                                   AS type_concept_id,  -- EHR billing record
    src.source_code                         AS source_code,
    src.source_vocabulary_id                AS source_vocabulary_id,
    COALESCE(vc.concept_id, 0)              AS source_concept_id,
    COALESCE(vc2.concept_id, 0)             AS target_concept_id,
    COALESCE(vc2.domain_id, 'Procedure')    AS target_domain_id,
    NULL::DOUBLE PRECISION                  AS quantity,
    'procedures_icd'                        AS unit_id,
    src.load_table_id                       AS load_table_id,
    src.load_row_id                         AS load_row_id,
    src.trace_id                            AS trace_id
FROM
    staging.lk_procedures_icd_clean src
LEFT JOIN
    vocab.concept vc
        ON REPLACE(vc.concept_code, '.', '') = REPLACE(TRIM(src.source_code), '.', '')
        AND vc.vocabulary_id = src.source_vocabulary_id
LEFT JOIN
    vocab.concept_relationship vcr
        ON  vc.concept_id = vcr.concept_id_1
        AND vcr.relationship_id IN ('Maps to', 'CPT4 - SNOMED eq')
LEFT JOIN
    vocab.concept vc2
        ON vc2.concept_id = vcr.concept_id_2
        AND vc2.standard_concept = 'S'
        AND vc2.invalid_reason IS NULL
;
