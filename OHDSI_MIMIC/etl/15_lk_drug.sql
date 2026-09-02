-- =======================================================================
-- 15_lk_drug.sql
-- Build lookup tables for drug_exposure from prescriptions.
-- Maps NDC codes and drug names to OMOP standard drug concepts.
--
-- Adapted from: https://github.com/OHDSI/MIMIC/blob/main/etl/etl/lk_drug.sql
-- =======================================================================

-- -----------------------------------------------------------------------
-- lk_prescriptions_clean
-- -----------------------------------------------------------------------
DROP TABLE IF EXISTS staging.lk_prescriptions_clean;
CREATE TABLE staging.lk_prescriptions_clean AS
SELECT
    src.subject_id                              AS subject_id,
    src.hadm_id                                 AS hadm_id,
    src.starttime                               AS start_datetime,
    COALESCE(src.stoptime, src.starttime)       AS end_datetime,
    src.route                                   AS route_source_code,
    src.form_unit_disp                          AS dose_unit_source_code,
    src.ndc::TEXT                               AS ndc_source_code,
    TRIM(COALESCE(src.drug,'') || ' ' || COALESCE(src.prod_strength,'')) AS gcpt_source_code,
    -- try to extract a numeric quantity from form_val_disp
    CASE
        WHEN src.form_val_disp ~ '^[-]?[0-9]+[.]?[0-9]*'
        THEN substring(src.form_val_disp FROM '[-]?[0-9]+[.]?[0-9]*')::FLOAT
        ELSE NULL
    END                                         AS quantity,
    src.pharmacy_id                             AS pharmacy_id,
    'prescriptions'                             AS unit_id,
    src.load_table_id                           AS load_table_id,
    src.load_row_id                             AS load_row_id,
    src.trace_id                                AS trace_id
FROM
    staging.src_prescriptions src
WHERE
    src.starttime IS NOT NULL
    AND src.drug IS NOT NULL
;

-- -----------------------------------------------------------------------
-- lk_pr_route_concept – map route to OMOP route concept
-- -----------------------------------------------------------------------
DROP TABLE IF EXISTS staging.lk_pr_route_concept;
CREATE TABLE staging.lk_pr_route_concept AS
SELECT DISTINCT
    src.route_source_code               AS source_code,
    COALESCE(vc2.concept_id, 0)         AS route_concept_id
FROM
    staging.lk_prescriptions_clean src
LEFT JOIN
    vocab.concept vc
        ON UPPER(vc.concept_code) = UPPER(src.route_source_code)
        AND vc.vocabulary_id = 'SNOMED'
        AND vc.domain_id = 'Route'
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

-- -----------------------------------------------------------------------
-- lk_drug_mapped – try NDC first, then drug name
-- -----------------------------------------------------------------------
DROP TABLE IF EXISTS staging.lk_drug_mapped;
CREATE TABLE staging.lk_drug_mapped AS
SELECT
    src.subject_id                              AS subject_id,
    src.hadm_id                                 AS hadm_id,
    src.start_datetime                          AS start_datetime,
    src.end_datetime                            AS end_datetime,
    32838                                       AS type_concept_id, -- EHR dispensing record
    src.quantity                                AS quantity,
    COALESCE(rt.route_concept_id, 0)            AS route_concept_id,
    src.route_source_code                       AS route_source_value,
    src.dose_unit_source_code                   AS dose_unit_source_value,
    -- prefer NDC mapping, fall back to drug name (concept_code match)
    COALESCE(
        NULLIF(vc_ndc2.concept_id, 0),
        NULLIF(vc_name2.concept_id, 0),
        0)                                      AS target_concept_id,
    COALESCE(
        NULLIF(vc_ndc.concept_id, 0),
        NULLIF(vc_name.concept_id, 0),
        0)                                      AS source_concept_id,
    COALESCE(src.ndc_source_code, src.gcpt_source_code) AS source_code,
    'drug'                                      AS unit_id,
    src.load_table_id                           AS load_table_id,
    src.load_row_id                             AS load_row_id,
    src.trace_id                                AS trace_id
FROM
    staging.lk_prescriptions_clean src
LEFT JOIN
    staging.lk_pr_route_concept rt
        ON rt.source_code = src.route_source_code
-- NDC lookup
LEFT JOIN
    vocab.concept vc_ndc
        ON vc_ndc.concept_code = src.ndc_source_code
        AND vc_ndc.vocabulary_id = 'NDC'
        AND vc_ndc.invalid_reason IS NULL
LEFT JOIN
    vocab.concept_relationship vcr_ndc
        ON  vc_ndc.concept_id = vcr_ndc.concept_id_1
        AND vcr_ndc.relationship_id = 'Maps to'
LEFT JOIN
    vocab.concept vc_ndc2
        ON vc_ndc2.concept_id = vcr_ndc.concept_id_2
        AND vc_ndc2.standard_concept = 'S'
        AND vc_ndc2.invalid_reason IS NULL
-- Drug name lookup (ingredient level)
LEFT JOIN
    vocab.concept vc_name
        ON UPPER(vc_name.concept_name) = UPPER(src.gcpt_source_code)
        AND vc_name.domain_id = 'Drug'
        AND vc_name.concept_class_id = 'Ingredient'
        AND vc_name.standard_concept = 'S'
        AND vc_name.invalid_reason IS NULL
LEFT JOIN
    vocab.concept vc_name2
        ON vc_name2.concept_id = vc_name.concept_id  -- already standard
;
