-- =======================================================================
-- 07_lk_meas_labevents.sql
-- Build lookup / mapped intermediate tables for lab measurements.
-- Source: src_labevents + src_d_labitems
--
-- Adapted from: https://github.com/OHDSI/MIMIC/blob/main/etl/etl/lk_meas_labevents.sql
-- =======================================================================

-- -----------------------------------------------------------------------
-- lk_meas_d_labitems_clean – source code and vocabulary per itemid
-- -----------------------------------------------------------------------
DROP TABLE IF EXISTS staging.lk_meas_d_labitems_clean;
CREATE TABLE staging.lk_meas_d_labitems_clean AS
SELECT
    dlab.itemid                                                     AS itemid,
    COALESCE(dlab.loinc_code, dlab.itemid::TEXT)                    AS source_code,
    dlab.loinc_code                                                 AS loinc_code,
    dlab.label || '|' || COALESCE(dlab.fluid,'') || '|' || COALESCE(dlab.category,'') AS source_label,
    CASE WHEN dlab.loinc_code IS NOT NULL THEN 'LOINC' ELSE 'mimiciv_meas_lab_loinc' END
                                                                    AS source_vocabulary_id
FROM
    staging.src_d_labitems dlab
;

-- -----------------------------------------------------------------------
-- lk_meas_labevents_clean – one row per lab result with parsed value
-- -----------------------------------------------------------------------
DROP TABLE IF EXISTS staging.lk_meas_labevents_clean;
CREATE TABLE staging.lk_meas_labevents_clean AS
SELECT
    staging.omop_id()                                               AS measurement_id,
    src.subject_id                                                  AS subject_id,
    src.charttime                                                   AS start_datetime,
    src.hadm_id                                                     AS hadm_id,
    src.itemid                                                      AS itemid,
    src.value                                                       AS value,
    substring(src.value FROM '^(<=|>=|>|<|=|)')                     AS value_operator,
    substring(src.value FROM '[-]?[0-9]+[.]?[0-9]*')               AS value_number,
    CASE WHEN TRIM(COALESCE(src.valueuom,'')) <> '' THEN src.valueuom ELSE NULL END AS valueuom,
    src.ref_range_lower                                             AS ref_range_lower,
    src.ref_range_upper                                             AS ref_range_upper,
    'labevents'                                                     AS unit_id,
    src.load_table_id                                               AS load_table_id,
    src.load_row_id                                                 AS load_row_id,
    src.trace_id                                                    AS trace_id
FROM
    staging.src_labevents src
WHERE
    src.itemid IS NOT NULL
;

-- -----------------------------------------------------------------------
-- lk_meas_labevents_mapped – join to vocabulary to get standard concept_id
-- -----------------------------------------------------------------------
DROP TABLE IF EXISTS staging.lk_meas_labevents_mapped;
CREATE TABLE staging.lk_meas_labevents_mapped AS
SELECT
    src.measurement_id                          AS measurement_id,
    src.subject_id                              AS subject_id,
    src.hadm_id                                 AS hadm_id,
    src.start_datetime                          AS start_datetime,
    src.value                                   AS value_source_value,
    src.value_number                            AS value_as_number,
    src.valueuom                                AS unit_source_value,
    src.ref_range_lower                         AS range_low,
    src.ref_range_upper                         AS range_high,
    -- operator
    COALESCE(op.operator_concept_id, 0)         AS operator_concept_id,
    -- unit
    COALESCE(uc.unit_concept_id, 0)             AS unit_concept_id,
    -- source concept
    dlab.source_code                            AS source_code,
    COALESCE(vc.concept_id, 0)                  AS source_concept_id,
    -- standard concept
    COALESCE(vc2.concept_id, 0)                 AS target_concept_id,
    COALESCE(vc2.domain_id, 'Measurement')      AS target_domain_id,
    'labevents'                                 AS unit_id,
    src.load_table_id                           AS load_table_id,
    src.load_row_id                             AS load_row_id,
    src.trace_id                                AS trace_id
FROM
    staging.lk_meas_labevents_clean src
INNER JOIN
    staging.lk_meas_d_labitems_clean dlab
        ON src.itemid = dlab.itemid
LEFT JOIN
    staging.lk_meas_operator_concept op
        ON op.source_code = src.value_operator
LEFT JOIN
    staging.lk_meas_unit_concept uc
        ON uc.source_code = src.valueuom
LEFT JOIN
    vocab.concept vc
        ON REPLACE(vc.concept_code, '.', '') = REPLACE(TRIM(dlab.source_code), '.', '')
        AND vc.vocabulary_id = dlab.source_vocabulary_id
LEFT JOIN
    vocab.concept_relationship vcr
        ON  vc.concept_id = vcr.concept_id_1
        AND vcr.relationship_id = 'Maps to'
LEFT JOIN
    vocab.concept vc2
        ON  vc2.concept_id = vcr.concept_id_2
        AND vc2.standard_concept = 'S'
        AND vc2.invalid_reason IS NULL
;

CREATE INDEX IF NOT EXISTS idx_lk_lab_subj ON staging.lk_meas_labevents_mapped (subject_id);
CREATE INDEX IF NOT EXISTS idx_lk_lab_hadm ON staging.lk_meas_labevents_mapped (hadm_id);
