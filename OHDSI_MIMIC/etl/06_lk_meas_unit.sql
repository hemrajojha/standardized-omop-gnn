-- =======================================================================
-- 06_lk_meas_unit.sql
-- Build concept mapping for measurement units and operators.
--
-- Adapted from: https://github.com/OHDSI/MIMIC/blob/main/etl/etl/lk_meas_unit.sql
-- =======================================================================

-- -----------------------------------------------------------------------
-- lk_meas_unit_concept – map valueuom strings to UCUM unit concepts
-- -----------------------------------------------------------------------
DROP TABLE IF EXISTS staging.lk_meas_unit_concept;
CREATE TABLE staging.lk_meas_unit_concept AS
SELECT DISTINCT
    src.valueuom                AS source_code,
    vc.concept_id               AS unit_concept_id
FROM (
    SELECT DISTINCT valueuom FROM staging.src_labevents   WHERE valueuom IS NOT NULL AND TRIM(valueuom) <> ''
    UNION
    SELECT DISTINCT valueuom FROM staging.src_chartevents WHERE valueuom IS NOT NULL AND TRIM(valueuom) <> ''
) src
LEFT JOIN
    vocab.concept vc
        ON  vc.concept_code = src.valueuom
        AND vc.vocabulary_id = 'UCUM'
        AND vc.standard_concept = 'S'
        AND vc.invalid_reason IS NULL
;

-- -----------------------------------------------------------------------
-- lk_meas_operator_concept – map operator strings to OMOP operator concepts
-- -----------------------------------------------------------------------
DROP TABLE IF EXISTS staging.lk_meas_operator_concept;
CREATE TABLE staging.lk_meas_operator_concept AS
SELECT
    op.operator_source  AS source_code,
    vc.concept_id       AS operator_concept_id
FROM (
    VALUES
        ('=',  '='),
        ('<',  '<'),
        ('>',  '>'),
        ('<=', '<='),
        ('>=', '>=')
) AS op (operator_source, operator_match)
LEFT JOIN
    vocab.concept vc
        ON  vc.concept_code = op.operator_match
        AND vc.domain_id = 'Meas Value Operator'
        AND vc.standard_concept = 'S'
        AND vc.invalid_reason IS NULL
;
