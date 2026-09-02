-- =======================================================================
-- 21_cdm_cdm_source.sql
-- Populate staging.cdm_cdm_source with MIMIC-IV metadata.
--
-- Adapted from: https://github.com/OHDSI/MIMIC/blob/main/etl/etl/cdm_cdm_source.sql
-- =======================================================================

DROP TABLE IF EXISTS staging.cdm_cdm_source;
CREATE TABLE staging.cdm_cdm_source (
    cdm_source_name                 TEXT    NOT NULL,
    cdm_source_abbreviation         TEXT,
    cdm_holder                      TEXT,
    source_description              TEXT,
    source_documentation_reference  TEXT,
    cdm_etl_reference               TEXT,
    source_release_date             DATE,
    cdm_release_date                DATE,
    cdm_version                     TEXT,
    vocabulary_version              TEXT,
    -- tracking cols
    unit_id     TEXT,
    load_table_id TEXT,
    load_row_id BIGINT,
    trace_id    TEXT
);

INSERT INTO staging.cdm_cdm_source
SELECT
    'MIMIC IV'                                      AS cdm_source_name,
    'mimiciv'                                       AS cdm_source_abbreviation,
    'PhysioNet / MIT Laboratory for Computational Physiology' AS cdm_holder,
    'MIMIC-IV is a publicly available database of patients admitted to the '
    'Beth Israel Deaconess Medical Center in Boston, MA, USA.'
                                                    AS source_description,
    'https://mimic-iv.mit.edu/docs/'                AS source_documentation_reference,
    'https://github.com/OHDSI/MIMIC/'               AS cdm_etl_reference,
    '2020-09-01'::DATE                              AS source_release_date,
    CURRENT_DATE                                    AS cdm_release_date,
    '5.4.1'                                         AS cdm_version,
    v.vocabulary_version                            AS vocabulary_version,
    'cdm.source'    AS unit_id,
    'none'          AS load_table_id,
    1               AS load_row_id,
    NULL::TEXT      AS trace_id
FROM
    vocab.vocabulary v
WHERE
    v.vocabulary_id = 'None'
LIMIT 1
;

-- Fallback if vocabulary 'None' is absent
INSERT INTO staging.cdm_cdm_source
SELECT
    'MIMIC IV', 'mimiciv', 'PhysioNet',
    'MIMIC-IV OMOP CDM conversion',
    'https://mimic-iv.mit.edu/docs/',
    'https://github.com/OHDSI/MIMIC/',
    '2020-09-01'::DATE, CURRENT_DATE, '5.4.1', 'unknown',
    'cdm.source', 'none', 1, NULL
WHERE NOT EXISTS (SELECT 1 FROM staging.cdm_cdm_source)
;
