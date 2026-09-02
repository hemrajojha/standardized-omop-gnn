-- =======================================================================
-- 02_cdm_care_site.sql
-- Populate staging.cdm_care_site from distinct care units in transfers.
--
-- Adapted from: https://github.com/OHDSI/MIMIC/blob/main/etl/etl/cdm_care_site.sql
-- =======================================================================

-- -----------------------------------------------------------------------
-- lk_trans_careunit_clean – distinct care units from transfers
-- -----------------------------------------------------------------------
DROP TABLE IF EXISTS staging.lk_trans_careunit_clean;
CREATE TABLE staging.lk_trans_careunit_clean AS
SELECT
    src.careunit                        AS source_code,
    src.load_table_id                   AS load_table_id,
    0                                   AS load_row_id,
    MIN(src.trace_id)                   AS trace_id
FROM
    staging.src_transfers src
WHERE
    src.careunit IS NOT NULL
GROUP BY
    careunit,
    load_table_id
;

-- -----------------------------------------------------------------------
-- cdm_care_site
-- -----------------------------------------------------------------------
DROP TABLE IF EXISTS staging.cdm_care_site;
CREATE TABLE staging.cdm_care_site (
    care_site_id                  BIGINT,
    care_site_name                TEXT,
    place_of_service_concept_id   BIGINT,
    location_id                   BIGINT,
    care_site_source_value        TEXT,
    place_of_service_source_value TEXT,
    -- tracking cols
    unit_id         TEXT,
    load_table_id   TEXT,
    load_row_id     BIGINT,
    trace_id        TEXT
);

INSERT INTO staging.cdm_care_site
SELECT
    staging.omop_id()           AS care_site_id,
    src.source_code             AS care_site_name,
    COALESCE(vc2.concept_id, 0) AS place_of_service_concept_id,
    1                           AS location_id,
    src.source_code             AS care_site_source_value,
    src.source_code             AS place_of_service_source_value,
    'care_site.transfers'       AS unit_id,
    src.load_table_id           AS load_table_id,
    src.load_row_id             AS load_row_id,
    src.trace_id                AS trace_id
FROM
    staging.lk_trans_careunit_clean src
LEFT JOIN
    vocab.concept vc
        ON  vc.concept_code = src.source_code
        AND vc.vocabulary_id = 'mimiciv_cs_place_of_service'
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

-- Also add BIDMC as a top-level care site
INSERT INTO staging.cdm_care_site
VALUES (
    staging.omop_id(),
    'BIDMC', 0, 1, 'BIDMC', 'BIDMC',
    'care_site.bidmc', 'null', 0, NULL
);

CREATE INDEX IF NOT EXISTS idx_cs_name ON staging.cdm_care_site (care_site_name);
