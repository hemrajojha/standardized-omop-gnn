-- =======================================================================
-- 16_cdm_drug_exposure.sql
-- Populate staging.cdm_drug_exposure from lk_drug_mapped.
--
-- Adapted from: https://github.com/OHDSI/MIMIC/blob/main/etl/etl/cdm_drug_exposure.sql
-- =======================================================================

DROP TABLE IF EXISTS staging.cdm_drug_exposure;
CREATE TABLE staging.cdm_drug_exposure (
    drug_exposure_id                BIGINT  NOT NULL,
    person_id                       BIGINT  NOT NULL,
    drug_concept_id                 BIGINT  NOT NULL,
    drug_exposure_start_date        DATE    NOT NULL,
    drug_exposure_start_datetime    TIMESTAMP,
    drug_exposure_end_date          DATE    NOT NULL,
    drug_exposure_end_datetime      TIMESTAMP,
    verbatim_end_date               DATE,
    drug_type_concept_id            BIGINT  NOT NULL,
    stop_reason                     TEXT,
    refills                         INTEGER,
    quantity                        FLOAT,
    days_supply                     INTEGER,
    sig                             TEXT,
    route_concept_id                BIGINT,
    lot_number                      TEXT,
    provider_id                     BIGINT,
    visit_occurrence_id             BIGINT,
    visit_detail_id                 BIGINT,
    drug_source_value               TEXT,
    drug_source_concept_id          BIGINT,
    route_source_value              TEXT,
    dose_unit_source_value          TEXT,
    -- tracking cols
    unit_id         TEXT,
    load_table_id   TEXT,
    load_row_id     BIGINT,
    trace_id        TEXT
);

INSERT INTO staging.cdm_drug_exposure
SELECT
    staging.omop_id()                           AS drug_exposure_id,
    per.person_id                               AS person_id,
    COALESCE(src.target_concept_id, 0)          AS drug_concept_id,
    CAST(src.start_datetime AS DATE)            AS drug_exposure_start_date,
    src.start_datetime                          AS drug_exposure_start_datetime,
    CAST(src.end_datetime AS DATE)              AS drug_exposure_end_date,
    src.end_datetime                            AS drug_exposure_end_datetime,
    NULL::DATE                                  AS verbatim_end_date,
    src.type_concept_id                         AS drug_type_concept_id,
    NULL::TEXT                                  AS stop_reason,
    NULL::INTEGER                               AS refills,
    src.quantity                                AS quantity,
    NULL::INTEGER                               AS days_supply,
    NULL::TEXT                                  AS sig,
    src.route_concept_id                        AS route_concept_id,
    NULL::TEXT                                  AS lot_number,
    NULL::BIGINT                                AS provider_id,
    vis.visit_occurrence_id                     AS visit_occurrence_id,
    NULL::BIGINT                                AS visit_detail_id,
    src.source_code                             AS drug_source_value,
    COALESCE(src.source_concept_id, 0)          AS drug_source_concept_id,
    src.route_source_value                      AS route_source_value,
    src.dose_unit_source_value                  AS dose_unit_source_value,
    'drug.' || src.unit_id                      AS unit_id,
    src.load_table_id                           AS load_table_id,
    src.load_row_id                             AS load_row_id,
    src.trace_id                                AS trace_id
FROM
    staging.lk_drug_mapped src
INNER JOIN
    omopcdm.person per
        ON src.subject_id::TEXT = per.person_source_value
INNER JOIN
    omopcdm.visit_occurrence vis
        ON vis.visit_source_value =
            src.subject_id::TEXT || '|' || src.hadm_id::TEXT
;

CREATE INDEX IF NOT EXISTS idx_cdm_drug_person ON staging.cdm_drug_exposure (person_id);
CREATE INDEX IF NOT EXISTS idx_cdm_drug_visit  ON staging.cdm_drug_exposure (visit_occurrence_id);
