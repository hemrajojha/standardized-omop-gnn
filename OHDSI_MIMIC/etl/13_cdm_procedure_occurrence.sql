-- =======================================================================
-- 13_cdm_procedure_occurrence.sql
-- Populate staging.cdm_procedure_occurrence from lk_procedure_mapped.
--
-- Adapted from: https://github.com/OHDSI/MIMIC/blob/main/etl/etl/cdm_procedure_occurrence.sql
-- =======================================================================

DROP TABLE IF EXISTS staging.cdm_procedure_occurrence;
CREATE TABLE staging.cdm_procedure_occurrence (
    procedure_occurrence_id     BIGINT  NOT NULL,
    person_id                   BIGINT  NOT NULL,
    procedure_concept_id        BIGINT  NOT NULL,
    procedure_date              DATE    NOT NULL,
    procedure_datetime          TIMESTAMP,
    procedure_end_date          DATE,
    procedure_end_datetime      TIMESTAMP,
    procedure_type_concept_id   BIGINT  NOT NULL,
    modifier_concept_id         BIGINT,
    quantity                    INTEGER,
    provider_id                 BIGINT,
    visit_occurrence_id         BIGINT,
    visit_detail_id             BIGINT,
    procedure_source_value      TEXT,
    procedure_source_concept_id BIGINT,
    modifier_source_value       TEXT,
    -- tracking cols
    unit_id         TEXT,
    load_table_id   TEXT,
    load_row_id     BIGINT,
    trace_id        TEXT
);

INSERT INTO staging.cdm_procedure_occurrence
SELECT
    staging.omop_id()                           AS procedure_occurrence_id,
    per.person_id                               AS person_id,
    COALESCE(src.target_concept_id, 0)          AS procedure_concept_id,
    CAST(src.start_datetime AS DATE)            AS procedure_date,
    src.start_datetime                          AS procedure_datetime,
    NULL::DATE                                  AS procedure_end_date,
    NULL::TIMESTAMP                             AS procedure_end_datetime,
    src.type_concept_id                         AS procedure_type_concept_id,
    0                                           AS modifier_concept_id,
    CAST(src.quantity AS INTEGER)               AS quantity,
    NULL::BIGINT                                AS provider_id,
    vis.visit_occurrence_id                     AS visit_occurrence_id,
    NULL::BIGINT                                AS visit_detail_id,
    src.source_code                             AS procedure_source_value,
    COALESCE(src.source_concept_id, 0)          AS procedure_source_concept_id,
    NULL::TEXT                                  AS modifier_source_value,
    'procedure.' || src.unit_id                 AS unit_id,
    src.load_table_id                           AS load_table_id,
    src.load_row_id                             AS load_row_id,
    src.trace_id                                AS trace_id
FROM
    staging.lk_procedure_mapped src
INNER JOIN
    staging.cdm_person per
        ON src.subject_id::TEXT = per.person_source_value
INNER JOIN
    staging.cdm_visit_occurrence vis
        ON vis.visit_source_value =
            src.subject_id::TEXT || '|' || src.hadm_id::TEXT
WHERE
    src.target_domain_id = 'Procedure'
;

CREATE INDEX IF NOT EXISTS idx_cdm_po_person ON staging.cdm_procedure_occurrence (person_id);
CREATE INDEX IF NOT EXISTS idx_cdm_po_visit  ON staging.cdm_procedure_occurrence (visit_occurrence_id);
