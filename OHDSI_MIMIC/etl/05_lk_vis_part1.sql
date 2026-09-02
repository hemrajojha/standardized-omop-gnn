-- =======================================================================
-- 05_lk_vis_part1.sql
-- Build visit lookup tables (Part 1):
--   lk_admissions_clean, lk_transfers_clean, lk_services_clean,
--   lk_visit_clean (combined admissions + transfers),
--   lk_visit_concept (admission/discharge location mapping)
--
-- Adapted from: https://github.com/OHDSI/MIMIC/blob/main/etl/etl/lk_vis_part_1.sql
-- =======================================================================

-- -----------------------------------------------------------------------
-- lk_admissions_clean
-- -----------------------------------------------------------------------
DROP TABLE IF EXISTS staging.lk_admissions_clean;
CREATE TABLE staging.lk_admissions_clean AS
SELECT
    src.subject_id                                          AS subject_id,
    src.hadm_id                                             AS hadm_id,
    CASE WHEN src.edregtime < src.admittime
        THEN src.edregtime
        ELSE src.admittime
    END                                                     AS start_datetime,
    src.dischtime                                           AS end_datetime,
    src.admission_type                                      AS admission_type,
    src.admission_location                                  AS admission_location,
    src.discharge_location                                  AS discharge_location,
    CASE WHEN src.edregtime IS NULL THEN FALSE ELSE TRUE END AS is_er_admission,
    'admissions'                AS unit_id,
    src.load_table_id           AS load_table_id,
    src.load_row_id             AS load_row_id,
    src.trace_id                AS trace_id
FROM
    staging.src_admissions src
;

-- -----------------------------------------------------------------------
-- lk_transfers_clean
-- -----------------------------------------------------------------------
DROP TABLE IF EXISTS staging.lk_transfers_clean;
CREATE TABLE staging.lk_transfers_clean AS
SELECT
    src.subject_id                                          AS subject_id,
    COALESCE(src.hadm_id, vis.hadm_id)                      AS hadm_id,
    CAST(src.intime AS DATE)                                AS date_id,
    src.transfer_id                                         AS transfer_id,
    src.intime                                              AS start_datetime,
    src.outtime                                             AS end_datetime,
    src.careunit                                            AS current_location,
    'transfers'                 AS unit_id,
    src.load_table_id           AS load_table_id,
    src.load_row_id             AS load_row_id,
    src.trace_id                AS trace_id
FROM
    staging.src_transfers src
LEFT JOIN
    staging.lk_admissions_clean vis
        ON  vis.subject_id = src.subject_id
        AND src.intime BETWEEN vis.start_datetime AND vis.end_datetime
        AND src.hadm_id IS NULL
WHERE
    src.eventtype <> 'discharge'
;

-- -----------------------------------------------------------------------
-- lk_services_clean  (stub-safe – src_services may be empty)
-- -----------------------------------------------------------------------
DROP TABLE IF EXISTS staging.lk_services_clean;
CREATE TABLE staging.lk_services_clean AS
SELECT
    src.subject_id                          AS subject_id,
    src.hadm_id                             AS hadm_id,
    src.transfertime                        AS start_datetime,
    LEAD(src.transfertime) OVER (
        PARTITION BY src.subject_id, src.hadm_id
        ORDER BY src.transfertime
    )                                       AS end_datetime,
    src.curr_service                        AS curr_service,
    src.prev_service                        AS prev_service,
    'services'                  AS unit_id,
    src.load_table_id           AS load_table_id,
    src.load_row_id             AS load_row_id,
    src.trace_id                AS trace_id
FROM
    staging.src_services src
;

-- -----------------------------------------------------------------------
-- lk_visit_concept – map admission_type / location strings to concepts
-- -----------------------------------------------------------------------
DROP TABLE IF EXISTS staging.lk_visit_concept;
CREATE TABLE staging.lk_visit_concept AS
SELECT DISTINCT
    src.source_code             AS source_code,
    vc.concept_id               AS source_concept_id,
    vc2.concept_id              AS target_concept_id,
    vc2.domain_id               AS target_domain_id
FROM (
    SELECT DISTINCT admission_type AS source_code FROM staging.src_admissions WHERE admission_type IS NOT NULL
    UNION
    SELECT DISTINCT admission_location FROM staging.src_admissions WHERE admission_location IS NOT NULL
    UNION
    SELECT DISTINCT discharge_location FROM staging.src_admissions WHERE discharge_location IS NOT NULL
) src
LEFT JOIN
    vocab.concept vc
        ON UPPER(vc.concept_code) = UPPER(src.source_code)
        AND vc.vocabulary_id IN ('Visit', 'CMS Place of Service', 'SNOMED', 'UB04 Point of Origin', 'UB04 Pt dis status')
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
-- lk_visit_clean – merged visits (admissions drive visit_occurrence_id)
-- -----------------------------------------------------------------------
DROP TABLE IF EXISTS staging.lk_visit_clean;
CREATE TABLE staging.lk_visit_clean AS
SELECT
    staging.omop_id()                               AS visit_occurrence_id,
    src.subject_id                                  AS subject_id,
    src.hadm_id                                     AS hadm_id,
    src.start_datetime                              AS start_datetime,
    COALESCE(src.end_datetime, src.start_datetime)  AS end_datetime,
    src.admission_type                              AS admission_type,
    src.admission_location                          AS admission_location,
    src.discharge_location                          AS discharge_location,
    src.subject_id::TEXT || '|' || src.hadm_id::TEXT AS source_value,
    src.unit_id                                     AS unit_id,
    src.load_table_id                               AS load_table_id,
    src.load_row_id                                 AS load_row_id,
    src.trace_id                                    AS trace_id
FROM
    staging.lk_admissions_clean src
WHERE
    src.hadm_id IS NOT NULL
;

CREATE INDEX IF NOT EXISTS idx_lk_visit_subj  ON staging.lk_visit_clean (subject_id);
CREATE INDEX IF NOT EXISTS idx_lk_visit_hadm  ON staging.lk_visit_clean (hadm_id);
CREATE INDEX IF NOT EXISTS idx_lk_visit_src   ON staging.lk_visit_clean (source_value);
