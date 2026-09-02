-- =======================================================================
-- 03_cdm_person.sql
-- Populate staging.cdm_person from src_patients + src_admissions.
--
-- Adapted from: https://github.com/OHDSI/MIMIC/blob/main/etl/etl/cdm_person.sql
-- Race/ethnicity mapped through vocab.concept (domain_id IN ('Race','Ethnicity')).
-- =======================================================================

-- -----------------------------------------------------------------------
-- tmp_subject_ethnicity – first ethnicity per subject from admissions
-- -----------------------------------------------------------------------
DROP TABLE IF EXISTS staging.tmp_subject_ethnicity;
CREATE TABLE staging.tmp_subject_ethnicity AS
SELECT DISTINCT
    src.subject_id                          AS subject_id,
    FIRST_VALUE(src.ethnicity) OVER (
        PARTITION BY src.subject_id
        ORDER BY src.admittime ASC
        ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
    )                                       AS ethnicity_first
FROM
    staging.src_admissions src
;

-- -----------------------------------------------------------------------
-- lk_pat_ethnicity_concept – map MIMIC race string to OMOP standard
-- -----------------------------------------------------------------------
DROP TABLE IF EXISTS staging.lk_pat_ethnicity_concept;
CREATE TABLE staging.lk_pat_ethnicity_concept AS
SELECT DISTINCT
    src.ethnicity_first         AS source_code,
    vc.concept_id               AS source_concept_id,
    vc.vocabulary_id            AS source_vocabulary_id,
    vc1.concept_id              AS target_concept_id,
    vc1.vocabulary_id           AS target_vocabulary_id
FROM
    staging.tmp_subject_ethnicity src
LEFT JOIN
    vocab.concept vc
        ON UPPER(vc.concept_code) = UPPER(src.ethnicity_first)
        AND vc.domain_id IN ('Race', 'Ethnicity')
LEFT JOIN
    vocab.concept_relationship cr1
        ON  cr1.concept_id_1 = vc.concept_id
        AND cr1.relationship_id = 'Maps to'
LEFT JOIN
    vocab.concept vc1
        ON  cr1.concept_id_2 = vc1.concept_id
        AND vc1.invalid_reason IS NULL
        AND vc1.standard_concept = 'S'
;

-- -----------------------------------------------------------------------
-- cdm_person
-- -----------------------------------------------------------------------
DROP TABLE IF EXISTS staging.cdm_person;
CREATE TABLE staging.cdm_person (
    person_id                   BIGINT  NOT NULL,
    gender_concept_id           BIGINT  NOT NULL,
    year_of_birth               INTEGER NOT NULL,
    month_of_birth              INTEGER,
    day_of_birth                INTEGER,
    birth_datetime              TIMESTAMP,
    race_concept_id             BIGINT  NOT NULL,
    ethnicity_concept_id        BIGINT  NOT NULL,
    location_id                 BIGINT,
    provider_id                 BIGINT,
    care_site_id                BIGINT,
    person_source_value         TEXT,
    gender_source_value         TEXT,
    gender_source_concept_id    BIGINT,
    race_source_value           TEXT,
    race_source_concept_id      BIGINT,
    ethnicity_source_value      TEXT,
    ethnicity_source_concept_id BIGINT,
    -- tracking cols
    unit_id         TEXT,
    load_table_id   TEXT,
    load_row_id     BIGINT,
    trace_id        TEXT
);

INSERT INTO staging.cdm_person
SELECT
    staging.omop_id()                           AS person_id,
    CASE
        WHEN p.gender = 'F' THEN 8532
        WHEN p.gender = 'M' THEN 8507
        ELSE 0
    END                                         AS gender_concept_id,
    p.anchor_year - p.anchor_age                AS year_of_birth,
    NULL::INTEGER                               AS month_of_birth,
    NULL::INTEGER                               AS day_of_birth,
    NULL::TIMESTAMP                             AS birth_datetime,
    COALESCE(
        CASE
            WHEN map_eth.target_vocabulary_id <> 'Ethnicity'
                THEN map_eth.target_concept_id
            ELSE NULL
        END, 0)                                 AS race_concept_id,
    COALESCE(
        CASE
            WHEN map_eth.target_vocabulary_id = 'Ethnicity'
                THEN map_eth.target_concept_id
            ELSE NULL
        END, 0)                                 AS ethnicity_concept_id,
    NULL::BIGINT                                AS location_id,
    NULL::BIGINT                                AS provider_id,
    NULL::BIGINT                                AS care_site_id,
    p.subject_id::TEXT                          AS person_source_value,
    p.gender                                    AS gender_source_value,
    0                                           AS gender_source_concept_id,
    CASE
        WHEN map_eth.target_vocabulary_id <> 'Ethnicity'
            THEN eth.ethnicity_first
        ELSE NULL
    END                                         AS race_source_value,
    COALESCE(
        CASE
            WHEN map_eth.target_vocabulary_id <> 'Ethnicity'
                THEN map_eth.source_concept_id
            ELSE NULL
        END, 0)                                 AS race_source_concept_id,
    CASE
        WHEN map_eth.target_vocabulary_id = 'Ethnicity'
            THEN eth.ethnicity_first
        ELSE NULL
    END                                         AS ethnicity_source_value,
    COALESCE(
        CASE
            WHEN map_eth.target_vocabulary_id = 'Ethnicity'
                THEN map_eth.source_concept_id
            ELSE NULL
        END, 0)                                 AS ethnicity_source_concept_id,
    'person.patients'       AS unit_id,
    p.load_table_id         AS load_table_id,
    p.load_row_id           AS load_row_id,
    p.trace_id              AS trace_id
FROM
    staging.src_patients p
LEFT JOIN
    staging.tmp_subject_ethnicity eth
        ON  p.subject_id = eth.subject_id
LEFT JOIN
    staging.lk_pat_ethnicity_concept map_eth
        ON  eth.ethnicity_first = map_eth.source_code
;

DROP TABLE IF EXISTS staging.tmp_subject_ethnicity;

CREATE INDEX IF NOT EXISTS idx_cdm_person_src ON staging.cdm_person (person_source_value);
CREATE INDEX IF NOT EXISTS idx_cdm_person_id  ON staging.cdm_person (person_id);
