-- =======================================================================
-- 04_cdm_death.sql
-- Populate staging.cdm_death from src_admissions deathtime.
--
-- Adapted from: https://github.com/OHDSI/MIMIC/blob/main/etl/etl/cdm_death.sql
-- =======================================================================

-- -----------------------------------------------------------------------
-- lk_death_adm_mapped – earliest admission with a deathtime
-- -----------------------------------------------------------------------
DROP TABLE IF EXISTS staging.lk_death_adm_mapped;
CREATE TABLE staging.lk_death_adm_mapped AS
SELECT DISTINCT
    src.subject_id,
    FIRST_VALUE(src.deathtime) OVER (
        PARTITION BY src.subject_id
        ORDER BY src.admittime ASC
        ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
    )                                   AS deathtime,
    FIRST_VALUE(src.dischtime) OVER (
        PARTITION BY src.subject_id
        ORDER BY src.admittime ASC
        ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
    )                                   AS dischtime,
    32817                               AS type_concept_id,  -- EHR
    'admissions'                        AS unit_id,
    src.load_table_id                   AS load_table_id,
    FIRST_VALUE(src.load_row_id) OVER (
        PARTITION BY src.subject_id
        ORDER BY src.admittime ASC
        ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
    )                                   AS load_row_id,
    FIRST_VALUE(src.trace_id) OVER (
        PARTITION BY src.subject_id
        ORDER BY src.admittime ASC
        ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
    )                                   AS trace_id
FROM
    staging.src_admissions src
WHERE
    src.deathtime IS NOT NULL
;

-- -----------------------------------------------------------------------
-- cdm_death
-- -----------------------------------------------------------------------
DROP TABLE IF EXISTS staging.cdm_death;
CREATE TABLE staging.cdm_death (
    person_id               BIGINT  NOT NULL,
    death_date              DATE    NOT NULL,
    death_datetime          TIMESTAMP,
    death_type_concept_id   BIGINT  NOT NULL,
    cause_concept_id        BIGINT,
    cause_source_value      TEXT,
    cause_source_concept_id BIGINT,
    -- tracking cols
    unit_id         TEXT,
    load_table_id   TEXT,
    load_row_id     BIGINT,
    trace_id        TEXT
);

INSERT INTO staging.cdm_death
SELECT
    per.person_id                           AS person_id,
    CAST(
        CASE WHEN src.deathtime <= src.dischtime
            THEN src.deathtime
            ELSE src.dischtime
        END
    AS DATE)                                AS death_date,
    CASE WHEN src.deathtime <= src.dischtime
        THEN src.deathtime
        ELSE src.dischtime
    END                                     AS death_datetime,
    src.type_concept_id                     AS death_type_concept_id,
    0                                       AS cause_concept_id,
    NULL::TEXT                              AS cause_source_value,
    0                                       AS cause_source_concept_id,
    'death.' || src.unit_id                 AS unit_id,
    src.load_table_id                       AS load_table_id,
    src.load_row_id                         AS load_row_id,
    src.trace_id                            AS trace_id
FROM
    staging.lk_death_adm_mapped src
INNER JOIN
    staging.cdm_person per
        ON src.subject_id::TEXT = per.person_source_value
;
