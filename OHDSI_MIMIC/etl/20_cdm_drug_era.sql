-- =======================================================================
-- 20_cdm_drug_era.sql
-- Derive drug eras by collapsing drug_exposure records
-- within a 30-day gap window (OHDSI standard algorithm).
--
-- Adapted from: https://github.com/OHDSI/MIMIC/blob/main/etl/etl/cdm_drug_era.sql
-- =======================================================================

DROP TABLE IF EXISTS staging.cdm_drug_era;
CREATE TABLE staging.cdm_drug_era (
    drug_era_id             BIGINT  NOT NULL,
    person_id               BIGINT  NOT NULL,
    drug_concept_id         BIGINT  NOT NULL,
    drug_era_start_date     DATE    NOT NULL,
    drug_era_end_date       DATE    NOT NULL,
    drug_exposure_count     INTEGER,
    gap_days                INTEGER,
    -- tracking cols
    unit_id     TEXT,
    load_table_id TEXT,
    load_row_id BIGINT,
    trace_id    TEXT
);

WITH tmp_target AS (
    SELECT drug_exposure_id, person_id, drug_concept_id,
           drug_exposure_start_date,
           COALESCE(drug_exposure_end_date, drug_exposure_start_date + INTERVAL '1 day') AS drug_exposure_end_date
    FROM staging.cdm_drug_exposure
    WHERE drug_concept_id <> 0
),
tmp_dates AS (
    SELECT person_id, drug_concept_id,
           drug_exposure_start_date AS event_date, -1 AS event_type,
           ROW_NUMBER() OVER (PARTITION BY person_id, drug_concept_id ORDER BY drug_exposure_start_date) AS start_ordinal
    FROM tmp_target
    UNION ALL
    SELECT person_id, drug_concept_id,
           drug_exposure_end_date + INTERVAL '30 days', 1, NULL
    FROM tmp_target
),
tmp_rows AS (
    SELECT *,
        MAX(start_ordinal) OVER (
            PARTITION BY person_id, drug_concept_id
            ORDER BY event_date, event_type
            ROWS UNBOUNDED PRECEDING
        ) AS start_ord,
        ROW_NUMBER() OVER (
            PARTITION BY person_id, drug_concept_id
            ORDER BY event_date, event_type
        ) AS overall_ord
    FROM tmp_dates
),
tmp_ends AS (
    SELECT person_id, drug_concept_id, event_date AS era_end_date,
        ROW_NUMBER() OVER (PARTITION BY person_id, drug_concept_id ORDER BY event_date) AS end_ord
    FROM tmp_rows
    WHERE (2 * start_ord - overall_ord) = 0
),
tmp_starts AS (
    SELECT person_id, drug_concept_id, event_date AS era_start_date, start_ordinal AS start_ord
    FROM tmp_rows
    WHERE start_ordinal IS NOT NULL AND event_type = -1
    GROUP BY person_id, drug_concept_id, event_date, start_ordinal
)
INSERT INTO staging.cdm_drug_era
SELECT
    staging.omop_id()           AS drug_era_id,
    s.person_id,
    s.drug_concept_id,
    s.era_start_date            AS drug_era_start_date,
    e.era_end_date - INTERVAL '30 days' AS drug_era_end_date,
    COUNT(*) OVER (PARTITION BY s.person_id, s.drug_concept_id
        ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS drug_exposure_count,
    0                           AS gap_days,
    'drug_era'  AS unit_id, 'derived' AS load_table_id, 0 AS load_row_id, NULL::TEXT AS trace_id
FROM tmp_starts s
JOIN tmp_ends e
    ON  s.person_id = e.person_id
    AND s.drug_concept_id = e.drug_concept_id
    AND s.start_ord = e.end_ord
;
