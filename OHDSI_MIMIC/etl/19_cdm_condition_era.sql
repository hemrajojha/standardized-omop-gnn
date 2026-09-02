-- =======================================================================
-- 19_cdm_condition_era.sql
-- Derive condition eras by collapsing condition_occurrence records
-- within a 30-day gap window (OHDSI standard algorithm).
--
-- Adapted from: https://github.com/OHDSI/MIMIC/blob/main/etl/etl/cdm_condition_era.sql
-- =======================================================================

DROP TABLE IF EXISTS staging.cdm_condition_era;
CREATE TABLE staging.cdm_condition_era (
    condition_era_id            BIGINT  NOT NULL,
    person_id                   BIGINT  NOT NULL,
    condition_concept_id        BIGINT  NOT NULL,
    condition_era_start_date    DATE    NOT NULL,
    condition_era_end_date      DATE    NOT NULL,
    condition_occurrence_count  INTEGER,
    -- tracking cols
    unit_id     TEXT,
    load_table_id TEXT,
    load_row_id BIGINT,
    trace_id    TEXT
);

-- Step 1: target conditions with end date filled
WITH tmp_target AS (
    SELECT
        condition_occurrence_id,
        person_id,
        condition_concept_id,
        condition_start_date,
        COALESCE(condition_end_date, condition_start_date + INTERVAL '1 day') AS condition_end_date
    FROM staging.cdm_condition_occurrence
    WHERE condition_concept_id <> 0
),
-- Step 2: union of start/end events with ordinal
tmp_dates AS (
    SELECT person_id, condition_concept_id,
           condition_start_date AS event_date, -1 AS event_type,
           ROW_NUMBER() OVER (PARTITION BY person_id, condition_concept_id ORDER BY condition_start_date) AS start_ordinal
    FROM tmp_target
    UNION ALL
    SELECT person_id, condition_concept_id,
           condition_end_date + INTERVAL '30 days', 1, NULL
    FROM tmp_target
),
-- Step 3: running max of start ordinal
tmp_rows AS (
    SELECT *,
        MAX(start_ordinal) OVER (
            PARTITION BY person_id, condition_concept_id
            ORDER BY event_date, event_type
            ROWS UNBOUNDED PRECEDING
        ) AS start_ord,
        ROW_NUMBER() OVER (
            PARTITION BY person_id, condition_concept_id
            ORDER BY event_date, event_type
        ) AS overall_ord
    FROM tmp_dates
),
-- Step 4: identify era boundaries
tmp_ends AS (
    SELECT person_id, condition_concept_id, event_date AS era_end_date,
        ROW_NUMBER() OVER (PARTITION BY person_id, condition_concept_id ORDER BY event_date) AS end_ord
    FROM tmp_rows
    WHERE (2 * start_ord - overall_ord) = 0
),
tmp_starts AS (
    SELECT person_id, condition_concept_id, event_date AS era_start_date,
        start_ordinal AS start_ord
    FROM tmp_rows
    WHERE start_ordinal IS NOT NULL AND event_type = -1
    GROUP BY person_id, condition_concept_id, event_date, start_ordinal
)
INSERT INTO staging.cdm_condition_era
SELECT
    staging.omop_id()           AS condition_era_id,
    s.person_id                 AS person_id,
    s.condition_concept_id      AS condition_concept_id,
    s.era_start_date            AS condition_era_start_date,
    e.era_end_date - INTERVAL '30 days' AS condition_era_end_date,
    COUNT(*) OVER (PARTITION BY s.person_id, s.condition_concept_id
        ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS condition_occurrence_count,
    'condition_era'             AS unit_id,
    'derived'                   AS load_table_id,
    0                           AS load_row_id,
    NULL::TEXT                  AS trace_id
FROM tmp_starts s
JOIN tmp_ends e
    ON  s.person_id = e.person_id
    AND s.condition_concept_id = e.condition_concept_id
    AND s.start_ord = e.end_ord
;
