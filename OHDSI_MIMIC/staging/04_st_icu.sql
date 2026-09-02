-- =======================================================================
-- 04_st_icu.sql
-- Create staging snapshots of MIMIC-IV ICU tables:
--   src_procedureevents, src_d_items, src_datetimeevents,
--   src_chartevents, src_outputevents
--
-- Adapted from: https://github.com/OHDSI/MIMIC/blob/main/etl/staging/st_icu.sql
-- Note: chartevents, procedureevents, datetimeevents, outputevents, d_items
--       are stubs (empty) in this local dataset. Only icustays is populated.
-- =======================================================================

-- -----------------------------------------------------------------------
-- src_procedureevents  (stub)
-- -----------------------------------------------------------------------
DROP TABLE IF EXISTS staging.src_procedureevents;
CREATE TABLE staging.src_procedureevents AS
SELECT
    hadm_id         ::INTEGER   AS hadm_id,
    subject_id      ::INTEGER   AS subject_id,
    stay_id         ::INTEGER   AS stay_id,
    itemid          ::INTEGER   AS itemid,
    starttime       ::TIMESTAMP AS starttime,
    value           ::DOUBLE PRECISION AS value,
    0::INTEGER                  AS cancelreason,   -- removed in MIMIC IV 2.0
    'procedureevents'::TEXT     AS load_table_id,
    0::BIGINT                   AS load_row_id,
    ''::TEXT                    AS trace_id
FROM
    mimiciv_icu.procedureevents
;

-- -----------------------------------------------------------------------
-- src_d_items  (stub)
-- -----------------------------------------------------------------------
DROP TABLE IF EXISTS staging.src_d_items;
CREATE TABLE staging.src_d_items AS
SELECT
    itemid          ::INTEGER   AS itemid,
    label           ::TEXT      AS label,
    linksto         ::TEXT      AS linksto,
    'd_items'::TEXT             AS load_table_id,
    0::BIGINT                   AS load_row_id,
    'd_items|' || itemid::TEXT  AS trace_id
FROM
    mimiciv_icu.d_items
;

-- -----------------------------------------------------------------------
-- src_datetimeevents  (stub)
-- -----------------------------------------------------------------------
DROP TABLE IF EXISTS staging.src_datetimeevents;
CREATE TABLE staging.src_datetimeevents AS
SELECT
    subject_id      ::INTEGER   AS subject_id,
    hadm_id         ::INTEGER   AS hadm_id,
    stay_id         ::INTEGER   AS stay_id,
    itemid          ::INTEGER   AS itemid,
    charttime       ::TIMESTAMP AS charttime,
    value           ::TIMESTAMP AS value,
    'datetimeevents'::TEXT      AS load_table_id,
    0::BIGINT                   AS load_row_id,
    ''::TEXT                    AS trace_id
FROM
    mimiciv_icu.datetimeevents
;

-- -----------------------------------------------------------------------
-- src_chartevents  (stub)
-- -----------------------------------------------------------------------
DROP TABLE IF EXISTS staging.src_chartevents;
CREATE TABLE staging.src_chartevents AS
SELECT
    subject_id      ::INTEGER           AS subject_id,
    hadm_id         ::INTEGER           AS hadm_id,
    stay_id         ::INTEGER           AS stay_id,
    itemid          ::INTEGER           AS itemid,
    charttime       ::TIMESTAMP         AS charttime,
    value           ::TEXT              AS value,
    valuenum        ::DOUBLE PRECISION  AS valuenum,
    valueuom        ::TEXT              AS valueuom,
    'chartevents'::TEXT                 AS load_table_id,
    0::BIGINT                           AS load_row_id,
    ''::TEXT                            AS trace_id
FROM
    mimiciv_icu.chartevents
;

-- -----------------------------------------------------------------------
-- src_outputevents  (stub)
-- -----------------------------------------------------------------------
DROP TABLE IF EXISTS staging.src_outputevents;
CREATE TABLE staging.src_outputevents AS
SELECT
    subject_id      ::INTEGER           AS subject_id,
    hadm_id         ::INTEGER           AS hadm_id,
    stay_id         ::INTEGER           AS stay_id,
    charttime       ::TIMESTAMP         AS charttime,
    NULL::TIMESTAMP                     AS storetime,
    itemid          ::INTEGER           AS itemid,
    value           ::DOUBLE PRECISION  AS value,
    valueuom        ::TEXT              AS valueuom,
    'outputevents'::TEXT                AS load_table_id,
    0::BIGINT                           AS load_row_id,
    ''::TEXT                            AS trace_id
FROM
    mimiciv_icu.outputevents
;
