-- =======================================================================
-- 02_st_core.sql
-- Create staging snapshots of MIMIC-IV core tables:
--   src_patients, src_admissions, src_transfers
--
-- Adapted from: https://github.com/OHDSI/MIMIC/blob/main/etl/staging/st_core.sql
-- Changes: BigQuery → PostgreSQL, @variables → schema-qualified names,
--   FARM_FINGERPRINT/GENERATE_UUID → staging.omop_id(),
--   TO_JSON_STRING(STRUCT(...)) → simple text concatenation for trace_id.
-- =======================================================================

-- -----------------------------------------------------------------------
-- src_patients
-- -----------------------------------------------------------------------
DROP TABLE IF EXISTS staging.src_patients;
CREATE TABLE staging.src_patients AS
SELECT
    subject_id                              AS subject_id,
    anchor_year                             AS anchor_year,
    anchor_age                              AS anchor_age,
    anchor_year_group                       AS anchor_year_group,
    gender                                  AS gender,
    --
    'patients'                              AS load_table_id,
    ROW_NUMBER() OVER (ORDER BY subject_id) AS load_row_id,
    'patients|' || subject_id::TEXT         AS trace_id
FROM
    mimiciv_hosp.patients
;

-- -----------------------------------------------------------------------
-- src_admissions
-- -----------------------------------------------------------------------
DROP TABLE IF EXISTS staging.src_admissions;
CREATE TABLE staging.src_admissions AS
SELECT
    hadm_id                                     AS hadm_id,
    subject_id                                  AS subject_id,
    admittime                                   AS admittime,
    dischtime                                   AS dischtime,
    deathtime                                   AS deathtime,
    admission_type                              AS admission_type,
    admission_location                          AS admission_location,
    discharge_location                          AS discharge_location,
    race                                        AS ethnicity,   -- MIMIC IV 2.0: race replaced ethnicity
    edregtime                                   AS edregtime,
    insurance                                   AS insurance,
    marital_status                              AS marital_status,
    language                                    AS language,
    --
    'admissions'                                AS load_table_id,
    ROW_NUMBER() OVER (ORDER BY hadm_id)        AS load_row_id,
    'admissions|' || subject_id::TEXT || '|' || hadm_id::TEXT AS trace_id
FROM
    mimiciv_hosp.admissions
;

-- -----------------------------------------------------------------------
-- src_transfers
-- -----------------------------------------------------------------------
DROP TABLE IF EXISTS staging.src_transfers;
CREATE TABLE staging.src_transfers AS
SELECT
    transfer_id                                     AS transfer_id,
    hadm_id                                         AS hadm_id,
    subject_id                                      AS subject_id,
    careunit                                        AS careunit,
    intime                                          AS intime,
    outtime                                         AS outtime,
    eventtype                                       AS eventtype,
    --
    'transfers'                                     AS load_table_id,
    ROW_NUMBER() OVER (ORDER BY transfer_id)        AS load_row_id,
    'transfers|' || subject_id::TEXT || '|' || hadm_id::TEXT || '|' || transfer_id::TEXT AS trace_id
FROM
    mimiciv_hosp.transfers
;

-- Indexes for ETL join performance
CREATE INDEX IF NOT EXISTS idx_src_patients_subj  ON staging.src_patients (subject_id);
CREATE INDEX IF NOT EXISTS idx_src_adm_hadm       ON staging.src_admissions (hadm_id);
CREATE INDEX IF NOT EXISTS idx_src_adm_subj       ON staging.src_admissions (subject_id);
CREATE INDEX IF NOT EXISTS idx_src_tr_hadm        ON staging.src_transfers (hadm_id);
CREATE INDEX IF NOT EXISTS idx_src_tr_subj        ON staging.src_transfers (subject_id);
