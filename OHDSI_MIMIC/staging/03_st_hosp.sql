-- =======================================================================
-- 03_st_hosp.sql
-- Create staging snapshots of MIMIC-IV hospital tables:
--   src_diagnoses_icd, src_services (stub), src_labevents, src_d_labitems,
--   src_procedures_icd, src_hcpcsevents (stub), src_drgcodes (stub),
--   src_prescriptions, src_microbiologyevents (stub), src_d_micro (stub),
--   src_pharmacy (stub)
--
-- Adapted from: https://github.com/OHDSI/MIMIC/blob/main/etl/staging/st_hosp.sql
-- Stubs are created as empty tables where source data is unavailable locally.
-- =======================================================================

-- -----------------------------------------------------------------------
-- src_diagnoses_icd
-- -----------------------------------------------------------------------
DROP TABLE IF EXISTS staging.src_diagnoses_icd;
CREATE TABLE staging.src_diagnoses_icd AS
SELECT
    subject_id                                          AS subject_id,
    hadm_id                                             AS hadm_id,
    seq_num                                             AS seq_num,
    icd_code                                            AS icd_code,
    icd_version                                         AS icd_version,
    --
    'diagnoses_icd'                                     AS load_table_id,
    ROW_NUMBER() OVER (ORDER BY hadm_id, seq_num)       AS load_row_id,
    'diagnoses_icd|' || hadm_id::TEXT || '|' || seq_num::TEXT AS trace_id
FROM
    mimiciv_hosp.diagnoses_icd
;

-- -----------------------------------------------------------------------
-- src_services  (stub – table is empty in local dataset)
-- -----------------------------------------------------------------------
DROP TABLE IF EXISTS staging.src_services;
CREATE TABLE staging.src_services AS
SELECT
    subject_id      ::INTEGER   AS subject_id,
    hadm_id         ::INTEGER   AS hadm_id,
    transfertime    ::TIMESTAMP AS transfertime,
    prev_service    ::TEXT      AS prev_service,
    curr_service    ::TEXT      AS curr_service,
    --
    'services'::TEXT            AS load_table_id,
    0::BIGINT                   AS load_row_id,
    ''::TEXT                    AS trace_id
FROM
    mimiciv_hosp.services
;

-- -----------------------------------------------------------------------
-- src_labevents
-- -----------------------------------------------------------------------
DROP TABLE IF EXISTS staging.src_labevents;
CREATE TABLE staging.src_labevents AS
SELECT
    labevent_id                                         AS labevent_id,
    subject_id                                          AS subject_id,
    charttime                                           AS charttime,
    hadm_id                                             AS hadm_id,
    itemid                                              AS itemid,
    valueuom                                            AS valueuom,
    value                                               AS value,
    flag                                                AS flag,
    ref_range_lower                                     AS ref_range_lower,
    ref_range_upper                                     AS ref_range_upper,
    --
    'labevents'                                         AS load_table_id,
    ROW_NUMBER() OVER (ORDER BY labevent_id)            AS load_row_id,
    'labevents|' || labevent_id::TEXT                   AS trace_id
FROM
    mimiciv_hosp.labevents
;

-- -----------------------------------------------------------------------
-- src_d_labitems
-- -----------------------------------------------------------------------
DROP TABLE IF EXISTS staging.src_d_labitems;
CREATE TABLE staging.src_d_labitems AS
SELECT
    itemid                                          AS itemid,
    label                                           AS label,
    fluid                                           AS fluid,
    category                                        AS category,
    NULL::TEXT                                      AS loinc_code,  -- removed in MIMIC IV 2.0
    --
    'd_labitems'                                    AS load_table_id,
    ROW_NUMBER() OVER (ORDER BY itemid)             AS load_row_id,
    'd_labitems|' || itemid::TEXT                   AS trace_id
FROM
    mimiciv_hosp.d_labitems
;

-- -----------------------------------------------------------------------
-- src_procedures_icd
-- -----------------------------------------------------------------------
DROP TABLE IF EXISTS staging.src_procedures_icd;
CREATE TABLE staging.src_procedures_icd AS
SELECT
    subject_id                                          AS subject_id,
    hadm_id                                             AS hadm_id,
    icd_code                                            AS icd_code,
    icd_version                                         AS icd_version,
    --
    'procedures_icd'                                    AS load_table_id,
    ROW_NUMBER() OVER (ORDER BY hadm_id, icd_code)      AS load_row_id,
    'procedures_icd|' || subject_id::TEXT || '|' || hadm_id::TEXT || '|' || icd_code AS trace_id
FROM
    mimiciv_hosp.procedures_icd
;

-- -----------------------------------------------------------------------
-- src_hcpcsevents  (stub)
-- -----------------------------------------------------------------------
DROP TABLE IF EXISTS staging.src_hcpcsevents;
CREATE TABLE staging.src_hcpcsevents AS
SELECT
    hadm_id         ::INTEGER   AS hadm_id,
    subject_id      ::INTEGER   AS subject_id,
    hcpcs_cd        ::TEXT      AS hcpcs_cd,
    seq_num         ::INTEGER   AS seq_num,
    short_description::TEXT     AS short_description,
    'hcpcsevents'::TEXT         AS load_table_id,
    0::BIGINT                   AS load_row_id,
    ''::TEXT                    AS trace_id
FROM
    mimiciv_hosp.hcpcsevents
;

-- -----------------------------------------------------------------------
-- src_drgcodes  (stub)
-- -----------------------------------------------------------------------
DROP TABLE IF EXISTS staging.src_drgcodes;
CREATE TABLE staging.src_drgcodes AS
SELECT
    hadm_id     ::INTEGER   AS hadm_id,
    subject_id  ::INTEGER   AS subject_id,
    drg_code    ::TEXT      AS drg_code,
    description ::TEXT      AS description,
    'drgcodes'::TEXT        AS load_table_id,
    0::BIGINT               AS load_row_id,
    ''::TEXT                AS trace_id
FROM
    mimiciv_hosp.drgcodes
;

-- -----------------------------------------------------------------------
-- src_prescriptions
-- -----------------------------------------------------------------------
DROP TABLE IF EXISTS staging.src_prescriptions;
CREATE TABLE staging.src_prescriptions AS
SELECT
    hadm_id                                             AS hadm_id,
    subject_id                                          AS subject_id,
    pharmacy_id                                         AS pharmacy_id,
    starttime                                           AS starttime,
    stoptime                                            AS stoptime,
    drug_type                                           AS drug_type,
    drug                                                AS drug,
    gsn                                                 AS gsn,
    ndc                                                 AS ndc,
    prod_strength                                       AS prod_strength,
    form_rx                                             AS form_rx,
    dose_val_rx                                         AS dose_val_rx,
    dose_unit_rx                                        AS dose_unit_rx,
    form_val_disp                                       AS form_val_disp,
    form_unit_disp                                      AS form_unit_disp,
    doses_per_24_hrs                                    AS doses_per_24_hrs,
    route                                               AS route,
    --
    'prescriptions'                                     AS load_table_id,
    ROW_NUMBER() OVER (ORDER BY hadm_id, starttime)     AS load_row_id,
    'prescriptions|' || subject_id::TEXT || '|' || hadm_id::TEXT || '|' ||
        COALESCE(pharmacy_id::TEXT, '') || '|' ||
        COALESCE(starttime::TEXT, '')                   AS trace_id
FROM
    mimiciv_hosp.prescriptions
WHERE
    starttime IS NOT NULL
    AND drug IS NOT NULL
;

-- -----------------------------------------------------------------------
-- src_microbiologyevents  (stub)
-- -----------------------------------------------------------------------
DROP TABLE IF EXISTS staging.src_microbiologyevents;
CREATE TABLE staging.src_microbiologyevents AS
SELECT
    microevent_id       ::INTEGER   AS microevent_id,
    subject_id          ::INTEGER   AS subject_id,
    hadm_id             ::INTEGER   AS hadm_id,
    chartdate           ::DATE      AS chartdate,
    charttime           ::TIMESTAMP AS charttime,
    spec_itemid         ::INTEGER   AS spec_itemid,
    spec_type_desc      ::TEXT      AS spec_type_desc,
    test_itemid         ::INTEGER   AS test_itemid,
    test_name           ::TEXT      AS test_name,
    org_itemid          ::INTEGER   AS org_itemid,
    org_name            ::TEXT      AS org_name,
    ab_itemid           ::INTEGER   AS ab_itemid,
    ab_name             ::TEXT      AS ab_name,
    dilution_comparison ::TEXT      AS dilution_comparison,
    dilution_value      ::DOUBLE PRECISION AS dilution_value,
    interpretation      ::TEXT      AS interpretation,
    'microbiologyevents'::TEXT      AS load_table_id,
    0::BIGINT                       AS load_row_id,
    ''::TEXT                        AS trace_id
FROM
    mimiciv_hosp.microbiologyevents
;

-- -----------------------------------------------------------------------
-- src_d_micro  – derived from microbiologyevents (empty in this dataset)
-- -----------------------------------------------------------------------
DROP TABLE IF EXISTS staging.src_d_micro;
CREATE TABLE staging.src_d_micro AS
WITH d_micro AS (
    SELECT DISTINCT ab_itemid AS itemid, ab_name AS label, 'ANTIBIOTIC' AS category,
        'ab_itemid|' || ab_itemid::TEXT AS trace_id
    FROM mimiciv_hosp.microbiologyevents WHERE ab_itemid IS NOT NULL
    UNION ALL
    SELECT DISTINCT test_itemid, test_name, 'MICROTEST',
        'test_itemid|' || test_itemid::TEXT
    FROM mimiciv_hosp.microbiologyevents WHERE test_itemid IS NOT NULL
    UNION ALL
    SELECT DISTINCT org_itemid, org_name, 'ORGANISM',
        'org_itemid|' || org_itemid::TEXT
    FROM mimiciv_hosp.microbiologyevents WHERE org_itemid IS NOT NULL
    UNION ALL
    SELECT DISTINCT spec_itemid, spec_type_desc, 'SPECIMEN',
        'spec_itemid|' || spec_itemid::TEXT
    FROM mimiciv_hosp.microbiologyevents WHERE spec_itemid IS NOT NULL
)
SELECT
    itemid                          AS itemid,
    label                           AS label,
    category                        AS category,
    'microbiologyevents'            AS load_table_id,
    ROW_NUMBER() OVER (ORDER BY itemid) AS load_row_id,
    trace_id                        AS trace_id
FROM d_micro
;

-- -----------------------------------------------------------------------
-- src_pharmacy  (stub)
-- -----------------------------------------------------------------------
DROP TABLE IF EXISTS staging.src_pharmacy;
CREATE TABLE staging.src_pharmacy AS
SELECT
    pharmacy_id ::INTEGER   AS pharmacy_id,
    medication  ::TEXT      AS medication,
    'pharmacy'::TEXT        AS load_table_id,
    0::BIGINT               AS load_row_id,
    ''::TEXT                AS trace_id
FROM
    mimiciv_hosp.pharmacy
;

-- Indexes
CREATE INDEX IF NOT EXISTS idx_src_diag_hadm   ON staging.src_diagnoses_icd (hadm_id);
CREATE INDEX IF NOT EXISTS idx_src_lab_subj    ON staging.src_labevents (subject_id);
CREATE INDEX IF NOT EXISTS idx_src_lab_hadm    ON staging.src_labevents (hadm_id);
CREATE INDEX IF NOT EXISTS idx_src_dlab_item   ON staging.src_d_labitems (itemid);
CREATE INDEX IF NOT EXISTS idx_src_proc_hadm   ON staging.src_procedures_icd (hadm_id);
CREATE INDEX IF NOT EXISTS idx_src_rx_hadm     ON staging.src_prescriptions (hadm_id);
