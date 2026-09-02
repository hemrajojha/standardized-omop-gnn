-- =======================================================================
-- 00_load_raw_csv.sql
-- Load MIMIC-IV CSV files into mimiciv_hosp and mimiciv_icu schemas.
--
-- NOTE: \COPY must be on a single line (psql meta-command limitation).
--
-- Run via psql from the project root:
--   psql -U postgres -d mimiciv_omop -f "OHDSI_MIMIC/staging/00_load_raw_csv.sql"
-- =======================================================================

-- -----------------------------------------------------------------------
-- mimiciv_hosp
-- -----------------------------------------------------------------------

TRUNCATE mimiciv_hosp.patients;
\COPY mimiciv_hosp.patients (subject_id, gender, anchor_age, anchor_year, anchor_year_group, dod) FROM '/path/to/mimiciv/hosp/patients.csv' CSV HEADER;

ALTER TABLE mimiciv_hosp.admissions ALTER COLUMN language TYPE VARCHAR(50);
TRUNCATE mimiciv_hosp.admissions;
\COPY mimiciv_hosp.admissions (subject_id, hadm_id, admittime, dischtime, deathtime, admission_type, admit_provider_id, admission_location, discharge_location, insurance, language, marital_status, race, edregtime, edouttime, hospital_expire_flag) FROM '/path/to/mimiciv/hosp/admissions.csv' CSV HEADER;

TRUNCATE mimiciv_hosp.diagnoses_icd;
\COPY mimiciv_hosp.diagnoses_icd (subject_id, hadm_id, seq_num, icd_code, icd_version) FROM '/path/to/mimiciv/hosp/diagnoses_icd.csv' CSV HEADER;

TRUNCATE mimiciv_hosp.d_icd_diagnoses;
\COPY mimiciv_hosp.d_icd_diagnoses (icd_code, icd_version, long_title) FROM '/path/to/mimiciv/hosp/d_icd_diagnoses.csv' CSV HEADER;

TRUNCATE mimiciv_hosp.procedures_icd;
\COPY mimiciv_hosp.procedures_icd (subject_id, hadm_id, seq_num, chartdate, icd_code, icd_version) FROM '/path/to/mimiciv/hosp/procedures_icd.csv' CSV HEADER;

TRUNCATE mimiciv_hosp.d_icd_procedures;
\COPY mimiciv_hosp.d_icd_procedures (icd_code, icd_version, long_title) FROM '/path/to/mimiciv/hosp/d_icd_procedures.csv' CSV HEADER;

TRUNCATE mimiciv_hosp.labevents;
\COPY mimiciv_hosp.labevents (labevent_id, subject_id, hadm_id, specimen_id, itemid, order_provider_id, charttime, storetime, value, valuenum, valueuom, ref_range_lower, ref_range_upper, flag, priority, comments) FROM '/path/to/mimiciv/hosp/labevents.csv' CSV HEADER;

TRUNCATE mimiciv_hosp.d_labitems;
\COPY mimiciv_hosp.d_labitems (itemid, label, fluid, category) FROM '/path/to/mimiciv/hosp/d_labitems.csv' CSV HEADER;

TRUNCATE mimiciv_hosp.prescriptions;
\COPY mimiciv_hosp.prescriptions (subject_id, hadm_id, pharmacy_id, poe_id, poe_seq, order_provider_id, starttime, stoptime, drug_type, drug, formulary_drug_cd, gsn, ndc, prod_strength, form_rx, dose_val_rx, dose_unit_rx, form_val_disp, form_unit_disp, doses_per_24_hrs, route) FROM '/path/to/mimiciv/hosp/prescriptions.csv' CSV HEADER ENCODING 'LATIN1';

-- -----------------------------------------------------------------------
-- mimiciv_icu
-- -----------------------------------------------------------------------

TRUNCATE mimiciv_icu.icustays;
\COPY mimiciv_icu.icustays (subject_id, hadm_id, stay_id, first_careunit, last_careunit, intime, outtime, los) FROM '/path/to/mimiciv/icu/icustays.csv' CSV HEADER;

-- -----------------------------------------------------------------------
-- Populate transfers synthetically from admissions + icustays
-- -----------------------------------------------------------------------
TRUNCATE mimiciv_hosp.transfers;

INSERT INTO mimiciv_hosp.transfers (subject_id, hadm_id, transfer_id, eventtype, careunit, intime, outtime)
SELECT
    a.subject_id,
    a.hadm_id,
    ROW_NUMBER() OVER (ORDER BY a.hadm_id)      AS transfer_id,
    'admit'                                      AS eventtype,
    COALESCE(a.admission_location, 'UNKNOWN')    AS careunit,
    a.admittime                                  AS intime,
    a.dischtime                                  AS outtime
FROM mimiciv_hosp.admissions a
WHERE a.hadm_id IS NOT NULL;

INSERT INTO mimiciv_hosp.transfers (subject_id, hadm_id, transfer_id, eventtype, careunit, intime, outtime)
SELECT
    i.subject_id,
    i.hadm_id,
    (SELECT COALESCE(MAX(transfer_id), 0) FROM mimiciv_hosp.transfers) + ROW_NUMBER() OVER (ORDER BY i.stay_id),
    'transfer'       AS eventtype,
    i.first_careunit AS careunit,
    i.intime,
    i.outtime
FROM mimiciv_icu.icustays i;

-- -----------------------------------------------------------------------
-- Row counts for verification
-- -----------------------------------------------------------------------
SELECT 'patients'       AS tbl, COUNT(*) AS n FROM mimiciv_hosp.patients
UNION ALL SELECT 'admissions',     COUNT(*) FROM mimiciv_hosp.admissions
UNION ALL SELECT 'diagnoses_icd',  COUNT(*) FROM mimiciv_hosp.diagnoses_icd
UNION ALL SELECT 'procedures_icd', COUNT(*) FROM mimiciv_hosp.procedures_icd
UNION ALL SELECT 'labevents',      COUNT(*) FROM mimiciv_hosp.labevents
UNION ALL SELECT 'd_labitems',     COUNT(*) FROM mimiciv_hosp.d_labitems
UNION ALL SELECT 'prescriptions',  COUNT(*) FROM mimiciv_hosp.prescriptions
UNION ALL SELECT 'transfers',      COUNT(*) FROM mimiciv_hosp.transfers
UNION ALL SELECT 'icustays',       COUNT(*) FROM mimiciv_icu.icustays
ORDER BY tbl;
