-- =======================================================================
-- 02_raw_mimic_tables.sql
-- DDL for raw MIMIC-IV source tables.
-- Data is loaded via COPY in staging/00_load_raw_csv.sql.
--
-- Available files (data/raw/mimiciv/):
--   hosp: admissions, patients, diagnoses_icd, d_icd_diagnoses,
--         d_icd_procedures, d_labitems, labevents, prescriptions,
--         procedures_icd
--   icu:  icustays
--
-- Stubs (empty tables) are created for tables referenced in the OHDSI
-- MIMIC ETL but absent from the local dataset so ETL scripts compile.
-- =======================================================================

-- -----------------------------------------------------------------------
-- mimiciv_hosp
-- -----------------------------------------------------------------------

DROP TABLE IF EXISTS mimiciv_hosp.patients CASCADE;
CREATE TABLE mimiciv_hosp.patients (
    subject_id          INTEGER     NOT NULL,
    gender              VARCHAR(1)  NOT NULL,
    anchor_age          INTEGER     NOT NULL,
    anchor_year         INTEGER     NOT NULL,
    anchor_year_group   VARCHAR(20),
    dod                 DATE
);

DROP TABLE IF EXISTS mimiciv_hosp.admissions CASCADE;
CREATE TABLE mimiciv_hosp.admissions (
    subject_id          INTEGER     NOT NULL,
    hadm_id             INTEGER     NOT NULL,
    admittime           TIMESTAMP   NOT NULL,
    dischtime           TIMESTAMP,
    deathtime           TIMESTAMP,
    admission_type      VARCHAR(50),
    admit_provider_id   VARCHAR(10),
    admission_location  VARCHAR(60),
    discharge_location  VARCHAR(60),
    insurance           VARCHAR(30),
    language            VARCHAR(50),
    marital_status      VARCHAR(30),
    race                VARCHAR(80),
    edregtime           TIMESTAMP,
    edouttime           TIMESTAMP,
    hospital_expire_flag SMALLINT
);

DROP TABLE IF EXISTS mimiciv_hosp.diagnoses_icd CASCADE;
CREATE TABLE mimiciv_hosp.diagnoses_icd (
    subject_id  INTEGER NOT NULL,
    hadm_id     INTEGER NOT NULL,
    seq_num     INTEGER NOT NULL,
    icd_code    VARCHAR(10),
    icd_version SMALLINT
);

DROP TABLE IF EXISTS mimiciv_hosp.d_icd_diagnoses CASCADE;
CREATE TABLE mimiciv_hosp.d_icd_diagnoses (
    icd_code    VARCHAR(10) NOT NULL,
    icd_version SMALLINT    NOT NULL,
    long_title  TEXT
);

DROP TABLE IF EXISTS mimiciv_hosp.procedures_icd CASCADE;
CREATE TABLE mimiciv_hosp.procedures_icd (
    subject_id  INTEGER NOT NULL,
    hadm_id     INTEGER NOT NULL,
    seq_num     INTEGER NOT NULL,
    chartdate   DATE,
    icd_code    VARCHAR(10),
    icd_version SMALLINT
);

DROP TABLE IF EXISTS mimiciv_hosp.d_icd_procedures CASCADE;
CREATE TABLE mimiciv_hosp.d_icd_procedures (
    icd_code    VARCHAR(10) NOT NULL,
    icd_version SMALLINT    NOT NULL,
    long_title  TEXT
);

DROP TABLE IF EXISTS mimiciv_hosp.labevents CASCADE;
CREATE TABLE mimiciv_hosp.labevents (
    labevent_id     BIGINT      NOT NULL,
    subject_id      INTEGER     NOT NULL,
    hadm_id         INTEGER,
    specimen_id     INTEGER,
    itemid          INTEGER     NOT NULL,
    order_provider_id VARCHAR(10),
    charttime       TIMESTAMP,
    storetime       TIMESTAMP,
    value           TEXT,
    valuenum        DOUBLE PRECISION,
    valueuom        VARCHAR(20),
    ref_range_lower DOUBLE PRECISION,
    ref_range_upper DOUBLE PRECISION,
    flag            VARCHAR(10),
    priority        VARCHAR(10),
    comments        TEXT
);

DROP TABLE IF EXISTS mimiciv_hosp.d_labitems CASCADE;
CREATE TABLE mimiciv_hosp.d_labitems (
    itemid      INTEGER     NOT NULL,
    label       VARCHAR(100),
    fluid       VARCHAR(50),
    category    VARCHAR(50)
);

DROP TABLE IF EXISTS mimiciv_hosp.prescriptions CASCADE;
CREATE TABLE mimiciv_hosp.prescriptions (
    subject_id      INTEGER     NOT NULL,
    hadm_id         INTEGER     NOT NULL,
    pharmacy_id     INTEGER,
    poe_id          VARCHAR(25),
    poe_seq         INTEGER,
    order_provider_id VARCHAR(10),
    starttime       TIMESTAMP,
    stoptime        TIMESTAMP,
    drug_type       VARCHAR(20),
    drug            VARCHAR(255),
    formulary_drug_cd VARCHAR(20),
    gsn             TEXT,
    ndc             VARCHAR(25),
    prod_strength   VARCHAR(120),
    form_rx         VARCHAR(25),
    dose_val_rx     VARCHAR(50),
    dose_unit_rx    VARCHAR(50),
    form_val_disp   VARCHAR(50),
    form_unit_disp  VARCHAR(50),
    doses_per_24_hrs DOUBLE PRECISION,
    route           VARCHAR(50)
);

-- -----------------------------------------------------------------------
-- Stubs – tables referenced in the OHDSI MIMIC ETL but not locally available
-- -----------------------------------------------------------------------

DROP TABLE IF EXISTS mimiciv_hosp.transfers CASCADE;
CREATE TABLE mimiciv_hosp.transfers (
    subject_id  INTEGER,
    hadm_id     INTEGER,
    transfer_id INTEGER,
    eventtype   VARCHAR(20),
    careunit    VARCHAR(50),
    intime      TIMESTAMP,
    outtime     TIMESTAMP
);

DROP TABLE IF EXISTS mimiciv_hosp.services CASCADE;
CREATE TABLE mimiciv_hosp.services (
    subject_id      INTEGER,
    hadm_id         INTEGER,
    transfertime    TIMESTAMP,
    prev_service    VARCHAR(20),
    curr_service    VARCHAR(20)
);

DROP TABLE IF EXISTS mimiciv_hosp.hcpcsevents CASCADE;
CREATE TABLE mimiciv_hosp.hcpcsevents (
    subject_id          INTEGER,
    hadm_id             INTEGER,
    chartdate           DATE,
    hcpcs_cd            VARCHAR(10),
    seq_num             INTEGER,
    short_description   TEXT
);

DROP TABLE IF EXISTS mimiciv_hosp.drgcodes CASCADE;
CREATE TABLE mimiciv_hosp.drgcodes (
    subject_id      INTEGER,
    hadm_id         INTEGER,
    drg_type        VARCHAR(10),
    drg_code        VARCHAR(10),
    description     TEXT,
    drg_severity    SMALLINT,
    drg_mortality   SMALLINT
);

DROP TABLE IF EXISTS mimiciv_hosp.microbiologyevents CASCADE;
CREATE TABLE mimiciv_hosp.microbiologyevents (
    microevent_id       INTEGER,
    subject_id          INTEGER,
    hadm_id             INTEGER,
    micro_specimen_id   INTEGER,
    order_provider_id   VARCHAR(10),
    chartdate           DATE,
    charttime           TIMESTAMP,
    spec_itemid         INTEGER,
    spec_type_desc      VARCHAR(100),
    test_seq            INTEGER,
    storedate           DATE,
    storetime           TIMESTAMP,
    test_itemid         INTEGER,
    test_name           VARCHAR(100),
    org_itemid          INTEGER,
    org_name            VARCHAR(100),
    isolate_num         SMALLINT,
    quantity            VARCHAR(50),
    ab_itemid           INTEGER,
    ab_name             VARCHAR(60),
    dilution_text       VARCHAR(10),
    dilution_comparison VARCHAR(20),
    dilution_value      DOUBLE PRECISION,
    interpretation      VARCHAR(5),
    comments            TEXT
);

DROP TABLE IF EXISTS mimiciv_hosp.pharmacy CASCADE;
CREATE TABLE mimiciv_hosp.pharmacy (
    subject_id      INTEGER,
    hadm_id         INTEGER,
    pharmacy_id     INTEGER,
    poe_id          VARCHAR(25),
    starttime       TIMESTAMP,
    stoptime        TIMESTAMP,
    medication      TEXT,
    proc_type       VARCHAR(50),
    status          VARCHAR(50),
    entertime       TIMESTAMP,
    verifiedtime    TIMESTAMP,
    route           VARCHAR(50),
    frequency       VARCHAR(50),
    disp_sched      TEXT,
    infusion_type   VARCHAR(15),
    sliding_scale   VARCHAR(1),
    lockout_interval VARCHAR(50),
    basal_rate      DOUBLE PRECISION,
    one_hr_max      VARCHAR(10),
    doses_per_24_hrs DOUBLE PRECISION,
    duration        DOUBLE PRECISION,
    duration_interval VARCHAR(50),
    expiration_value DOUBLE PRECISION,
    expiration_unit VARCHAR(50),
    expirationdate  TIMESTAMP,
    dispensation    VARCHAR(50),
    fill_quantity   VARCHAR(50)
);

-- -----------------------------------------------------------------------
-- mimiciv_icu
-- -----------------------------------------------------------------------

DROP TABLE IF EXISTS mimiciv_icu.icustays CASCADE;
CREATE TABLE mimiciv_icu.icustays (
    subject_id      INTEGER     NOT NULL,
    hadm_id         INTEGER     NOT NULL,
    stay_id         INTEGER     NOT NULL,
    first_careunit  VARCHAR(50),
    last_careunit   VARCHAR(50),
    intime          TIMESTAMP,
    outtime         TIMESTAMP,
    los             DOUBLE PRECISION
);

-- ICU stubs
DROP TABLE IF EXISTS mimiciv_icu.chartevents CASCADE;
CREATE TABLE mimiciv_icu.chartevents (
    subject_id  INTEGER,
    hadm_id     INTEGER,
    stay_id     INTEGER,
    itemid      INTEGER,
    charttime   TIMESTAMP,
    storetime   TIMESTAMP,
    value       TEXT,
    valuenum    DOUBLE PRECISION,
    valueuom    VARCHAR(20),
    warning     SMALLINT
);

DROP TABLE IF EXISTS mimiciv_icu.d_items CASCADE;
CREATE TABLE mimiciv_icu.d_items (
    itemid      INTEGER,
    label       TEXT,
    abbreviation VARCHAR(50),
    linksto     VARCHAR(50),
    category    VARCHAR(50),
    unitname    VARCHAR(50),
    param_type  VARCHAR(30),
    lownormalvalue DOUBLE PRECISION,
    highnormalvalue DOUBLE PRECISION
);

DROP TABLE IF EXISTS mimiciv_icu.procedureevents CASCADE;
CREATE TABLE mimiciv_icu.procedureevents (
    subject_id      INTEGER,
    hadm_id         INTEGER,
    stay_id         INTEGER,
    starttime       TIMESTAMP,
    endtime         TIMESTAMP,
    itemid          INTEGER,
    value           DOUBLE PRECISION,
    valueuom        VARCHAR(20),
    location        VARCHAR(100),
    locationcategory VARCHAR(50),
    storetime       TIMESTAMP,
    orderid         INTEGER,
    linkorderid     INTEGER,
    ordercategoryname VARCHAR(50),
    ordercategorydescription VARCHAR(30),
    patientweight   DOUBLE PRECISION,
    totalamount     DOUBLE PRECISION,
    totalamountuom  VARCHAR(20),
    isopenbag       SMALLINT,
    continueinnextdept SMALLINT,
    statusdescription VARCHAR(30),
    comments_editedby VARCHAR(30),
    comments_canceledby VARCHAR(40),
    comments_date   TIMESTAMP,
    secondaryordercategoryname VARCHAR(50),
    ordercategoryname2 VARCHAR(50)
);

DROP TABLE IF EXISTS mimiciv_icu.datetimeevents CASCADE;
CREATE TABLE mimiciv_icu.datetimeevents (
    subject_id  INTEGER,
    hadm_id     INTEGER,
    stay_id     INTEGER,
    charttime   TIMESTAMP,
    storetime   TIMESTAMP,
    itemid      INTEGER,
    value       TIMESTAMP,
    valueuom    VARCHAR(20),
    warning     SMALLINT
);

DROP TABLE IF EXISTS mimiciv_icu.outputevents CASCADE;
CREATE TABLE mimiciv_icu.outputevents (
    subject_id  INTEGER,
    hadm_id     INTEGER,
    stay_id     INTEGER,
    charttime   TIMESTAMP,
    storetime   TIMESTAMP,
    itemid      INTEGER,
    value       DOUBLE PRECISION,
    valueuom    VARCHAR(20)
);
