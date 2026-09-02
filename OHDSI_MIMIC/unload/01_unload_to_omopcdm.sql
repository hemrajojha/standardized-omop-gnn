-- =======================================================================
-- 01_unload_to_omopcdm.sql
-- Copy finalised staging.cdm_* tables into the clean omopcdm schema,
-- stripping ETL tracking columns (unit_id, load_table_id, etc.).
-- Also copies vocabulary tables so omopcdm is fully self-contained
-- for ATLAS / WebAPI.
--
-- Adapted from: https://github.com/OHDSI/MIMIC/blob/main/etl/unload/unload_to_atlas_gen.sql
-- =======================================================================

-- -----------------------------------------------------------------------
-- Vocabulary tables
-- -----------------------------------------------------------------------

TRUNCATE omopcdm.concept;
INSERT INTO omopcdm.concept SELECT * FROM vocab.concept;

TRUNCATE omopcdm.vocabulary;
INSERT INTO omopcdm.vocabulary SELECT * FROM vocab.vocabulary;

TRUNCATE omopcdm.domain;
INSERT INTO omopcdm.domain SELECT * FROM vocab.domain;

TRUNCATE omopcdm.concept_class;
INSERT INTO omopcdm.concept_class SELECT * FROM vocab.concept_class;

TRUNCATE omopcdm.concept_relationship;
INSERT INTO omopcdm.concept_relationship SELECT * FROM vocab.concept_relationship;

TRUNCATE omopcdm.relationship;
INSERT INTO omopcdm.relationship SELECT * FROM vocab.relationship;

TRUNCATE omopcdm.concept_synonym;
INSERT INTO omopcdm.concept_synonym SELECT * FROM vocab.concept_synonym;

TRUNCATE omopcdm.concept_ancestor;
INSERT INTO omopcdm.concept_ancestor SELECT * FROM vocab.concept_ancestor;

TRUNCATE omopcdm.drug_strength;
INSERT INTO omopcdm.drug_strength SELECT * FROM vocab.drug_strength;

-- -----------------------------------------------------------------------
-- Widen surrogate ID columns INTEGER → BIGINT
-- (staging.omop_id() generates 60-bit BIGINT values that overflow INTEGER)
-- -----------------------------------------------------------------------

ALTER TABLE omopcdm.location
    ALTER COLUMN location_id TYPE BIGINT;

ALTER TABLE omopcdm.care_site
    ALTER COLUMN care_site_id TYPE BIGINT,
    ALTER COLUMN location_id  TYPE BIGINT;

ALTER TABLE omopcdm.person
    ALTER COLUMN person_id   TYPE BIGINT,
    ALTER COLUMN location_id TYPE BIGINT,
    ALTER COLUMN care_site_id TYPE BIGINT,
    ALTER COLUMN provider_id  TYPE BIGINT;

ALTER TABLE omopcdm.death
    ALTER COLUMN person_id TYPE BIGINT;

ALTER TABLE omopcdm.observation_period
    ALTER COLUMN observation_period_id TYPE BIGINT,
    ALTER COLUMN person_id             TYPE BIGINT;

ALTER TABLE omopcdm.visit_occurrence
    ALTER COLUMN visit_occurrence_id           TYPE BIGINT,
    ALTER COLUMN person_id                     TYPE BIGINT,
    ALTER COLUMN care_site_id                  TYPE BIGINT,
    ALTER COLUMN provider_id                   TYPE BIGINT,
    ALTER COLUMN preceding_visit_occurrence_id TYPE BIGINT;

ALTER TABLE omopcdm.visit_detail
    ALTER COLUMN visit_detail_id             TYPE BIGINT,
    ALTER COLUMN person_id                   TYPE BIGINT,
    ALTER COLUMN care_site_id                TYPE BIGINT,
    ALTER COLUMN provider_id                 TYPE BIGINT,
    ALTER COLUMN preceding_visit_detail_id   TYPE BIGINT,
    ALTER COLUMN parent_visit_detail_id      TYPE BIGINT,
    ALTER COLUMN visit_occurrence_id         TYPE BIGINT;

ALTER TABLE omopcdm.condition_occurrence
    ALTER COLUMN condition_occurrence_id TYPE BIGINT,
    ALTER COLUMN person_id               TYPE BIGINT,
    ALTER COLUMN provider_id             TYPE BIGINT,
    ALTER COLUMN visit_occurrence_id     TYPE BIGINT,
    ALTER COLUMN visit_detail_id         TYPE BIGINT;

ALTER TABLE omopcdm.procedure_occurrence
    ALTER COLUMN procedure_occurrence_id TYPE BIGINT,
    ALTER COLUMN person_id               TYPE BIGINT,
    ALTER COLUMN provider_id             TYPE BIGINT,
    ALTER COLUMN visit_occurrence_id     TYPE BIGINT,
    ALTER COLUMN visit_detail_id         TYPE BIGINT;

ALTER TABLE omopcdm.drug_exposure
    ALTER COLUMN drug_exposure_id    TYPE BIGINT,
    ALTER COLUMN person_id           TYPE BIGINT,
    ALTER COLUMN provider_id         TYPE BIGINT,
    ALTER COLUMN visit_occurrence_id TYPE BIGINT,
    ALTER COLUMN visit_detail_id     TYPE BIGINT;

ALTER TABLE omopcdm.measurement
    ALTER COLUMN measurement_id      TYPE BIGINT,
    ALTER COLUMN person_id           TYPE BIGINT,
    ALTER COLUMN provider_id         TYPE BIGINT,
    ALTER COLUMN visit_occurrence_id TYPE BIGINT,
    ALTER COLUMN visit_detail_id     TYPE BIGINT;

ALTER TABLE omopcdm.condition_era
    ALTER COLUMN condition_era_id TYPE BIGINT,
    ALTER COLUMN person_id        TYPE BIGINT;

ALTER TABLE omopcdm.drug_era
    ALTER COLUMN drug_era_id TYPE BIGINT,
    ALTER COLUMN person_id   TYPE BIGINT;

-- -----------------------------------------------------------------------
-- Widen VARCHAR(50) source-value columns that staging stores as TEXT
-- -----------------------------------------------------------------------

ALTER TABLE omopcdm.drug_exposure
    ALTER COLUMN drug_source_value      TYPE TEXT,
    ALTER COLUMN route_source_value     TYPE TEXT,
    ALTER COLUMN dose_unit_source_value TYPE TEXT;

ALTER TABLE omopcdm.measurement
    ALTER COLUMN measurement_source_value TYPE TEXT,
    ALTER COLUMN unit_source_value        TYPE TEXT,
    ALTER COLUMN value_source_value       TYPE TEXT;

-- -----------------------------------------------------------------------
-- Clinical tables (strip tracking columns)
-- -----------------------------------------------------------------------

TRUNCATE omopcdm.location;
INSERT INTO omopcdm.location
SELECT location_id, address_1, address_2, city, state, zip, county,
       location_source_value, NULL, NULL, NULL, NULL
FROM staging.cdm_location;

TRUNCATE omopcdm.care_site;
INSERT INTO omopcdm.care_site
SELECT care_site_id, care_site_name, place_of_service_concept_id,
       location_id, care_site_source_value, place_of_service_source_value
FROM staging.cdm_care_site;

TRUNCATE omopcdm.person;
INSERT INTO omopcdm.person
SELECT person_id, gender_concept_id, year_of_birth, month_of_birth, day_of_birth,
       birth_datetime, race_concept_id, ethnicity_concept_id, location_id,
       provider_id, care_site_id, person_source_value, gender_source_value,
       gender_source_concept_id, race_source_value, race_source_concept_id,
       ethnicity_source_value, ethnicity_source_concept_id
FROM staging.cdm_person;

TRUNCATE omopcdm.death;
INSERT INTO omopcdm.death
SELECT person_id, death_date, death_datetime, death_type_concept_id,
       cause_concept_id, cause_source_value, cause_source_concept_id
FROM staging.cdm_death;

TRUNCATE omopcdm.observation_period;
INSERT INTO omopcdm.observation_period
SELECT observation_period_id, person_id,
       observation_period_start_date, observation_period_end_date,
       period_type_concept_id
FROM staging.cdm_observation_period;

TRUNCATE omopcdm.visit_occurrence;
INSERT INTO omopcdm.visit_occurrence
SELECT visit_occurrence_id, person_id, visit_concept_id,
       visit_start_date, visit_start_datetime, visit_end_date, visit_end_datetime,
       visit_type_concept_id, provider_id, care_site_id,
       visit_source_value, visit_source_concept_id,
       admitted_from_concept_id, admitted_from_source_value,
       discharge_to_concept_id  AS discharged_to_concept_id,
       discharge_to_source_value AS discharged_to_source_value,
       preceding_visit_occurrence_id
FROM staging.cdm_visit_occurrence;

TRUNCATE omopcdm.visit_detail;
INSERT INTO omopcdm.visit_detail
SELECT visit_detail_id, person_id, visit_detail_concept_id,
       visit_detail_start_date, visit_detail_start_datetime,
       visit_detail_end_date, visit_detail_end_datetime,
       visit_detail_type_concept_id, provider_id, care_site_id,
       visit_detail_source_value, visit_detail_source_concept_id,
       admitted_from_concept_id, admitted_from_source_value,
       discharge_to_source_value AS discharged_to_source_value,
       discharge_to_concept_id   AS discharged_to_concept_id,
       preceding_visit_detail_id, visit_detail_parent_id,
       visit_occurrence_id
FROM staging.cdm_visit_detail;

TRUNCATE omopcdm.condition_occurrence;
INSERT INTO omopcdm.condition_occurrence
SELECT condition_occurrence_id, person_id, condition_concept_id,
       condition_start_date, condition_start_datetime,
       condition_end_date, condition_end_datetime,
       condition_type_concept_id, condition_status_concept_id,
       stop_reason, provider_id, visit_occurrence_id, visit_detail_id,
       condition_source_value, condition_source_concept_id,
       condition_status_source_value
FROM staging.cdm_condition_occurrence;

TRUNCATE omopcdm.procedure_occurrence;
INSERT INTO omopcdm.procedure_occurrence
SELECT procedure_occurrence_id, person_id, procedure_concept_id,
       procedure_date, procedure_datetime, procedure_end_date, procedure_end_datetime,
       procedure_type_concept_id, modifier_concept_id, quantity,
       provider_id, visit_occurrence_id, visit_detail_id,
       procedure_source_value, procedure_source_concept_id, modifier_source_value
FROM staging.cdm_procedure_occurrence;

TRUNCATE omopcdm.drug_exposure;
INSERT INTO omopcdm.drug_exposure
SELECT drug_exposure_id, person_id, drug_concept_id,
       drug_exposure_start_date, drug_exposure_start_datetime,
       drug_exposure_end_date, drug_exposure_end_datetime,
       verbatim_end_date, drug_type_concept_id, stop_reason,
       refills, quantity, days_supply, sig, route_concept_id,
       lot_number, provider_id, visit_occurrence_id, visit_detail_id,
       drug_source_value, drug_source_concept_id,
       route_source_value, dose_unit_source_value
FROM staging.cdm_drug_exposure;

TRUNCATE omopcdm.measurement;
INSERT INTO omopcdm.measurement
SELECT measurement_id, person_id, measurement_concept_id,
       measurement_date, measurement_datetime, measurement_time,
       measurement_type_concept_id, operator_concept_id,
       value_as_number, value_as_concept_id, unit_concept_id,
       range_low, range_high, provider_id,
       visit_occurrence_id, visit_detail_id,
       measurement_source_value, measurement_source_concept_id,
       unit_source_value, NULL, value_source_value, NULL, NULL
FROM staging.cdm_measurement;

TRUNCATE omopcdm.condition_era;
INSERT INTO omopcdm.condition_era
SELECT condition_era_id, person_id, condition_concept_id,
       condition_era_start_date, condition_era_end_date,
       condition_occurrence_count
FROM staging.cdm_condition_era;

TRUNCATE omopcdm.drug_era;
INSERT INTO omopcdm.drug_era
SELECT drug_era_id, person_id, drug_concept_id,
       drug_era_start_date, drug_era_end_date,
       drug_exposure_count, gap_days
FROM staging.cdm_drug_era;

TRUNCATE omopcdm.cdm_source;
INSERT INTO omopcdm.cdm_source
SELECT cdm_source_name, cdm_source_abbreviation, cdm_holder,
       source_description, source_documentation_reference,
       cdm_etl_reference, source_release_date, cdm_release_date,
       cdm_version, 756265 AS cdm_version_concept_id,   -- OMOP CDM v5.4
       COALESCE(vocabulary_version, 'unknown')
FROM staging.cdm_cdm_source
LIMIT 1;

-- -----------------------------------------------------------------------
-- Indexes on omopcdm for ATLAS query performance
-- -----------------------------------------------------------------------
CREATE INDEX IF NOT EXISTS idx_omop_co_person   ON omopcdm.condition_occurrence (person_id);
CREATE INDEX IF NOT EXISTS idx_omop_po_person   ON omopcdm.procedure_occurrence (person_id);
CREATE INDEX IF NOT EXISTS idx_omop_de_person   ON omopcdm.drug_exposure (person_id);
CREATE INDEX IF NOT EXISTS idx_omop_meas_person ON omopcdm.measurement (person_id);
CREATE INDEX IF NOT EXISTS idx_omop_vo_person   ON omopcdm.visit_occurrence (person_id);
CREATE INDEX IF NOT EXISTS idx_omop_vd_vo       ON omopcdm.visit_detail (visit_occurrence_id);
CREATE INDEX IF NOT EXISTS idx_omop_op_person   ON omopcdm.observation_period (person_id);
CREATE INDEX IF NOT EXISTS idx_omop_concept_id  ON omopcdm.concept (concept_id);
CREATE INDEX IF NOT EXISTS idx_omop_concept_code ON omopcdm.concept (concept_code, vocabulary_id);
CREATE INDEX IF NOT EXISTS idx_omop_cr_id1      ON omopcdm.concept_relationship (concept_id_1, relationship_id);

-- -----------------------------------------------------------------------
-- Row counts summary
-- -----------------------------------------------------------------------
SELECT 'person'               AS tbl, COUNT(*) AS n FROM omopcdm.person
UNION ALL SELECT 'observation_period',  COUNT(*) FROM omopcdm.observation_period
UNION ALL SELECT 'visit_occurrence',    COUNT(*) FROM omopcdm.visit_occurrence
UNION ALL SELECT 'visit_detail',        COUNT(*) FROM omopcdm.visit_detail
UNION ALL SELECT 'condition_occurrence',COUNT(*) FROM omopcdm.condition_occurrence
UNION ALL SELECT 'procedure_occurrence',COUNT(*) FROM omopcdm.procedure_occurrence
UNION ALL SELECT 'drug_exposure',       COUNT(*) FROM omopcdm.drug_exposure
UNION ALL SELECT 'measurement',         COUNT(*) FROM omopcdm.measurement
UNION ALL SELECT 'condition_era',       COUNT(*) FROM omopcdm.condition_era
UNION ALL SELECT 'drug_era',            COUNT(*) FROM omopcdm.drug_era
UNION ALL SELECT 'death',               COUNT(*) FROM omopcdm.death
UNION ALL SELECT 'concept',             COUNT(*) FROM omopcdm.concept
ORDER BY tbl;
