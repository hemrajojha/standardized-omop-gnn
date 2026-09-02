-- =======================================================================
-- 04_omop_cdm_tables.sql
-- OMOP CDM 5.4.1 tables in the `omopcdm` schema.
--
-- Source: sql/CommonDataModel-5.4.1/OMOPCDM_postgresql_5.4_ddl.sql
-- Schema renamed from omop_cdm → omopcdm (OHDSI / Broadsea convention).
-- Vocabulary tables included here so the omopcdm schema is fully
-- self-contained and ATLAS-ready after the unload step.
-- =======================================================================

-- -----------------------------------------------------------------------
-- Clinical tables
-- -----------------------------------------------------------------------

DROP TABLE IF EXISTS omopcdm.person CASCADE;
CREATE TABLE omopcdm.person (
    person_id                   INTEGER     NOT NULL,
    gender_concept_id           INTEGER     NOT NULL,
    year_of_birth               INTEGER     NOT NULL,
    month_of_birth              INTEGER,
    day_of_birth                INTEGER,
    birth_datetime              TIMESTAMP,
    race_concept_id             INTEGER     NOT NULL,
    ethnicity_concept_id        INTEGER     NOT NULL,
    location_id                 INTEGER,
    provider_id                 INTEGER,
    care_site_id                INTEGER,
    person_source_value         VARCHAR(50),
    gender_source_value         VARCHAR(50),
    gender_source_concept_id    INTEGER,
    race_source_value           VARCHAR(50),
    race_source_concept_id      INTEGER,
    ethnicity_source_value      VARCHAR(50),
    ethnicity_source_concept_id INTEGER
);

DROP TABLE IF EXISTS omopcdm.observation_period CASCADE;
CREATE TABLE omopcdm.observation_period (
    observation_period_id           INTEGER NOT NULL,
    person_id                       INTEGER NOT NULL,
    observation_period_start_date   DATE    NOT NULL,
    observation_period_end_date     DATE    NOT NULL,
    period_type_concept_id          INTEGER NOT NULL
);

DROP TABLE IF EXISTS omopcdm.visit_occurrence CASCADE;
CREATE TABLE omopcdm.visit_occurrence (
    visit_occurrence_id             INTEGER     NOT NULL,
    person_id                       INTEGER     NOT NULL,
    visit_concept_id                INTEGER     NOT NULL,
    visit_start_date                DATE        NOT NULL,
    visit_start_datetime            TIMESTAMP,
    visit_end_date                  DATE        NOT NULL,
    visit_end_datetime              TIMESTAMP,
    visit_type_concept_id           INTEGER     NOT NULL,
    provider_id                     INTEGER,
    care_site_id                    INTEGER,
    visit_source_value              VARCHAR(50),
    visit_source_concept_id         INTEGER,
    admitted_from_concept_id        INTEGER,
    admitted_from_source_value      VARCHAR(50),
    discharged_to_concept_id        INTEGER,
    discharged_to_source_value      VARCHAR(50),
    preceding_visit_occurrence_id   INTEGER
);

DROP TABLE IF EXISTS omopcdm.visit_detail CASCADE;
CREATE TABLE omopcdm.visit_detail (
    visit_detail_id                 INTEGER     NOT NULL,
    person_id                       INTEGER     NOT NULL,
    visit_detail_concept_id         INTEGER     NOT NULL,
    visit_detail_start_date         DATE        NOT NULL,
    visit_detail_start_datetime     TIMESTAMP,
    visit_detail_end_date           DATE        NOT NULL,
    visit_detail_end_datetime       TIMESTAMP,
    visit_detail_type_concept_id    INTEGER     NOT NULL,
    provider_id                     INTEGER,
    care_site_id                    INTEGER,
    visit_detail_source_value       VARCHAR(50),
    visit_detail_source_concept_id  INTEGER,
    admitted_from_concept_id        INTEGER,
    admitted_from_source_value      VARCHAR(50),
    discharged_to_source_value      VARCHAR(50),
    discharged_to_concept_id        INTEGER,
    preceding_visit_detail_id       INTEGER,
    parent_visit_detail_id          INTEGER,
    visit_occurrence_id             INTEGER     NOT NULL
);

DROP TABLE IF EXISTS omopcdm.condition_occurrence CASCADE;
CREATE TABLE omopcdm.condition_occurrence (
    condition_occurrence_id         INTEGER     NOT NULL,
    person_id                       INTEGER     NOT NULL,
    condition_concept_id            INTEGER     NOT NULL,
    condition_start_date            DATE        NOT NULL,
    condition_start_datetime        TIMESTAMP,
    condition_end_date              DATE,
    condition_end_datetime          TIMESTAMP,
    condition_type_concept_id       INTEGER     NOT NULL,
    condition_status_concept_id     INTEGER,
    stop_reason                     VARCHAR(20),
    provider_id                     INTEGER,
    visit_occurrence_id             INTEGER,
    visit_detail_id                 INTEGER,
    condition_source_value          VARCHAR(50),
    condition_source_concept_id     INTEGER,
    condition_status_source_value   VARCHAR(50)
);

DROP TABLE IF EXISTS omopcdm.drug_exposure CASCADE;
CREATE TABLE omopcdm.drug_exposure (
    drug_exposure_id                INTEGER     NOT NULL,
    person_id                       INTEGER     NOT NULL,
    drug_concept_id                 INTEGER     NOT NULL,
    drug_exposure_start_date        DATE        NOT NULL,
    drug_exposure_start_datetime    TIMESTAMP,
    drug_exposure_end_date          DATE        NOT NULL,
    drug_exposure_end_datetime      TIMESTAMP,
    verbatim_end_date               DATE,
    drug_type_concept_id            INTEGER     NOT NULL,
    stop_reason                     VARCHAR(20),
    refills                         INTEGER,
    quantity                        NUMERIC,
    days_supply                     INTEGER,
    sig                             TEXT,
    route_concept_id                INTEGER,
    lot_number                      VARCHAR(50),
    provider_id                     INTEGER,
    visit_occurrence_id             INTEGER,
    visit_detail_id                 INTEGER,
    drug_source_value               VARCHAR(50),
    drug_source_concept_id          INTEGER,
    route_source_value              VARCHAR(50),
    dose_unit_source_value          VARCHAR(50)
);

DROP TABLE IF EXISTS omopcdm.procedure_occurrence CASCADE;
CREATE TABLE omopcdm.procedure_occurrence (
    procedure_occurrence_id     INTEGER     NOT NULL,
    person_id                   INTEGER     NOT NULL,
    procedure_concept_id        INTEGER     NOT NULL,
    procedure_date              DATE        NOT NULL,
    procedure_datetime          TIMESTAMP,
    procedure_end_date          DATE,
    procedure_end_datetime      TIMESTAMP,
    procedure_type_concept_id   INTEGER     NOT NULL,
    modifier_concept_id         INTEGER,
    quantity                    INTEGER,
    provider_id                 INTEGER,
    visit_occurrence_id         INTEGER,
    visit_detail_id             INTEGER,
    procedure_source_value      VARCHAR(50),
    procedure_source_concept_id INTEGER,
    modifier_source_value       VARCHAR(50)
);

DROP TABLE IF EXISTS omopcdm.device_exposure CASCADE;
CREATE TABLE omopcdm.device_exposure (
    device_exposure_id              INTEGER     NOT NULL,
    person_id                       INTEGER     NOT NULL,
    device_concept_id               INTEGER     NOT NULL,
    device_exposure_start_date      DATE        NOT NULL,
    device_exposure_start_datetime  TIMESTAMP,
    device_exposure_end_date        DATE,
    device_exposure_end_datetime    TIMESTAMP,
    device_type_concept_id          INTEGER     NOT NULL,
    unique_device_id                VARCHAR(255),
    production_id                   VARCHAR(255),
    quantity                        INTEGER,
    provider_id                     INTEGER,
    visit_occurrence_id             INTEGER,
    visit_detail_id                 INTEGER,
    device_source_value             VARCHAR(50),
    device_source_concept_id        INTEGER,
    unit_concept_id                 INTEGER,
    unit_source_value               VARCHAR(50),
    unit_source_concept_id          INTEGER
);

DROP TABLE IF EXISTS omopcdm.measurement CASCADE;
CREATE TABLE omopcdm.measurement (
    measurement_id                  INTEGER     NOT NULL,
    person_id                       INTEGER     NOT NULL,
    measurement_concept_id          INTEGER     NOT NULL,
    measurement_date                DATE        NOT NULL,
    measurement_datetime            TIMESTAMP,
    measurement_time                VARCHAR(10),
    measurement_type_concept_id     INTEGER     NOT NULL,
    operator_concept_id             INTEGER,
    value_as_number                 NUMERIC,
    value_as_concept_id             INTEGER,
    unit_concept_id                 INTEGER,
    range_low                       NUMERIC,
    range_high                      NUMERIC,
    provider_id                     INTEGER,
    visit_occurrence_id             INTEGER,
    visit_detail_id                 INTEGER,
    measurement_source_value        VARCHAR(50),
    measurement_source_concept_id   INTEGER,
    unit_source_value               VARCHAR(50),
    unit_source_concept_id          INTEGER,
    value_source_value              VARCHAR(50),
    measurement_event_id            INTEGER,
    meas_event_field_concept_id     INTEGER
);

DROP TABLE IF EXISTS omopcdm.observation CASCADE;
CREATE TABLE omopcdm.observation (
    observation_id                  INTEGER     NOT NULL,
    person_id                       INTEGER     NOT NULL,
    observation_concept_id          INTEGER     NOT NULL,
    observation_date                DATE        NOT NULL,
    observation_datetime            TIMESTAMP,
    observation_type_concept_id     INTEGER     NOT NULL,
    value_as_number                 NUMERIC,
    value_as_string                 VARCHAR(60),
    value_as_concept_id             INTEGER,
    qualifier_concept_id            INTEGER,
    unit_concept_id                 INTEGER,
    provider_id                     INTEGER,
    visit_occurrence_id             INTEGER,
    visit_detail_id                 INTEGER,
    observation_source_value        VARCHAR(50),
    observation_source_concept_id   INTEGER,
    unit_source_value               VARCHAR(50),
    qualifier_source_value          VARCHAR(50),
    value_source_value              VARCHAR(50),
    observation_event_id            INTEGER,
    obs_event_field_concept_id      INTEGER
);

DROP TABLE IF EXISTS omopcdm.death CASCADE;
CREATE TABLE omopcdm.death (
    person_id               INTEGER NOT NULL,
    death_date              DATE    NOT NULL,
    death_datetime          TIMESTAMP,
    death_type_concept_id   INTEGER,
    cause_concept_id        INTEGER,
    cause_source_value      VARCHAR(50),
    cause_source_concept_id INTEGER
);

DROP TABLE IF EXISTS omopcdm.note CASCADE;
CREATE TABLE omopcdm.note (
    note_id                     INTEGER     NOT NULL,
    person_id                   INTEGER     NOT NULL,
    note_date                   DATE        NOT NULL,
    note_datetime               TIMESTAMP,
    note_type_concept_id        INTEGER     NOT NULL,
    note_class_concept_id       INTEGER     NOT NULL,
    note_title                  VARCHAR(250),
    note_text                   TEXT        NOT NULL,
    encoding_concept_id         INTEGER     NOT NULL,
    language_concept_id         INTEGER     NOT NULL,
    provider_id                 INTEGER,
    visit_occurrence_id         INTEGER,
    visit_detail_id             INTEGER,
    note_source_value           VARCHAR(50),
    note_event_id               INTEGER,
    note_event_field_concept_id INTEGER
);

DROP TABLE IF EXISTS omopcdm.specimen CASCADE;
CREATE TABLE omopcdm.specimen (
    specimen_id                 INTEGER     NOT NULL,
    person_id                   INTEGER     NOT NULL,
    specimen_concept_id         INTEGER     NOT NULL,
    specimen_type_concept_id    INTEGER     NOT NULL,
    specimen_date               DATE        NOT NULL,
    specimen_datetime           TIMESTAMP,
    quantity                    NUMERIC,
    unit_concept_id             INTEGER,
    anatomic_site_concept_id    INTEGER,
    disease_status_concept_id   INTEGER,
    specimen_source_id          VARCHAR(50),
    specimen_source_value       VARCHAR(50),
    unit_source_value           VARCHAR(50),
    anatomic_site_source_value  VARCHAR(50),
    disease_status_source_value VARCHAR(50)
);

DROP TABLE IF EXISTS omopcdm.fact_relationship CASCADE;
CREATE TABLE omopcdm.fact_relationship (
    domain_concept_id_1     INTEGER NOT NULL,
    fact_id_1               INTEGER NOT NULL,
    domain_concept_id_2     INTEGER NOT NULL,
    fact_id_2               INTEGER NOT NULL,
    relationship_concept_id INTEGER NOT NULL
);

-- -----------------------------------------------------------------------
-- Health system tables
-- -----------------------------------------------------------------------

DROP TABLE IF EXISTS omopcdm.location CASCADE;
CREATE TABLE omopcdm.location (
    location_id             INTEGER     NOT NULL,
    address_1               VARCHAR(50),
    address_2               VARCHAR(50),
    city                    VARCHAR(50),
    state                   VARCHAR(2),
    zip                     VARCHAR(9),
    county                  VARCHAR(20),
    location_source_value   VARCHAR(50),
    country_concept_id      INTEGER,
    country_source_value    VARCHAR(80),
    latitude                NUMERIC,
    longitude               NUMERIC
);

DROP TABLE IF EXISTS omopcdm.care_site CASCADE;
CREATE TABLE omopcdm.care_site (
    care_site_id                  INTEGER     NOT NULL,
    care_site_name                VARCHAR(255),
    place_of_service_concept_id   INTEGER,
    location_id                   INTEGER,
    care_site_source_value        VARCHAR(50),
    place_of_service_source_value VARCHAR(50)
);

DROP TABLE IF EXISTS omopcdm.provider CASCADE;
CREATE TABLE omopcdm.provider (
    provider_id                 INTEGER     NOT NULL,
    provider_name               VARCHAR(255),
    npi                         VARCHAR(20),
    dea                         VARCHAR(20),
    specialty_concept_id        INTEGER,
    care_site_id                INTEGER,
    year_of_birth               INTEGER,
    gender_concept_id           INTEGER,
    provider_source_value       VARCHAR(50),
    specialty_source_value      VARCHAR(50),
    specialty_source_concept_id INTEGER,
    gender_source_value         VARCHAR(50),
    gender_source_concept_id    INTEGER
);

-- -----------------------------------------------------------------------
-- Derived tables
-- -----------------------------------------------------------------------

DROP TABLE IF EXISTS omopcdm.drug_era CASCADE;
CREATE TABLE omopcdm.drug_era (
    drug_era_id             INTEGER NOT NULL,
    person_id               INTEGER NOT NULL,
    drug_concept_id         INTEGER NOT NULL,
    drug_era_start_date     DATE    NOT NULL,
    drug_era_end_date       DATE    NOT NULL,
    drug_exposure_count     INTEGER,
    gap_days                INTEGER
);

DROP TABLE IF EXISTS omopcdm.dose_era CASCADE;
CREATE TABLE omopcdm.dose_era (
    dose_era_id             INTEGER NOT NULL,
    person_id               INTEGER NOT NULL,
    drug_concept_id         INTEGER NOT NULL,
    unit_concept_id         INTEGER NOT NULL,
    dose_value              NUMERIC NOT NULL,
    dose_era_start_date     DATE    NOT NULL,
    dose_era_end_date       DATE    NOT NULL
);

DROP TABLE IF EXISTS omopcdm.condition_era CASCADE;
CREATE TABLE omopcdm.condition_era (
    condition_era_id            INTEGER NOT NULL,
    person_id                   INTEGER NOT NULL,
    condition_concept_id        INTEGER NOT NULL,
    condition_era_start_date    DATE    NOT NULL,
    condition_era_end_date      DATE    NOT NULL,
    condition_occurrence_count  INTEGER
);

-- -----------------------------------------------------------------------
-- Metadata / CDM source
-- -----------------------------------------------------------------------

DROP TABLE IF EXISTS omopcdm.cdm_source CASCADE;
CREATE TABLE omopcdm.cdm_source (
    cdm_source_name                 VARCHAR(255) NOT NULL,
    cdm_source_abbreviation         VARCHAR(25)  NOT NULL,
    cdm_holder                      VARCHAR(255) NOT NULL,
    source_description              TEXT,
    source_documentation_reference  VARCHAR(255),
    cdm_etl_reference               VARCHAR(255),
    source_release_date             DATE         NOT NULL,
    cdm_release_date                DATE         NOT NULL,
    cdm_version                     VARCHAR(10),
    cdm_version_concept_id          INTEGER      NOT NULL,
    vocabulary_version              VARCHAR(20)  NOT NULL
);

-- -----------------------------------------------------------------------
-- Vocabulary tables (mirrored in omopcdm for ATLAS self-contained access)
-- -----------------------------------------------------------------------

DROP TABLE IF EXISTS omopcdm.concept CASCADE;
CREATE TABLE omopcdm.concept (
    concept_id          INTEGER      NOT NULL,
    concept_name        VARCHAR(255) NOT NULL,
    domain_id           VARCHAR(20)  NOT NULL,
    vocabulary_id       VARCHAR(20)  NOT NULL,
    concept_class_id    VARCHAR(20)  NOT NULL,
    standard_concept    VARCHAR(1),
    concept_code        VARCHAR(50)  NOT NULL,
    valid_start_date    DATE         NOT NULL,
    valid_end_date      DATE         NOT NULL,
    invalid_reason      VARCHAR(1)
);

DROP TABLE IF EXISTS omopcdm.vocabulary CASCADE;
CREATE TABLE omopcdm.vocabulary (
    vocabulary_id           VARCHAR(20)  NOT NULL,
    vocabulary_name         VARCHAR(255) NOT NULL,
    vocabulary_reference    VARCHAR(255),
    vocabulary_version      VARCHAR(255),
    vocabulary_concept_id   INTEGER      NOT NULL
);

DROP TABLE IF EXISTS omopcdm.domain CASCADE;
CREATE TABLE omopcdm.domain (
    domain_id           VARCHAR(20)  NOT NULL,
    domain_name         VARCHAR(255) NOT NULL,
    domain_concept_id   INTEGER      NOT NULL
);

DROP TABLE IF EXISTS omopcdm.concept_class CASCADE;
CREATE TABLE omopcdm.concept_class (
    concept_class_id            VARCHAR(20)  NOT NULL,
    concept_class_name          VARCHAR(255) NOT NULL,
    concept_class_concept_id    INTEGER      NOT NULL
);

DROP TABLE IF EXISTS omopcdm.concept_relationship CASCADE;
CREATE TABLE omopcdm.concept_relationship (
    concept_id_1        INTEGER     NOT NULL,
    concept_id_2        INTEGER     NOT NULL,
    relationship_id     VARCHAR(20) NOT NULL,
    valid_start_date    DATE        NOT NULL,
    valid_end_date      DATE        NOT NULL,
    invalid_reason      VARCHAR(1)
);

DROP TABLE IF EXISTS omopcdm.relationship CASCADE;
CREATE TABLE omopcdm.relationship (
    relationship_id             VARCHAR(20)  NOT NULL,
    relationship_name           VARCHAR(255) NOT NULL,
    is_hierarchical             VARCHAR(1)   NOT NULL,
    defines_ancestry            VARCHAR(1)   NOT NULL,
    reverse_relationship_id     VARCHAR(20)  NOT NULL,
    relationship_concept_id     INTEGER      NOT NULL
);

DROP TABLE IF EXISTS omopcdm.concept_synonym CASCADE;
CREATE TABLE omopcdm.concept_synonym (
    concept_id              INTEGER       NOT NULL,
    concept_synonym_name    VARCHAR(1000) NOT NULL,
    language_concept_id     INTEGER       NOT NULL
);

DROP TABLE IF EXISTS omopcdm.concept_ancestor CASCADE;
CREATE TABLE omopcdm.concept_ancestor (
    ancestor_concept_id         INTEGER NOT NULL,
    descendant_concept_id       INTEGER NOT NULL,
    min_levels_of_separation    INTEGER NOT NULL,
    max_levels_of_separation    INTEGER NOT NULL
);

DROP TABLE IF EXISTS omopcdm.source_to_concept_map CASCADE;
CREATE TABLE omopcdm.source_to_concept_map (
    source_code             VARCHAR(50)  NOT NULL,
    source_concept_id       INTEGER      NOT NULL,
    source_vocabulary_id    VARCHAR(20)  NOT NULL,
    source_code_description VARCHAR(255),
    target_concept_id       INTEGER      NOT NULL,
    target_vocabulary_id    VARCHAR(20)  NOT NULL,
    valid_start_date        DATE         NOT NULL,
    valid_end_date          DATE         NOT NULL,
    invalid_reason          VARCHAR(1)
);

DROP TABLE IF EXISTS omopcdm.drug_strength CASCADE;
CREATE TABLE omopcdm.drug_strength (
    drug_concept_id             INTEGER NOT NULL,
    ingredient_concept_id       INTEGER NOT NULL,
    amount_value                NUMERIC,
    amount_unit_concept_id      INTEGER,
    numerator_value             NUMERIC,
    numerator_unit_concept_id   INTEGER,
    denominator_value           NUMERIC,
    denominator_unit_concept_id INTEGER,
    box_size                    INTEGER,
    valid_start_date            DATE    NOT NULL,
    valid_end_date              DATE    NOT NULL,
    invalid_reason              VARCHAR(1)
);

-- ATLAS result tables (empty, populated by WebAPI)
DROP TABLE IF EXISTS omopcdm.cohort CASCADE;
CREATE TABLE omopcdm.cohort (
    cohort_definition_id    INTEGER NOT NULL,
    subject_id              INTEGER NOT NULL,
    cohort_start_date       DATE    NOT NULL,
    cohort_end_date         DATE    NOT NULL
);

DROP TABLE IF EXISTS omopcdm.cohort_definition CASCADE;
CREATE TABLE omopcdm.cohort_definition (
    cohort_definition_id            INTEGER     NOT NULL,
    cohort_definition_name          VARCHAR(255) NOT NULL,
    cohort_definition_description   TEXT,
    definition_type_concept_id      INTEGER     NOT NULL,
    cohort_definition_syntax        TEXT,
    subject_concept_id              INTEGER     NOT NULL,
    cohort_initiation_date          DATE
);
