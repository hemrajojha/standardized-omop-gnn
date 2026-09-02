-- Missing OMOP CDM 5.4 tables not populated by MIMIC ETL
-- Created as empty stubs so Achilles can query them without error

CREATE TABLE IF NOT EXISTS omopcdm.cost (
    cost_id                     BIGINT      NOT NULL,
    cost_event_id               BIGINT      NOT NULL,
    cost_domain_id              VARCHAR(20) NOT NULL,
    cost_type_concept_id        INTEGER     NOT NULL,
    currency_concept_id         INTEGER,
    total_charge                NUMERIC,
    total_cost                  NUMERIC,
    total_paid                  NUMERIC,
    paid_by_payer               NUMERIC,
    paid_by_patient             NUMERIC,
    paid_patient_copay          NUMERIC,
    paid_patient_coinsurance    NUMERIC,
    paid_patient_deductible     NUMERIC,
    paid_by_primary             NUMERIC,
    paid_ingredient_cost        NUMERIC,
    paid_dispensing_fee         NUMERIC,
    payer_plan_period_id        BIGINT,
    amount_allowed              NUMERIC,
    revenue_code_concept_id     INTEGER,
    revenue_code_source_value   VARCHAR(50),
    drg_concept_id              INTEGER,
    drg_source_value            VARCHAR(3)
);

CREATE TABLE IF NOT EXISTS omopcdm.note_nlp (
    note_nlp_id                 BIGINT       NOT NULL,
    note_id                     BIGINT       NOT NULL,
    section_concept_id          INTEGER,
    snippet                     VARCHAR(250),
    "offset"                    VARCHAR(50),
    lexical_variant             VARCHAR(250) NOT NULL,
    note_nlp_concept_id         INTEGER,
    note_nlp_source_concept_id  INTEGER,
    nlp_system                  VARCHAR(250),
    nlp_date                    DATE         NOT NULL,
    nlp_datetime                TIMESTAMP,
    term_exists                 VARCHAR(1),
    term_temporal               VARCHAR(50),
    term_modifiers              VARCHAR(2000)
);

CREATE TABLE IF NOT EXISTS omopcdm.survey_conduct (
    survey_conduct_id               BIGINT  NOT NULL,
    person_id                       BIGINT  NOT NULL,
    survey_concept_id               INTEGER NOT NULL,
    survey_start_date               DATE,
    survey_start_datetime           TIMESTAMP,
    survey_end_date                 DATE,
    survey_end_datetime             TIMESTAMP,
    provider_id                     BIGINT,
    assisted_concept_id             INTEGER,
    respondent_type_concept_id      INTEGER,
    timing_concept_id               INTEGER,
    collection_method_concept_id    INTEGER,
    assisted_source_value           VARCHAR(50),
    respondent_type_source_value    VARCHAR(100),
    timing_source_value             VARCHAR(100),
    collection_method_source_value  VARCHAR(100),
    survey_source_value             VARCHAR(100),
    survey_source_concept_id        INTEGER,
    survey_source_identifier        VARCHAR(100),
    validated_survey_concept_id     INTEGER,
    validated_survey_source_value   VARCHAR(100),
    survey_version_number           VARCHAR(20),
    visit_occurrence_id             BIGINT,
    visit_detail_id                 BIGINT,
    response_visit_occurrence_id    BIGINT
);
