-- =======================================================================
-- 03_vocab_tables.sql
-- DDL for OHDSI Athena vocabulary tables in the `vocab` schema.
-- Data is loaded via COPY in staging/01_load_vocab.sql.
--
-- These tables mirror the standard OMOP CDM vocabulary tables.
-- The staging ETL references them as vocab.* (aliased as voc_* in
-- staging intermediate joins for naming consistency with OHDSI MIMIC).
-- =======================================================================

DROP TABLE IF EXISTS vocab.concept CASCADE;
CREATE TABLE vocab.concept (
    concept_id          INTEGER     NOT NULL,
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

DROP TABLE IF EXISTS vocab.vocabulary CASCADE;
CREATE TABLE vocab.vocabulary (
    vocabulary_id           VARCHAR(20)  NOT NULL,
    vocabulary_name         VARCHAR(255) NOT NULL,
    vocabulary_reference    VARCHAR(255),
    vocabulary_version      VARCHAR(255),
    vocabulary_concept_id   INTEGER      NOT NULL
);

DROP TABLE IF EXISTS vocab.domain CASCADE;
CREATE TABLE vocab.domain (
    domain_id           VARCHAR(20)  NOT NULL,
    domain_name         VARCHAR(255) NOT NULL,
    domain_concept_id   INTEGER      NOT NULL
);

DROP TABLE IF EXISTS vocab.concept_class CASCADE;
CREATE TABLE vocab.concept_class (
    concept_class_id            VARCHAR(20)  NOT NULL,
    concept_class_name          VARCHAR(255) NOT NULL,
    concept_class_concept_id    INTEGER      NOT NULL
);

DROP TABLE IF EXISTS vocab.concept_relationship CASCADE;
CREATE TABLE vocab.concept_relationship (
    concept_id_1        INTEGER     NOT NULL,
    concept_id_2        INTEGER     NOT NULL,
    relationship_id     VARCHAR(20) NOT NULL,
    valid_start_date    DATE        NOT NULL,
    valid_end_date      DATE        NOT NULL,
    invalid_reason      VARCHAR(1)
);

DROP TABLE IF EXISTS vocab.relationship CASCADE;
CREATE TABLE vocab.relationship (
    relationship_id             VARCHAR(20)  NOT NULL,
    relationship_name           VARCHAR(255) NOT NULL,
    is_hierarchical             VARCHAR(1)   NOT NULL,
    defines_ancestry            VARCHAR(1)   NOT NULL,
    reverse_relationship_id     VARCHAR(20)  NOT NULL,
    relationship_concept_id     INTEGER      NOT NULL
);

DROP TABLE IF EXISTS vocab.concept_synonym CASCADE;
CREATE TABLE vocab.concept_synonym (
    concept_id              INTEGER      NOT NULL,
    concept_synonym_name    VARCHAR(1000) NOT NULL,
    language_concept_id     INTEGER      NOT NULL
);

DROP TABLE IF EXISTS vocab.concept_ancestor CASCADE;
CREATE TABLE vocab.concept_ancestor (
    ancestor_concept_id         INTEGER NOT NULL,
    descendant_concept_id       INTEGER NOT NULL,
    min_levels_of_separation    INTEGER NOT NULL,
    max_levels_of_separation    INTEGER NOT NULL
);

DROP TABLE IF EXISTS vocab.source_to_concept_map CASCADE;
CREATE TABLE vocab.source_to_concept_map (
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

DROP TABLE IF EXISTS vocab.drug_strength CASCADE;
CREATE TABLE vocab.drug_strength (
    drug_concept_id             INTEGER     NOT NULL,
    ingredient_concept_id       INTEGER     NOT NULL,
    amount_value                NUMERIC,
    amount_unit_concept_id      INTEGER,
    numerator_value             NUMERIC,
    numerator_unit_concept_id   INTEGER,
    denominator_value           NUMERIC,
    denominator_unit_concept_id INTEGER,
    box_size                    INTEGER,
    valid_start_date            DATE        NOT NULL,
    valid_end_date              DATE        NOT NULL,
    invalid_reason              VARCHAR(1)
);
