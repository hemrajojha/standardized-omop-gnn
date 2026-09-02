-- =======================================================================
-- 01_create_schemas.sql
-- Create all schemas for the OHDSI MIMIC-IV OMOP CDM pipeline.
--
-- Schemas
--   mimiciv_hosp  – raw MIMIC-IV hospital + core source tables
--   mimiciv_icu   – raw MIMIC-IV ICU source tables
--   vocab         – OHDSI Athena vocabulary (loaded once, read-only after)
--   staging       – intermediate ETL working area (src_*, lk_*, cdm_* with tracking cols)
--   omopcdm       – final OMOP CDM 5.4.1 tables (ATLAS-ready, clean)
--   results       – placeholder for ATLAS cohort result tables
-- =======================================================================

CREATE SCHEMA IF NOT EXISTS mimiciv_hosp;
CREATE SCHEMA IF NOT EXISTS mimiciv_icu;
CREATE SCHEMA IF NOT EXISTS vocab;
CREATE SCHEMA IF NOT EXISTS staging;
CREATE SCHEMA IF NOT EXISTS omopcdm;
CREATE SCHEMA IF NOT EXISTS results;

-- =======================================================================
-- Helper function: omop_id()
-- Replacement for BigQuery FARM_FINGERPRINT(GENERATE_UUID()).
-- Returns a reproducibly-random BIGINT suitable for surrogate OMOP IDs.
-- =======================================================================
CREATE OR REPLACE FUNCTION staging.omop_id()
RETURNS BIGINT
LANGUAGE sql VOLATILE
AS $$
    SELECT ('x' || substr(md5(random()::text || clock_timestamp()::text), 1, 15))::bit(60)::bigint;
$$;
