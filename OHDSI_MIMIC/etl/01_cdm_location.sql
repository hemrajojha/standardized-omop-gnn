-- =======================================================================
-- 01_cdm_location.sql
-- Populate staging.cdm_location
-- Beth Israel Deaconess Medical Center, Boston MA (single location)
--
-- Adapted from: https://github.com/OHDSI/MIMIC/blob/main/etl/etl/cdm_care_site.sql
-- =======================================================================

DROP TABLE IF EXISTS staging.cdm_location;
CREATE TABLE staging.cdm_location (
    location_id             BIGINT,
    address_1               TEXT,
    address_2               TEXT,
    city                    TEXT,
    state                   TEXT,
    zip                     TEXT,
    county                  TEXT,
    location_source_value   TEXT,
    -- tracking cols
    unit_id                 TEXT,
    load_table_id           TEXT,
    load_row_id             BIGINT,
    trace_id                TEXT
);

INSERT INTO staging.cdm_location
VALUES (
    1,
    '330 Brookline Ave',
    NULL,
    'Boston',
    'MA',
    '02215',
    'Suffolk',
    'Beth Israel Deaconess Medical Center',
    'location.null',
    'null',
    0,
    NULL
);
