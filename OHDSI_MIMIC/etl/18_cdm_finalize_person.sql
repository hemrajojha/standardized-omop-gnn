-- =======================================================================
-- 18_cdm_finalize_person.sql
-- Remove persons with no observation period records (DQD requirement).
--
-- Adapted from: https://github.com/OHDSI/MIMIC/blob/main/etl/etl/cdm_finalize_person.sql
-- =======================================================================

CREATE TEMP TABLE tmp_person_keep AS
SELECT per.*
FROM
    staging.cdm_person per
INNER JOIN
    staging.cdm_observation_period op
        ON per.person_id = op.person_id
;

TRUNCATE staging.cdm_person;

INSERT INTO staging.cdm_person
SELECT * FROM tmp_person_keep;

DROP TABLE tmp_person_keep;
