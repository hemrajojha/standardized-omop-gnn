-- =======================================================================
-- custom_mapping.sql
-- MIMIC-IV-specific concept mappings not found in standard Athena vocab.
-- Run AFTER staging/01_load_vocab.sql and BEFORE staging/02_st_core.sql.
-- =======================================================================

-- -----------------------------------------------------------------------
-- Admission types (MIMIC admission_type → OMOP Visit concepts)
-- -----------------------------------------------------------------------
INSERT INTO vocab.source_to_concept_map
    (source_code, source_concept_id, source_vocabulary_id, source_code_description,
     target_concept_id, target_vocabulary_id, valid_start_date, valid_end_date, invalid_reason)
SELECT v.source_code, v.source_concept_id, v.source_vocabulary_id, v.source_code_description,
       v.target_concept_id, v.target_vocabulary_id, v.valid_start_date, v.valid_end_date, v.invalid_reason
FROM (VALUES
    ('EMERGENCY',                   0, 'mimiciv_visit_type', 'Emergency visit',      9203, 'Visit', '1970-01-01'::date, '2099-12-31'::date, NULL::varchar),
    ('ELECTIVE',                    0, 'mimiciv_visit_type', 'Elective visit',        9201, 'Visit', '1970-01-01', '2099-12-31', NULL),
    ('URGENT',                      0, 'mimiciv_visit_type', 'Urgent visit',          9203, 'Visit', '1970-01-01', '2099-12-31', NULL),
    ('OBSERVATION',                 0, 'mimiciv_visit_type', 'Observation',           9201, 'Visit', '1970-01-01', '2099-12-31', NULL),
    ('DIRECT EMER.',                0, 'mimiciv_visit_type', 'Direct emergency',      9203, 'Visit', '1970-01-01', '2099-12-31', NULL),
    ('DIRECT OBSERVATION',          0, 'mimiciv_visit_type', 'Direct observation',    9201, 'Visit', '1970-01-01', '2099-12-31', NULL),
    ('EU OBSERVATION',              0, 'mimiciv_visit_type', 'EU Observation',        9201, 'Visit', '1970-01-01', '2099-12-31', NULL),
    ('SURGICAL SAME DAY ADMISSION', 0, 'mimiciv_visit_type', 'Same day surgery',      9201, 'Visit', '1970-01-01', '2099-12-31', NULL)
) AS v(source_code, source_concept_id, source_vocabulary_id, source_code_description,
       target_concept_id, target_vocabulary_id, valid_start_date, valid_end_date, invalid_reason)
WHERE NOT EXISTS (
    SELECT 1 FROM vocab.source_to_concept_map x
    WHERE x.source_code = v.source_code AND x.source_vocabulary_id = v.source_vocabulary_id
);

-- -----------------------------------------------------------------------
-- Race / Ethnicity custom concepts
-- -----------------------------------------------------------------------
INSERT INTO vocab.concept
    (concept_id, concept_name, domain_id, vocabulary_id, concept_class_id,
     standard_concept, concept_code, valid_start_date, valid_end_date, invalid_reason)
SELECT v.concept_id, v.concept_name, v.domain_id, v.vocabulary_id, v.concept_class_id,
       v.standard_concept, v.concept_code, v.valid_start_date, v.valid_end_date, v.invalid_reason
FROM (VALUES
    (2000000001, 'WHITE',                        'Race',      'mimiciv_ethnicity', 'Race',      'S', 'WHITE',                        '1970-01-01'::date, '2099-12-31'::date, NULL::varchar),
    (2000000002, 'BLACK/AFRICAN AMERICAN',       'Race',      'mimiciv_ethnicity', 'Race',      'S', 'BLACK/AFRICAN AMERICAN',       '1970-01-01', '2099-12-31', NULL),
    (2000000003, 'HISPANIC OR LATINO',           'Ethnicity', 'mimiciv_ethnicity', 'Ethnicity', 'S', 'HISPANIC OR LATINO',           '1970-01-01', '2099-12-31', NULL),
    (2000000004, 'ASIAN',                        'Race',      'mimiciv_ethnicity', 'Race',      'S', 'ASIAN',                        '1970-01-01', '2099-12-31', NULL),
    (2000000005, 'OTHER',                        'Race',      'mimiciv_ethnicity', 'Race',      'S', 'OTHER',                        '1970-01-01', '2099-12-31', NULL),
    (2000000006, 'UNKNOWN',                      'Race',      'mimiciv_ethnicity', 'Race',      'S', 'UNKNOWN',                      '1970-01-01', '2099-12-31', NULL),
    (2000000007, 'UNABLE TO OBTAIN',             'Race',      'mimiciv_ethnicity', 'Race',      'S', 'UNABLE TO OBTAIN',             '1970-01-01', '2099-12-31', NULL),
    (2000000008, 'PATIENT DECLINED TO ANSWER',   'Race',      'mimiciv_ethnicity', 'Race',      'S', 'PATIENT DECLINED TO ANSWER',   '1970-01-01', '2099-12-31', NULL),
    (2000000009, 'AMERICAN INDIAN/ALASKA NATIVE','Race',      'mimiciv_ethnicity', 'Race',      'S', 'AMERICAN INDIAN/ALASKA NATIVE','1970-01-01', '2099-12-31', NULL),
    (2000000010, 'WHITE - OTHER EUROPEAN',       'Race',      'mimiciv_ethnicity', 'Race',      'S', 'WHITE - OTHER EUROPEAN',       '1970-01-01', '2099-12-31', NULL)
) AS v(concept_id, concept_name, domain_id, vocabulary_id, concept_class_id,
       standard_concept, concept_code, valid_start_date, valid_end_date, invalid_reason)
WHERE NOT EXISTS (
    SELECT 1 FROM vocab.concept x WHERE x.concept_id = v.concept_id
);

-- -----------------------------------------------------------------------
-- Map custom race concepts → OMOP standard concepts
-- -----------------------------------------------------------------------
INSERT INTO vocab.concept_relationship
    (concept_id_1, concept_id_2, relationship_id, valid_start_date, valid_end_date, invalid_reason)
SELECT v.concept_id_1, v.concept_id_2, v.relationship_id, v.valid_start_date, v.valid_end_date, v.invalid_reason
FROM (VALUES
    (2000000001, 8527,     'Maps to', '1970-01-01'::date, '2099-12-31'::date, NULL::varchar),
    (2000000002, 8516,     'Maps to', '1970-01-01', '2099-12-31', NULL),
    (2000000003, 38003563, 'Maps to', '1970-01-01', '2099-12-31', NULL),
    (2000000004, 8515,     'Maps to', '1970-01-01', '2099-12-31', NULL),
    (2000000005, 8522,     'Maps to', '1970-01-01', '2099-12-31', NULL),
    (2000000006, 0,        'Maps to', '1970-01-01', '2099-12-31', NULL),
    (2000000007, 0,        'Maps to', '1970-01-01', '2099-12-31', NULL),
    (2000000008, 0,        'Maps to', '1970-01-01', '2099-12-31', NULL),
    (2000000009, 8657,     'Maps to', '1970-01-01', '2099-12-31', NULL),
    (2000000010, 8527,     'Maps to', '1970-01-01', '2099-12-31', NULL)
) AS v(concept_id_1, concept_id_2, relationship_id, valid_start_date, valid_end_date, invalid_reason)
WHERE NOT EXISTS (
    SELECT 1 FROM vocab.concept_relationship x
    WHERE x.concept_id_1 = v.concept_id_1 AND x.relationship_id = v.relationship_id
);
