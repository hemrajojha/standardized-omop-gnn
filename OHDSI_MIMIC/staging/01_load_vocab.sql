-- =======================================================================
-- 01_load_vocab.sql
-- Load OHDSI Athena vocabulary CSVs into the `vocab` schema.
--
-- NOTE: \COPY must be on a single line (psql meta-command limitation).
-- Athena files use TAB delimiter, not comma.
--
-- Run via psql:
--   psql -U postgres -d mimiciv_omop -f "OHDSI_MIMIC/staging/01_load_vocab.sql"
--
-- CONCEPT.csv can be >1 GB — this step may take several minutes.
-- =======================================================================

SET work_mem = '512MB';

TRUNCATE vocab.concept;
\COPY vocab.concept (concept_id, concept_name, domain_id, vocabulary_id, concept_class_id, standard_concept, concept_code, valid_start_date, valid_end_date, invalid_reason) FROM '/path/to/vocab/CONCEPT.csv' CSV HEADER DELIMITER E'\t' QUOTE E'\b' ENCODING 'LATIN1';

TRUNCATE vocab.vocabulary;
\COPY vocab.vocabulary (vocabulary_id, vocabulary_name, vocabulary_reference, vocabulary_version, vocabulary_concept_id) FROM '/path/to/vocab/VOCABULARY.csv' CSV HEADER DELIMITER E'\t' QUOTE E'\b';

TRUNCATE vocab.domain;
\COPY vocab.domain (domain_id, domain_name, domain_concept_id) FROM '/path/to/vocab/DOMAIN.csv' CSV HEADER DELIMITER E'\t' QUOTE E'\b';

TRUNCATE vocab.concept_class;
\COPY vocab.concept_class (concept_class_id, concept_class_name, concept_class_concept_id) FROM '/path/to/vocab/CONCEPT_CLASS.csv' CSV HEADER DELIMITER E'\t' QUOTE E'\b';

TRUNCATE vocab.concept_relationship;
\COPY vocab.concept_relationship (concept_id_1, concept_id_2, relationship_id, valid_start_date, valid_end_date, invalid_reason) FROM '/path/to/vocab/CONCEPT_RELATIONSHIP.csv' CSV HEADER DELIMITER E'\t' QUOTE E'\b';

TRUNCATE vocab.relationship;
\COPY vocab.relationship (relationship_id, relationship_name, is_hierarchical, defines_ancestry, reverse_relationship_id, relationship_concept_id) FROM '/path/to/vocab/RELATIONSHIP.csv' CSV HEADER DELIMITER E'\t' QUOTE E'\b';

TRUNCATE vocab.concept_synonym;
\COPY vocab.concept_synonym (concept_id, concept_synonym_name, language_concept_id) FROM '/path/to/vocab/CONCEPT_SYNONYM.csv' CSV HEADER DELIMITER E'\t' QUOTE E'\b' ENCODING 'LATIN1';

TRUNCATE vocab.concept_ancestor;
\COPY vocab.concept_ancestor (ancestor_concept_id, descendant_concept_id, min_levels_of_separation, max_levels_of_separation) FROM '/path/to/vocab/CONCEPT_ANCESTOR.csv' CSV HEADER DELIMITER E'\t' QUOTE E'\b';

TRUNCATE vocab.drug_strength;
\COPY vocab.drug_strength (drug_concept_id, ingredient_concept_id, amount_value, amount_unit_concept_id, numerator_value, numerator_unit_concept_id, denominator_value, denominator_unit_concept_id, box_size, valid_start_date, valid_end_date, invalid_reason) FROM '/path/to/vocab/DRUG_STRENGTH.csv' CSV HEADER DELIMITER E'\t' QUOTE E'\b';

-- -----------------------------------------------------------------------
-- Indexes for vocabulary join performance
-- -----------------------------------------------------------------------
CREATE INDEX IF NOT EXISTS idx_voc_concept_id   ON vocab.concept (concept_id);
CREATE INDEX IF NOT EXISTS idx_voc_concept_code ON vocab.concept (concept_code, vocabulary_id);
CREATE INDEX IF NOT EXISTS idx_voc_cr_id1       ON vocab.concept_relationship (concept_id_1, relationship_id);
CREATE INDEX IF NOT EXISTS idx_voc_cr_id2       ON vocab.concept_relationship (concept_id_2);
CREATE INDEX IF NOT EXISTS idx_voc_ca_anc       ON vocab.concept_ancestor (ancestor_concept_id);
CREATE INDEX IF NOT EXISTS idx_voc_ca_desc      ON vocab.concept_ancestor (descendant_concept_id);

-- -----------------------------------------------------------------------
-- Row counts
-- -----------------------------------------------------------------------
SELECT 'concept'              AS tbl, COUNT(*) AS n FROM vocab.concept
UNION ALL SELECT 'concept_relationship', COUNT(*) FROM vocab.concept_relationship
UNION ALL SELECT 'concept_ancestor',     COUNT(*) FROM vocab.concept_ancestor
UNION ALL SELECT 'vocabulary',           COUNT(*) FROM vocab.vocabulary
UNION ALL SELECT 'drug_strength',        COUNT(*) FROM vocab.drug_strength
ORDER BY tbl;
