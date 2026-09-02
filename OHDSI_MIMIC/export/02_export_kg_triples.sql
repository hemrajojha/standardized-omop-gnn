-- =======================================================================
-- 02_export_kg_triples.sql
-- Export OMOP concept relationships as KG triples for embedding training
--
-- Produces 3 files:
--   kg_triples.csv          -- (head, relation, tail) triples
--   kg_concepts.csv         -- concept metadata (id, name, domain, vocab)
--   kg_concept_ancestors.csv -- hierarchical ancestor relationships
--
-- These are used to train KG embeddings (TransE, RotatE, etc.)
-- on HPC before injecting into the GNN model.
-- =======================================================================

-- -----------------------------------------------------------------------
-- Export 1: kg_triples.csv
-- Core KG triples from concept_relationship
-- Filtered to clinically meaningful relationship types
-- -----------------------------------------------------------------------
COPY (
    SELECT
        cr.concept_id_1                 AS head_concept_id,
        c1.concept_name                 AS head_concept_name,
        c1.domain_id                    AS head_domain,
        c1.vocabulary_id                AS head_vocab,
        cr.relationship_id              AS relation,
        cr.concept_id_2                 AS tail_concept_id,
        c2.concept_name                 AS tail_concept_name,
        c2.domain_id                    AS tail_domain,
        c2.vocabulary_id                AS tail_vocab
    FROM omopcdm.concept_relationship cr
    JOIN omopcdm.concept c1 ON c1.concept_id = cr.concept_id_1
    JOIN omopcdm.concept c2 ON c2.concept_id = cr.concept_id_2
    WHERE cr.invalid_reason IS NULL
      AND cr.relationship_id IN (
        -- Hierarchical
        'Is a',
        'Subsumes',
        -- Drug-specific
        'Has ingredient',
        'Ingredient of',
        'Has precise ingredient',
        'RxNorm has ing',
        'RxNorm ing of',
        -- Clinical finding
        'Has finding site',
        'Has pathology',
        'Has causative agent',
        'Has associated morph',
        -- Mapping
        'Maps to',
        'Mapped from',
        'Has mapping target',
        -- ATC hierarchy
        'ATC - RxNorm',
        'RxNorm - ATC',
        -- SNOMED
        'SNOMED - OMOP eq',
        'Has asso finding'
      )
      -- Only include concepts from clinically relevant domains
      AND c1.domain_id IN ('Condition', 'Drug', 'Procedure', 'Measurement', 'Observation')
      AND c2.domain_id IN ('Condition', 'Drug', 'Procedure', 'Measurement', 'Observation')
      AND c1.standard_concept IS NOT NULL
      AND c2.standard_concept IS NOT NULL
    ORDER BY cr.relationship_id, cr.concept_id_1
) TO 'C:/omop_export/kg_triples.csv' CSV HEADER ENCODING 'UTF8';

-- -----------------------------------------------------------------------
-- Export 2: kg_concepts.csv
-- Full concept metadata for all concepts appearing in patient data
-- Used to build vocabulary index for the GNN
-- -----------------------------------------------------------------------
COPY (
    WITH patient_concepts AS (
        -- All concept IDs that actually appear in patient clinical data
        SELECT DISTINCT condition_concept_id AS concept_id FROM omopcdm.condition_occurrence
        UNION
        SELECT DISTINCT procedure_concept_id  FROM omopcdm.procedure_occurrence
        UNION
        SELECT DISTINCT drug_concept_id       FROM omopcdm.drug_exposure
        UNION
        SELECT DISTINCT measurement_concept_id FROM omopcdm.measurement
            WHERE value_as_number IS NOT NULL
    )
    SELECT
        c.concept_id,
        c.concept_name,
        c.domain_id,
        c.vocabulary_id,
        c.concept_class_id,
        c.standard_concept,
        c.concept_code,
        c.valid_start_date,
        c.valid_end_date,
        c.invalid_reason,
        -- Count of patients who have this concept (for filtering rare codes)
        COALESCE(co_cnt.cnt, 0)  AS condition_patient_count,
        COALESCE(po_cnt.cnt, 0)  AS procedure_patient_count,
        COALESCE(de_cnt.cnt, 0)  AS drug_patient_count,
        COALESCE(me_cnt.cnt, 0)  AS measurement_patient_count
    FROM patient_concepts pc
    JOIN omopcdm.concept c ON c.concept_id = pc.concept_id
    LEFT JOIN (
        SELECT condition_concept_id AS concept_id, COUNT(DISTINCT person_id) AS cnt
        FROM omopcdm.condition_occurrence GROUP BY condition_concept_id
    ) co_cnt ON co_cnt.concept_id = c.concept_id
    LEFT JOIN (
        SELECT procedure_concept_id AS concept_id, COUNT(DISTINCT person_id) AS cnt
        FROM omopcdm.procedure_occurrence GROUP BY procedure_concept_id
    ) po_cnt ON po_cnt.concept_id = c.concept_id
    LEFT JOIN (
        SELECT drug_concept_id AS concept_id, COUNT(DISTINCT person_id) AS cnt
        FROM omopcdm.drug_exposure GROUP BY drug_concept_id
    ) de_cnt ON de_cnt.concept_id = c.concept_id
    LEFT JOIN (
        SELECT measurement_concept_id AS concept_id, COUNT(DISTINCT person_id) AS cnt
        FROM omopcdm.measurement WHERE value_as_number IS NOT NULL
        GROUP BY measurement_concept_id
    ) me_cnt ON me_cnt.concept_id = c.concept_id
    WHERE c.concept_id != 0
    ORDER BY c.domain_id, c.vocabulary_id, c.concept_id
) TO 'C:/omop_export/kg_concepts.csv' CSV HEADER ENCODING 'UTF8';

-- -----------------------------------------------------------------------
-- Export 3: kg_concept_ancestors.csv
-- Hierarchical ancestor relationships (IS-A transitive closure)
-- Used for Option B semantic edges — ancestor codes as additional nodes
-- -----------------------------------------------------------------------
COPY (
    WITH patient_concepts AS (
        SELECT DISTINCT condition_concept_id AS concept_id FROM omopcdm.condition_occurrence
        UNION
        SELECT DISTINCT procedure_concept_id  FROM omopcdm.procedure_occurrence
        UNION
        SELECT DISTINCT drug_concept_id       FROM omopcdm.drug_exposure
    )
    SELECT
        ca.ancestor_concept_id,
        ca.descendant_concept_id,
        ca.min_levels_of_separation,
        ca.max_levels_of_separation,
        c1.concept_name     AS ancestor_name,
        c1.domain_id        AS ancestor_domain,
        c1.vocabulary_id    AS ancestor_vocab,
        c2.concept_name     AS descendant_name,
        c2.domain_id        AS descendant_domain,
        c2.vocabulary_id    AS descendant_vocab
    FROM omopcdm.concept_ancestor ca
    JOIN patient_concepts pc ON pc.concept_id = ca.descendant_concept_id
    JOIN omopcdm.concept c1 ON c1.concept_id = ca.ancestor_concept_id
    JOIN omopcdm.concept c2 ON c2.concept_id = ca.descendant_concept_id
    WHERE ca.min_levels_of_separation BETWEEN 1 AND 3
      AND c1.domain_id IN ('Condition', 'Drug', 'Procedure')
      AND c2.domain_id IN ('Condition', 'Drug', 'Procedure')
    ORDER BY ca.descendant_concept_id, ca.min_levels_of_separation
) TO 'C:/omop_export/kg_concept_ancestors.csv' CSV HEADER ENCODING 'UTF8';
