-- =======================================================================
-- 03_export_concept_metadata.sql
-- Export concept metadata and code mappings for interpretability
--
-- Produces 4 files:
--   concept_icd_to_omop.csv     -- ICD → OMOP standard concept mapping
--   concept_drug_mapping.csv    -- RxNorm/NDC → ATC hierarchy mapping
--   concept_vocab_summary.csv   -- vocabulary coverage statistics
--   concept_relationship_types.csv -- all relationship types with counts
--
-- Used for:
--   - Interpreting GNN predictions back to clinical codes
--   - Code normalization (ICD-9/10 → standard OMOP)
--   - ATC drug hierarchy for TRANS-compatible drug encoding
-- =======================================================================

-- -----------------------------------------------------------------------
-- Export 1: concept_icd_to_omop.csv
-- Maps raw ICD source codes to standard OMOP condition concepts
-- Critical for aligning TRANS (uses raw ICD) with OMOP (uses standard)
-- -----------------------------------------------------------------------
COPY (
    SELECT DISTINCT
        sc.concept_id                   AS source_concept_id,
        sc.concept_code                 AS icd_code,
        sc.vocabulary_id                AS icd_vocabulary,
        sc.concept_name                 AS icd_description,
        tc.concept_id                   AS omop_concept_id,
        tc.concept_code                 AS omop_code,
        tc.vocabulary_id                AS omop_vocabulary,
        tc.concept_name                 AS omop_description,
        tc.domain_id,
        cr.relationship_id
    FROM omopcdm.concept_relationship cr
    JOIN omopcdm.concept sc ON sc.concept_id = cr.concept_id_1
    JOIN omopcdm.concept tc ON tc.concept_id = cr.concept_id_2
    WHERE cr.relationship_id IN ('Maps to', 'SNOMED - OMOP eq')
      AND sc.vocabulary_id IN ('ICD9CM', 'ICD10CM', 'ICD10', 'ICD9Proc', 'ICD10PCS')
      AND tc.standard_concept = 'S'
      AND cr.invalid_reason IS NULL
      AND tc.invalid_reason IS NULL
    ORDER BY sc.vocabulary_id, sc.concept_code
) TO 'C:/omop_export/concept_icd_to_omop.csv' CSV HEADER;

-- -----------------------------------------------------------------------
-- Export 2: concept_drug_mapping.csv
-- RxNorm drug concepts with ATC hierarchy (for TRANS drug encoding)
-- TRANS uses ATC level-3 codes — this provides the full hierarchy
-- -----------------------------------------------------------------------
COPY (
    SELECT DISTINCT
        rxnorm.concept_id               AS rxnorm_concept_id,
        rxnorm.concept_code             AS rxnorm_code,
        rxnorm.concept_name             AS drug_name,
        rxnorm.concept_class_id         AS rxnorm_class,
        atc.concept_id                  AS atc_concept_id,
        atc.concept_code                AS atc_code,
        atc.concept_name                AS atc_name,
        atc.concept_class_id            AS atc_level,
        -- ATC level-3 prefix (first 4 chars) — matches TRANS encoding
        LEFT(atc.concept_code, 4)       AS atc_level3_code
    FROM omopcdm.concept rxnorm
    JOIN omopcdm.concept_relationship cr
        ON cr.concept_id_1 = rxnorm.concept_id
        AND cr.relationship_id IN ('RxNorm - ATC', 'ATC - RxNorm')
        AND cr.invalid_reason IS NULL
    JOIN omopcdm.concept atc
        ON atc.concept_id = cr.concept_id_2
        AND atc.vocabulary_id = 'ATC'
        AND atc.invalid_reason IS NULL
    WHERE rxnorm.vocabulary_id IN ('RxNorm', 'RxNorm Extension')
      AND rxnorm.standard_concept = 'S'
      AND rxnorm.invalid_reason IS NULL
    ORDER BY rxnorm.concept_code, atc.concept_class_id
) TO 'C:/omop_export/concept_drug_mapping.csv' CSV HEADER;

-- -----------------------------------------------------------------------
-- Export 3: concept_vocab_summary.csv
-- Coverage statistics — how many patient events map to standard concepts
-- Helps identify unmapped codes that need attention
-- -----------------------------------------------------------------------
COPY (
    SELECT
        domain,
        vocabulary_id,
        standard_concept,
        total_events,
        distinct_concepts,
        distinct_patients,
        ROUND(mapped_events * 100.0 / NULLIF(total_events, 0), 2) AS pct_mapped
    FROM (
        SELECT
            'Condition'                     AS domain,
            c.vocabulary_id,
            c.standard_concept,
            COUNT(*)                        AS total_events,
            COUNT(DISTINCT co.condition_concept_id) AS distinct_concepts,
            COUNT(DISTINCT co.person_id)    AS distinct_patients,
            SUM(CASE WHEN co.condition_concept_id != 0 THEN 1 ELSE 0 END) AS mapped_events
        FROM omopcdm.condition_occurrence co
        LEFT JOIN omopcdm.concept c ON c.concept_id = co.condition_concept_id
        GROUP BY c.vocabulary_id, c.standard_concept

        UNION ALL

        SELECT
            'Procedure',
            c.vocabulary_id,
            c.standard_concept,
            COUNT(*),
            COUNT(DISTINCT po.procedure_concept_id),
            COUNT(DISTINCT po.person_id),
            SUM(CASE WHEN po.procedure_concept_id != 0 THEN 1 ELSE 0 END)
        FROM omopcdm.procedure_occurrence po
        LEFT JOIN omopcdm.concept c ON c.concept_id = po.procedure_concept_id
        GROUP BY c.vocabulary_id, c.standard_concept

        UNION ALL

        SELECT
            'Drug',
            c.vocabulary_id,
            c.standard_concept,
            COUNT(*),
            COUNT(DISTINCT de.drug_concept_id),
            COUNT(DISTINCT de.person_id),
            SUM(CASE WHEN de.drug_concept_id != 0 THEN 1 ELSE 0 END)
        FROM omopcdm.drug_exposure de
        LEFT JOIN omopcdm.concept c ON c.concept_id = de.drug_concept_id
        GROUP BY c.vocabulary_id, c.standard_concept

        UNION ALL

        SELECT
            'Measurement',
            c.vocabulary_id,
            c.standard_concept,
            COUNT(*),
            COUNT(DISTINCT m.measurement_concept_id),
            COUNT(DISTINCT m.person_id),
            SUM(CASE WHEN m.measurement_concept_id != 0 THEN 1 ELSE 0 END)
        FROM omopcdm.measurement m
        LEFT JOIN omopcdm.concept c ON c.concept_id = m.measurement_concept_id
        GROUP BY c.vocabulary_id, c.standard_concept
    ) summary
    ORDER BY domain, total_events DESC
) TO 'C:/omop_export/concept_vocab_summary.csv' CSV HEADER;

-- -----------------------------------------------------------------------
-- Export 4: concept_relationship_types.csv
-- All relationship types with counts — helps select which to include in KG
-- -----------------------------------------------------------------------
COPY (
    SELECT
        cr.relationship_id,
        r.relationship_name,
        COUNT(*)                        AS triple_count,
        COUNT(DISTINCT cr.concept_id_1) AS distinct_heads,
        COUNT(DISTINCT cr.concept_id_2) AS distinct_tails,
        -- Domain distribution
        MODE() WITHIN GROUP (ORDER BY c1.domain_id) AS most_common_head_domain,
        MODE() WITHIN GROUP (ORDER BY c2.domain_id) AS most_common_tail_domain
    FROM omopcdm.concept_relationship cr
    JOIN omopcdm.concept c1 ON c1.concept_id = cr.concept_id_1
    JOIN omopcdm.concept c2 ON c2.concept_id = cr.concept_id_2
    LEFT JOIN omopcdm.relationship r ON r.relationship_id = cr.relationship_id
    WHERE cr.invalid_reason IS NULL
      AND c1.domain_id IN ('Condition', 'Drug', 'Procedure', 'Measurement', 'Observation')
      AND c2.domain_id IN ('Condition', 'Drug', 'Procedure', 'Measurement', 'Observation')
    GROUP BY cr.relationship_id, r.relationship_name
    ORDER BY triple_count DESC
) TO 'C:/omop_export/concept_relationship_types.csv' CSV HEADER;
