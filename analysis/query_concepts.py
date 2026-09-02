"""
query_concepts.py
=================
Step 2 of the explainability pipeline — runs locally.

Reads the attribution JSON produced by explain_patient.py (on HPC),
queries the local PostgreSQL OMOP database for concept names and
SNOMED hierarchy, and prints a human-readable explanation.

Usage:
    py -3 analysis/query_concepts.py \\
        --attribution  attribution_patient42_label44054006.json \\
        --db_host      localhost \\
        --db_name      ohdsi \\
        --db_user      ohdsi \\
        --db_password  ohdsi

    # If DB credentials are in .env:
    py -3 analysis/query_concepts.py --attribution attribution_*.json
"""

import argparse
import json
import os
from pathlib import Path


def get_db_conn(host, dbname, user, password, port=5432):
    import psycopg2
    return psycopg2.connect(
        host=host, dbname=dbname, user=user,
        password=password, port=port,
    )


def lookup_concepts(cur, concept_ids: list) -> dict:
    """Fetch concept names, codes, and ancestors from OMOP."""
    if not concept_ids:
        return {}

    id_list = ", ".join(str(i) for i in concept_ids)

    cur.execute(f"""
        SELECT concept_id, concept_name, domain_id, concept_class_id, concept_code
        FROM omopcdm.concept
        WHERE concept_id IN ({id_list})
    """)
    concepts = {}
    for concept_id, name, domain, cls, code in cur.fetchall():
        concepts[concept_id] = {
            "name":      name,
            "domain":    domain,
            "class":     cls,
            "code":      code,
            "ancestors": [],
        }

    # Ancestors up to 3 levels up
    cur.execute(f"""
        SELECT ca.descendant_concept_id,
               ca.min_levels_of_separation,
               c.concept_name,
               c.concept_class_id
        FROM omopcdm.concept_ancestor ca
        JOIN omopcdm.concept c ON c.concept_id = ca.ancestor_concept_id
        WHERE ca.descendant_concept_id IN ({id_list})
          AND ca.min_levels_of_separation BETWEEN 1 AND 3
        ORDER BY ca.descendant_concept_id, ca.min_levels_of_separation
    """)
    for desc_id, levels, anc_name, anc_class in cur.fetchall():
        if desc_id in concepts:
            concepts[desc_id]["ancestors"].append({
                "levels_up": levels,
                "name":      anc_name,
                "class":     anc_class,
            })

    return concepts


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--attribution", required=True,
                   help="JSON file from explain_patient.py")
    p.add_argument("--db_host",     default=os.getenv("DB_HOST", "localhost"))
    p.add_argument("--db_name",     default=os.getenv("DB_NAME", "ohdsi"))
    p.add_argument("--db_user",     default=os.getenv("DB_USER", "ohdsi"))
    p.add_argument("--db_password", default=os.getenv("DB_PASSWORD", "ohdsi"))
    p.add_argument("--db_port",     type=int, default=5432)
    args = p.parse_args()

    # -- Load attribution file --------------------------------------------------
    with open(args.attribution) as f:
        data = json.load(f)

    patient_idx  = data["patient_idx"]
    label_snomed = data["label_snomed"]
    prob         = data["probability"]
    attributions = data["attributions"]   # {domain: [{concept_id, score}]}

    print(f"\n{'='*60}")
    print(f"Patient index : {patient_idx}")
    print(f"Predicted label (SNOMED): {label_snomed}")
    print(f"Prediction probability  : {prob:.4f}")
    print(f"{'='*60}")

    # Collect all concept IDs
    all_ids = [
        entry["concept_id"]
        for entries in attributions.values()
        for entry in entries
    ]

    # -- Query OMOP ------------------------------------------------------------
    print(f"\nConnecting to {args.db_host}/{args.db_name}...")
    conn = get_db_conn(args.db_host, args.db_name,
                       args.db_user, args.db_password, args.db_port)
    cur  = conn.cursor()

    # Also look up the predicted label itself
    label_info = {}
    cur.execute("""
        SELECT concept_id, concept_name, concept_code
        FROM omopcdm.concept WHERE concept_id = %s
    """, (int(label_snomed),))
    row = cur.fetchone()
    if row:
        label_info = {"name": row[1], "code": row[2]}
        print(f"Predicted condition     : {row[1]} (SNOMED: {row[2]})")

    concepts = lookup_concepts(cur, all_ids)
    cur.close()
    conn.close()

    # -- Print results ---------------------------------------------------------
    for domain, entries in attributions.items():
        if not entries:
            continue
        print(f"\n{'-'*60}")
        print(f"Top influential {domain.upper()} concepts")
        print(f"{'-'*60}")
        for entry in entries:
            cid   = entry["concept_id"]
            score = entry["score"]
            info  = concepts.get(cid, {})
            name  = info.get("name", f"concept_id={cid}")
            code  = info.get("code", "?")
            print(f"  [{score:.4f}]  {name}")
            print(f"             SNOMED: {code}  |  concept_id: {cid}")
            for anc in info.get("ancestors", [])[:2]:
                print(f"             +-- +{anc['levels_up']} level: "
                      f"{anc['name']} ({anc['class']})")

    # -- Save enriched output ---------------------------------------------------
    out_path = Path(args.attribution).stem + "_enriched.json"
    enriched = {**data, "label_name": label_info.get("name"),
                "concepts": {str(k): v for k, v in concepts.items()}}
    with open(out_path, "w") as f:
        json.dump(enriched, f, indent=2)
    print(f"\nSaved enriched results -> {out_path}")


if __name__ == "__main__":
    main()
