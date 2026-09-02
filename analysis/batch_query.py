"""
batch_query.py
==============
Enriches every attribution_p*.json in --in_dir with:
  1. OMOP concept names + SNOMED hierarchy  (omopcdm.concept / concept_ancestor)
  2. Concept names back-filled into visit timeline already embedded in the JSON
     (visit timeline is extracted from graph edges by batch_explain.py — no DB
     visit queries needed)

Runs LOCALLY against the PostgreSQL OMOP database.
Opens one DB connection and batches ALL concept IDs into efficient queries.

Usage:
    py -3 analysis/batch_query.py \
        --in_dir      analysis/explain/batch \
        --db_host     localhost \
        --db_name     mimiciv_omop \
        --db_user     postgres \
        --db_password postgres

Options:
    --overwrite      Re-enrich even if _enriched.json already exists.
    --min_prob       Only enrich files where probability >= threshold (default 0.0).
"""

import argparse
import json
import os
from pathlib import Path


CHUNK = 500   # max person_ids per SQL IN clause


def get_db_conn(host, dbname, user, password, port=5432):
    import psycopg2
    return psycopg2.connect(
        host=host, dbname=dbname, user=user, password=password, port=port,
    )


# ── Concept lookup ────────────────────────────────────────────────────────────

def batch_lookup_concepts(cur, concept_ids: set) -> dict:
    """
    Single pair of queries covering ALL concept IDs.
    Returns {concept_id (int): {name, domain, class, code, ancestors}}.
    """
    if not concept_ids:
        return {}

    concepts = {}
    id_list_chunks = list(_chunks(sorted(concept_ids), CHUNK))  # materialise — generators exhaust

    for chunk in id_list_chunks:
        id_list = ", ".join(str(i) for i in chunk)
        cur.execute(f"""
            SELECT concept_id, concept_name, domain_id, concept_class_id, concept_code
            FROM omopcdm.concept
            WHERE concept_id IN ({id_list})
        """)
        for concept_id, name, domain, cls, code in cur.fetchall():
            concepts[int(concept_id)] = {
                "name":      name,
                "domain":    domain,
                "class":     cls,
                "code":      code,
                "ancestors": [],
            }

    for chunk in id_list_chunks:
        id_list = ", ".join(str(i) for i in chunk)
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
            desc_id = int(desc_id)
            if desc_id in concepts:
                concepts[desc_id]["ancestors"].append({
                    "levels_up": int(levels),
                    "name":      anc_name,
                    "class":     anc_class,
                })

    return concepts


def _chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]


def lookup_label(cur, label_snomed: str) -> dict:
    try:
        snomed_int = int(label_snomed)
    except ValueError:
        return {}
    cur.execute("""
        SELECT concept_id, concept_name, concept_code
        FROM omopcdm.concept WHERE concept_id = %s
    """, (snomed_int,))
    row = cur.fetchone()
    if row:
        return {"name": row[1], "code": row[2]}
    return {}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in_dir",       default="analysis/explain/batch")
    p.add_argument("--db_host",      default=os.getenv("DB_HOST", "localhost"))
    p.add_argument("--db_name",      default=os.getenv("DB_NAME", "mimiciv_omop"))
    p.add_argument("--db_user",      default=os.getenv("DB_USER", "postgres"))
    p.add_argument("--db_password",  default=os.getenv("DB_PASSWORD", "postgres"))
    p.add_argument("--db_port",      type=int, default=5432)
    p.add_argument("--overwrite",    action="store_true")
    p.add_argument("--min_prob",     type=float, default=0.0)
    args = p.parse_args()

    in_dir = Path(args.in_dir)
    files  = sorted(f for f in in_dir.glob("attribution_p*.json")
                    if not f.stem.endswith("_enriched"))
    print(f"Found {len(files)} attribution files in {in_dir}")

    if not files:
        print("Nothing to enrich.")
        return

    # Filter by probability
    if args.min_prob > 0:
        filtered = []
        for f in files:
            with open(f) as fh:
                d = json.load(fh)
            if d.get("probability", 0) >= args.min_prob:
                filtered.append(f)
        print(f"After min_prob={args.min_prob}: {len(filtered)} files")
        files = filtered

    # Skip already enriched
    if not args.overwrite:
        todo = [f for f in files
                if not (f.parent / (f.stem + "_enriched.json")).exists()]
        print(f"To enrich: {len(todo)}  (already done: {len(files)-len(todo)})")
        files = todo

    if not files:
        print("All files already enriched. Use --overwrite to redo.")
        return

    # ── Load all attribution files + collect concept IDs ──────────────────────
    print("Scanning concept IDs...")
    file_data        = {}
    all_concept_ids  = set()

    for f in files:
        with open(f) as fh:
            data = json.load(fh)
        file_data[f] = data
        # Attribution concept IDs (top-K)
        for entries in data.get("attributions", {}).values():
            for e in entries:
                all_concept_ids.add(int(e["concept_id"]))
        # All-scores concept IDs (for journey map)
        for domain_scores in data.get("all_scores", {}).values():
            for cid in domain_scores:
                all_concept_ids.add(int(cid))
        # Visit timeline concept IDs (from graph edge extraction)
        for visit in data.get("visits", []):
            for domain_key in ("conditions", "procedures", "drugs"):
                for cid in visit.get(domain_key, []):
                    all_concept_ids.add(int(cid))
        try:
            all_concept_ids.add(int(data["label_snomed"]))
        except (ValueError, KeyError):
            pass

    # ── DB connection ─────────────────────────────────────────────────────────
    print(f"Connecting to {args.db_host}/{args.db_name}...")
    conn = get_db_conn(args.db_host, args.db_name,
                       args.db_user, args.db_password, args.db_port)
    cur  = conn.cursor()

    # Visit timeline is already embedded in the raw JSON (from batch_explain.py).
    # Concept IDs from visits were already collected in the scanning loop above.

    # ── Concept lookup (attribution + visit + label IDs in one batch) ─────────
    print(f"Total unique concept IDs to look up: {len(all_concept_ids)}")
    concepts = batch_lookup_concepts(cur, all_concept_ids)
    print(f"Resolved {len(concepts)} / {len(all_concept_ids)} concept IDs")

    cur.close()
    conn.close()

    # ── Write enriched files ──────────────────────────────────────────────────
    print("Writing enriched JSONs...")
    written = 0
    for f, data in file_data.items():
        label_snomed = data.get("label_snomed", "")
        try:
            label_cid  = int(label_snomed)
            label_name = concepts.get(label_cid, {}).get("name", label_snomed)
        except ValueError:
            label_name = label_snomed

        # Concept subset (attributed concepts only for JSON compactness)
        concept_subset = {}
        for entries in data.get("attributions", {}).values():
            for e in entries:
                cid = int(e["concept_id"])
                if cid in concepts:
                    concept_subset[cid] = concepts[cid]

        # Back-fill concept names into visits (raw JSON has concept_ids only)
        raw_visits = data.get("visits", [])
        enriched_visits = []
        for v in raw_visits:
            ev = dict(v)
            for domain_key in ("conditions", "procedures", "drugs"):
                ev[domain_key] = [
                    {"concept_id": cid,
                     "name": concepts.get(cid, {}).get("name", str(cid))}
                    for cid in v.get(domain_key, [])
                ]
            enriched_visits.append(ev)

        # Enrich all_scores with concept names
        enriched_all_scores = {}
        for domain, domain_scores in data.get("all_scores", {}).items():
            enriched_all_scores[domain] = {
                str(cid): {
                    "score": score,
                    "name":  concepts.get(int(cid), {}).get("name", str(cid)),
                }
                for cid, score in domain_scores.items()
            }

        enriched = {
            **data,
            "label_name": label_name,
            "concepts":   {str(k): v for k, v in concept_subset.items()},
            "all_scores": enriched_all_scores,
            "visits":     enriched_visits,
        }

        out_path = f.parent / (f.stem + "_enriched.json")
        with open(out_path, "w") as fh:
            json.dump(enriched, fh)
        written += 1

        if written % 500 == 0:
            print(f"  {written}/{len(files)} written...")

    print(f"\nDone. Enriched {written} files -> {in_dir}")


if __name__ == "__main__":
    main()