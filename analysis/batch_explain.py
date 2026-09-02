"""
batch_explain.py
================
Runs gradient attribution for every patient in the test split and saves
one JSON per patient to --out_dir.

Designed to run on HPC (GPU). Loads the model and graphs once, then
iterates. Skips patients whose JSON already exists (safe to resume after
interruption).

Usage (HPC):
    nohup singularity exec --nv pytorch-251-cuda124-pykeen.sif bash -c "
    source /opt/etc/bashrc && cd /path/to/OMOP_GNN_PROJECT && \
    python -u analysis/batch_explain.py \
        --graphs_path data/processed/patient_graphs.pt \
        --checkpoint  logs/e11/checkpoints/best_model.pt \
        --vocab_path  data/processed/concept_vocab.json \
        --label_vocab data/processed/label_vocab.json \
        --out_dir     analysis/explain/batch \
        --top_k       10 \
        --device      cuda \
        --min_conditions 1
    " > logs/batch_explain.out 2>&1 &

Options:
    --start / --end   Process a slice of test patients (for parallelism).
                      Default: all test patients.
    --min_conditions  Skip patients with fewer than N condition nodes (default 1).
    --overwrite       Re-compute even if JSON already exists.
"""

import argparse
import json
import sys
import traceback
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.gnn.model import PatientGNN, _sinusoidal_lpe, _metapath_rw_se
from analysis.explain_patient import (
    build_index_to_concept,
    gradient_attribution,
)


def extract_visit_timeline(graph, inv_vocab: dict) -> list:
    """
    Build a per-visit concept list directly from the graph's edge structure.
    This guarantees the concept_ids exactly match those used in attribution.

    Note: the last visit's condition edges are removed during preprocessing
    (to prevent label leakage), so the current visit will show no conditions —
    this is expected and should be indicated in the UI.

    Returns:
        list of dicts: [{visit_idx, is_current, conditions, procedures, drugs}]
        where conditions/procedures/drugs are lists of OMOP concept_id (int).
    """
    if "visit" not in graph.node_types:
        return []

    n_visits = graph["visit"].x.size(0)
    visits   = []

    # Map: domain key -> (edge_type tuple, inv_vocab domain key)
    domain_map = {
        "conditions": (("visit", "has_condition", "condition"), "condition"),
        "procedures": (("visit", "has_procedure", "procedure"), "procedure"),
        "drugs":      (("visit", "has_drug",       "drug"),      "drug"),
    }

    for vi in range(n_visits):
        visit = {
            "visit_idx":  vi,
            "is_current": vi == n_visits - 1,
            "conditions": [],
            "procedures": [],
            "drugs":      [],
        }

        for domain_key, (edge_type, vocab_key) in domain_map.items():
            if edge_type not in graph.edge_types:
                continue
            node_type = edge_type[2]   # "condition", "procedure", "drug"
            ei        = graph[edge_type].edge_index
            node_x    = graph[node_type].x  # node features: col 0 = vocab index
            mask      = ei[0] == vi
            for ni in ei[1][mask].tolist():
                # ni is graph node index; look up its vocab index from node features
                vocab_idx = int(node_x[ni, 0])
                cid = inv_vocab.get(vocab_key, {}).get(vocab_idx)
                if cid is not None:
                    visit[domain_key].append(int(cid))

        visits.append(visit)

    return visits


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--graphs_path",    required=True)
    p.add_argument("--checkpoint",     required=True)
    p.add_argument("--vocab_path",     required=True)
    p.add_argument("--label_vocab",    required=True)
    p.add_argument("--out_dir",        default="analysis/explain/batch")
    p.add_argument("--top_k",          type=int,   default=10)
    p.add_argument("--hidden_dim",     type=int,   default=128)
    p.add_argument("--pe_dim",         type=int,   default=4)
    p.add_argument("--nhead",          type=int,   default=4)
    p.add_argument("--alpha",          type=float, default=0.8)
    p.add_argument("--visit_feat_dim", type=int,   default=7)
    p.add_argument("--device",         default="cpu")
    p.add_argument("--start",          type=int,   default=0,
                   help="First test-patient index to process (default: 0)")
    p.add_argument("--end",            type=int,   default=None,
                   help="Last test-patient index (exclusive, default: all)")
    p.add_argument("--min_conditions", type=int,   default=1,
                   help="Skip patients with fewer condition nodes")
    p.add_argument("--overwrite",      action="store_true",
                   help="Re-compute even if output JSON already exists")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Vocab ------------------------------------------------------------------
    with open(args.vocab_path) as f:
        concept_vocab = json.load(f)
    with open(args.label_vocab) as f:
        label_vocab = json.load(f)

    inv_vocab   = build_index_to_concept(concept_vocab)
    vocab_sizes = {
        d: max(int(v) for v in concept_vocab.get(d, {"0": 0}).values()) + 1
        for d in ("condition", "procedure", "drug")
    }

    # ── Model -----------------------------------------------------------------
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt  = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)

    model = PatientGNN.from_config(
        no_kg=True,
        visit_feat_dim=args.visit_feat_dim,
        hidden_dim=args.hidden_dim,
        pe_dim=args.pe_dim,
        nhead=args.nhead,
        alpha=args.alpha,
        num_labels=len(label_vocab),
        cond_vocab_size=vocab_sizes["condition"],
        proc_vocab_size=vocab_sizes["procedure"],
        drug_vocab_size=vocab_sizes["drug"],
    )
    model.load_state_dict(state, strict=False)
    model.to(args.device)
    model.eval()
    print(f"Model loaded on {args.device}.")

    # ── Graphs ----------------------------------------------------------------
    print(f"Loading graphs: {args.graphs_path}")
    graphs     = torch.load(args.graphs_path, map_location="cpu",
                            weights_only=False)
    n          = len(graphs)
    test_start = int(n * 0.8)
    test_graphs = graphs[test_start:]
    print(f"Total graphs: {n} | Test split: {len(test_graphs)} patients")

    # Slice
    end = args.end if args.end is not None else len(test_graphs)
    subset = list(range(args.start, min(end, len(test_graphs))))
    print(f"Processing test patients [{args.start}, {end}) -> {len(subset)} patients")

    # ── Loop ------------------------------------------------------------------
    done = skipped = errors = 0

    for local_idx in subset:
        graph = test_graphs[local_idx]

        # Check min conditions
        n_cond = graph["condition"].x.size(0) if "condition" in graph.node_types else 0
        if n_cond < args.min_conditions:
            skipped += 1
            continue

        out_path = out_dir / f"attribution_p{local_idx}.json"
        if out_path.exists() and not args.overwrite:
            done += 1
            continue

        try:
            # Top predicted label
            with torch.no_grad():
                out   = model(graph.to(args.device), task="diagnosis")["diagnosis"]
                probs = torch.sigmoid(out[0])
            label_idx   = probs.argmax().item()
            label_snomed = label_vocab[label_idx]
            prob         = probs[label_idx].item()

            # Attribution
            result = gradient_attribution(model, graph, label_idx, top_k=args.top_k)

            # Map vocab indices -> OMOP concept IDs (top-K and all scores)
            domain_results = {}
            all_scores     = {}
            for domain in ("condition", "procedure", "drug"):
                # Top-K for attribution bars / concept graph
                ranked = result.get(domain, [])
                mapped = []
                for node_idx, score in ranked:
                    cid = inv_vocab.get(domain, {}).get(node_idx)
                    if cid:
                        mapped.append({"concept_id": cid, "score": score})
                if mapped:
                    domain_results[domain] = mapped

                # All scores {concept_id: score} for journey map coloring
                domain_all = {}
                for node_idx, score in result.get(f"{domain}_all", []):
                    cid = inv_vocab.get(domain, {}).get(node_idx)
                    if cid:
                        domain_all[cid] = score
                if domain_all:
                    all_scores[domain] = domain_all

            # Visit timeline from graph edge structure (exact concept_ids)
            visits = extract_visit_timeline(graph, inv_vocab)

            # Subject ID for reference
            subject_id = int(getattr(graph, "subject_id", -1))

            with open(out_path, "w") as f:
                json.dump({
                    "patient_idx":   local_idx,
                    "subject_id":    subject_id,
                    "label_snomed":  label_snomed,
                    "label_idx":     label_idx,
                    "logit":         result["logit"],
                    "probability":   prob,
                    "n_conditions":  n_cond,
                    "n_procedures":  graph["procedure"].x.size(0) if "procedure" in graph.node_types else 0,
                    "n_drugs":       graph["drug"].x.size(0)      if "drug"      in graph.node_types else 0,
                    "attributions":  domain_results,
                    "all_scores":    all_scores,
                    "visits":        visits,
                }, f)

            done += 1
            if done % 50 == 0:
                print(f"  [{done}/{len(subset)}] patient {local_idx} -> "
                      f"{label_snomed} p={prob:.3f}")

        except Exception:
            errors += 1
            print(f"  [ERROR] patient {local_idx}:")
            traceback.print_exc()

    print(f"\nDone. Processed: {done} | Skipped (low nodes): {skipped} | Errors: {errors}")
    print(f"JSONs saved to: {out_dir}")


if __name__ == "__main__":
    main()