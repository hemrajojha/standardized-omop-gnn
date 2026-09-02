"""
explain_patient.py
==================
Gradient-based attribution for PatientGNN — self-contained, no model.py changes needed.

For a given patient and predicted condition label, identifies which concept
nodes (conditions, procedures, drugs) most influenced the GNN prediction,
then saves results to JSON for local OMOP lookup via query_concepts.py.

Usage (on HPC):
    nohup singularity exec --nv pytorch-251-cuda124-pykeen.sif bash -c "
    source /opt/etc/bashrc && cd /path/to/OMOP_GNN_PROJECT && \
    python -u analysis/explain_patient.py \
        --graphs_path data/processed/patient_graphs.pt \
        --checkpoint  logs/e11/checkpoints/best_model.pt \
        --vocab_path  data/processed/concept_vocab.json \
        --label_vocab data/processed/label_vocab.json \
        --patient_idx 42 --top_k 10 \
        --out_json    analysis/explain/attribution_p42.json
    " > logs/explain_patient.out 2>&1 &
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.gnn.model import PatientGNN, _sinusoidal_lpe, _metapath_rw_se


# ── Inverse vocab ──────────────────────────────────────────────────────────────

def build_index_to_concept(concept_vocab: dict) -> dict:
    inv = {}
    for domain, mapping in concept_vocab.items():
        inv[domain] = {int(vidx): int(cid) for cid, vidx in mapping.items()}
    return inv


# ── Inline gradient attribution ───────────────────────────────────────────────

def gradient_attribution(model, graph, label_idx: int, top_k: int = 10) -> dict:
    """
    Computes gradient of output logit w.r.t. each concept node embedding.
    Self-contained — does not require model.explain() method.
    """
    model.eval()

    device    = next(model.parameters()).device
    graph     = graph.to(device)
    num_visits = graph["visit"].x.size(0)
    batch_vec  = graph["visit"].x.new_zeros(num_visits, dtype=torch.long)

    # Build embeddings with grad tracking
    embed_map = {
        "condition": model.cond_embed,
        "procedure": model.proc_embed,
        "drug":      model.drug_embed,
    }
    live_embeds = {}   # domain → (embedding tensor with grad, node vocab indices)

    for domain, embed_layer in embed_map.items():
        if domain in graph.node_types and graph[domain].x.size(0) > 0:
            idx = graph[domain].x[:, 0].long().clamp(0, embed_layer.weight.size(0) - 1)
            emb = embed_layer(idx).detach().requires_grad_(True)
            live_embeds[domain] = (emb, idx)

    # ---- Forward pass using live embeddings ----------------------------------
    lpe = _sinusoidal_lpe(
        model._visit_positions(batch_vec), model.pe_dim
    ).to(device)
    rws = _metapath_rw_se(graph, num_visits, model.pe_dim, device)

    visit_in = torch.cat([graph["visit"].x, lpe, rws], dim=-1)
    x_dict   = {"visit": model.visit_proj(visit_in)}

    for domain, embed_layer in embed_map.items():
        if domain in live_embeds:
            emb, _ = live_embeds[domain]
            x_dict[domain] = F.relu(model.concept_proj(emb))
        else:
            x_dict[domain] = x_dict["visit"].new_zeros((0, model.hidden_dim))

    # Build edge index dict (reversed for HGTConv)
    edge_index_dict = {}
    reverse_map = {
        ("visit", "has_condition", "condition"): ("condition", "r_has_condition", "visit"),
        ("visit", "has_procedure", "procedure"): ("procedure", "r_has_procedure", "visit"),
        ("visit", "has_drug",      "drug"):      ("drug",      "r_has_drug",      "visit"),
    }
    for fwd, rev in reverse_map.items():
        if fwd in graph.edge_types and graph[fwd].edge_index.size(1) > 0:
            edge_index_dict[rev] = graph[fwd].edge_index.flip(0).to(device)
    temporal = ("visit", "temporal_next", "visit")
    if temporal in graph.edge_types and graph[temporal].edge_index.size(1) > 0:
        edge_index_dict[temporal] = graph[temporal].edge_index.to(device)

    for conv, norm in zip(model.convs, model.norms):
        out   = conv(x_dict, edge_index_dict)
        v_out = out.get("visit", x_dict["visit"])
        v_out = F.dropout(F.relu(v_out), p=model.dropout, training=False)
        x_dict["visit"] = norm(v_out + x_dict["visit"])

    from torch_geometric.utils import to_dense_batch
    visit_padded, mask    = to_dense_batch(x_dict["visit"], batch_vec)
    key_padding_mask      = ~mask
    graph_out = model.graph_transformer(
        visit_padded, src_key_padding_mask=key_padding_mask
    )
    raw_padded, _ = to_dense_batch(model.seq_proj(graph["visit"].x), batch_vec)
    seq_out = model.seq_transformer(
        raw_padded, src_key_padding_mask=key_padding_mask
    )

    seq_lens     = mask.sum(dim=1) - 1
    g_last       = graph_out[0, seq_lens[0]]
    s_last       = seq_out[0, seq_lens[0]]
    logits       = model.alpha * model.graph_head(g_last) + \
                   (1 - model.alpha) * model.seq_head(s_last)

    # ---- Backward ------------------------------------------------------------
    logits[label_idx].backward()

    results = {
        "logit":       logits[label_idx].item(),
        "probability": torch.sigmoid(logits[label_idx]).item(),
    }
    for domain in ("condition", "procedure", "drug"):
        if domain in live_embeds:
            emb, node_idx = live_embeds[domain]
            if emb.grad is not None:
                scores = emb.grad.abs().sum(dim=-1)
                # Top-K for attribution bars / concept graph
                k = min(top_k, scores.size(0))
                top_vals, top_pos = scores.topk(k)
                results[domain] = [
                    (node_idx[i].item(), top_vals[j].item())
                    for j, i in enumerate(top_pos)
                ]
                # All scores sorted descending — for journey map coloring
                all_order = scores.argsort(descending=True)
                results[f"{domain}_all"] = [
                    (node_idx[i].item(), scores[i].item())
                    for i in all_order
                ]
            else:
                results[domain] = []
                results[f"{domain}_all"] = []
        else:
            results[domain] = []
            results[f"{domain}_all"] = []

    return results


# ── Find a patient with rich graph ────────────────────────────────────────────

def find_patient_with_conditions(graphs, start_idx: int, min_conditions: int = 5):
    """Return the first patient from start_idx onward with >= min_conditions nodes."""
    for i, g in enumerate(graphs[start_idx:], start=start_idx):
        n_cond = g["condition"].x.size(0) if "condition" in g.node_types else 0
        if n_cond >= min_conditions:
            return i, g
    return start_idx, graphs[start_idx]


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--graphs_path",    required=True)
    p.add_argument("--checkpoint",     required=True)
    p.add_argument("--vocab_path",     required=True)
    p.add_argument("--label_vocab",    required=True)
    p.add_argument("--patient_idx",    type=int, default=0)
    p.add_argument("--label_name",     default=None,
                   help="SNOMED concept_id string. If omitted, uses top predicted.")
    p.add_argument("--top_k",          type=int, default=10)
    p.add_argument("--hidden_dim",     type=int, default=128)
    p.add_argument("--pe_dim",         type=int, default=4)
    p.add_argument("--nhead",          type=int, default=4)
    p.add_argument("--alpha",          type=float, default=0.8)
    p.add_argument("--visit_feat_dim", type=int, default=7)
    p.add_argument("--device",         default="cpu")
    p.add_argument("--out_json",       default=None)
    p.add_argument("--min_conditions", type=int, default=3,
                   help="Auto-advance to first patient with >= this many condition nodes")
    args = p.parse_args()

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
    print("Model loaded.")

    # ── Graphs ----------------------------------------------------------------
    print(f"Loading graphs: {args.graphs_path}")
    graphs     = torch.load(args.graphs_path, map_location="cpu", weights_only=False)
    n          = len(graphs)
    test_start = int(n * 0.8)
    test_graphs = graphs[test_start:]

    # Auto-find a patient with enough condition nodes
    patient_idx, graph = find_patient_with_conditions(
        test_graphs, args.patient_idx, args.min_conditions
    )
    if patient_idx != args.patient_idx:
        print(f"[info] Patient {args.patient_idx} had 0 conditions — "
              f"using patient {patient_idx} instead.")

    n_cond = graph["condition"].x.size(0) if "condition" in graph.node_types else 0
    n_proc = graph["procedure"].x.size(0) if "procedure" in graph.node_types else 0
    n_drug = graph["drug"].x.size(0)      if "drug"      in graph.node_types else 0
    print(f"Patient {patient_idx}: {graph['visit'].x.size(0)} visits | "
          f"{n_cond} conditions | {n_proc} procedures | {n_drug} drugs")

    # ── Label -----------------------------------------------------------------
    if args.label_name:
        if args.label_name not in label_vocab:
            print(f"[error] '{args.label_name}' not in label_vocab. "
                  f"Examples: {label_vocab[:5]}")
            sys.exit(1)
        label_idx = label_vocab.index(args.label_name)
        print(f"Explaining label: {args.label_name} (index {label_idx})")
    else:
        model.eval()
        with torch.no_grad():
            out   = model(graph.to(args.device), task="diagnosis")["diagnosis"]
            probs = torch.sigmoid(out[0])
        label_idx = probs.argmax().item()
        print(f"Top predicted label: {label_vocab[label_idx]} "
              f"(p={probs[label_idx]:.4f})")

    # ── Attribution -----------------------------------------------------------
    print(f"\nRunning gradient attribution for label '{label_vocab[label_idx]}'...")
    result = gradient_attribution(model, graph, label_idx, top_k=args.top_k)
    print(f"Prediction: logit={result['logit']:.4f}  p={result['probability']:.4f}")

    # ── Map node indices → OMOP concept IDs ----------------------------------
    all_concept_ids = []
    domain_results  = {}

    print(f"\n{'='*60}")
    print(f"Top {args.top_k} influential concept nodes")
    print(f"{'='*60}")

    for domain in ("condition", "procedure", "drug"):
        ranked = result.get(domain, [])
        if not ranked:
            continue
        mapped = []
        for node_idx, score in ranked:
            cid = inv_vocab.get(domain, {}).get(node_idx)
            if cid:
                mapped.append((cid, score))
                all_concept_ids.append(cid)
        if mapped:
            domain_results[domain] = mapped
            print(f"\n{domain.upper()}:")
            for cid, score in mapped:
                print(f"  concept_id={cid:>10}   attribution={score:.6f}")

    # ── Save JSON -------------------------------------------------------------
    out_path = Path(args.out_json) if args.out_json else \
               Path(f"attribution_p{patient_idx}_{label_vocab[label_idx]}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        json.dump({
            "patient_idx":  patient_idx,
            "label_snomed": label_vocab[label_idx],
            "label_idx":    label_idx,
            "logit":        result["logit"],
            "probability":  result["probability"],
            "attributions": {
                domain: [{"concept_id": cid, "score": score}
                         for cid, score in pairs]
                for domain, pairs in domain_results.items()
            },
        }, f, indent=2)

    print(f"\nSaved → {out_path}")
    print("Copy this file locally and run:  py -3 analysis/query_concepts.py "
          f"--attribution {out_path.name}")


if __name__ == "__main__":
    main()