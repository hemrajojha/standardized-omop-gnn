"""
analyse_early_prediction.py
============================
Temporal early-warning analysis for PatientGNN E11.

Question: How many visits in advance can PatientGNN predict a patient's
eventual diagnoses?

Method
------
For each lead time k = 0, 1, 2, ..., max_lead (default 3):
  1. Build a truncated copy of each test graph, retaining only the first
     N-k visit nodes (where N is the patient's total number of visits).
  2. Run PatientGNN inference on the truncated graph.
     The model automatically uses the last retained visit's representation
     (via to_dense_batch + mask.sum(dim=1)-1 in the forward pass).
  3. Evaluate against the original y_diagnosis label — the conditions
     confirmed at the patient's MOST RECENT visit (visit N-1).

The label is FIXED across all lead times. Only the model input changes.
This answers:
  "Given a patient's medical record from k+1 visits ago, how well can
   PatientGNN predict the diagnoses confirmed at their most recent visit?"

Lead 0 (k=0) → standard evaluation (matches reported E11 AUROC ~0.909)
Lead k>0     → model sees k fewer recent visits; AUROC expected to decay
               gracefully if the model encodes meaningful trajectory signals

Leakage prevention
------------------
The original graphs have the LAST visit's condition edges removed (matching
training). Truncated graphs have the NEW last visit's condition edges removed
to maintain the same prevention. Procedure and drug edges are fully retained
for all included visits (no leakage risk since they are features, not labels).

Outputs (written to --out_dir)
-------------------------------
  early_prediction_results.csv     per-label AUROC at each lead time
  fig_lead_vs_auroc.png            macro-AUROC vs lead time (primary figure)
  fig_lead_heatmap.png             per-condition AUROC heatmap (top-50 by freq)
  fig_lead_decay_top15.png         per-condition AUROC line plot (top-15)

Usage
-----
python analyse_early_prediction.py \\
  --gnn_ckpt    /path/to/OMOP_GNN_PROJECT/logs/e11/checkpoints/best_model.pt \\
  --graphs_path /path/to/OMOP_GNN_PROJECT/data/processed/patient_graphs.pt \\
  --out_dir     analysis/early_prediction \\
  --device      cuda:1 \\
  --max_lead    3
"""

import argparse
import csv
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch_geometric.data import Batch as GeometricBatch, DataLoader, HeteroData
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Temporal early-warning analysis for PatientGNN E11"
    )
    # Required
    p.add_argument("--gnn_ckpt",    required=True,
                   help="Path to E11 PatientGNN best_model.pt checkpoint")
    p.add_argument("--graphs_path", required=True,
                   help="Path to patient_graphs.pt")

    # Optional
    p.add_argument("--out_dir",       default="analysis/early_prediction")
    p.add_argument("--device",        default="cuda:0")
    p.add_argument("--batch_size",    type=int,   default=64)
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--val_frac",      type=float, default=0.1)
    p.add_argument("--test_frac",     type=float, default=0.15)
    p.add_argument("--max_lead",      type=int,   default=3,
                   help="Maximum lead time k to evaluate (default: 3)")
    p.add_argument("--min_positives", type=int,   default=10,
                   help="Min positive cases per label to include in AUROC (default: 10)")

    # Model hyperparameters — must match E11 training
    p.add_argument("--hidden_dim",              type=int,   default=128)
    p.add_argument("--kg_embed_dim",            type=int,   default=128)
    p.add_argument("--num_gnn_layers",          type=int,   default=2)
    p.add_argument("--num_transformer_layers",  type=int,   default=2)
    p.add_argument("--nhead",                   type=int,   default=4)
    p.add_argument("--pe_dim",                  type=int,   default=4)
    p.add_argument("--alpha",                   type=float, default=0.8)
    p.add_argument("--dropout",                 type=float, default=0.3)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Graph preprocessing (identical to train.py / analyse_per_label.py)
# ---------------------------------------------------------------------------

def load_and_preprocess(args):
    """
    Load patient_graphs.pt, apply normalisation and leakage prevention
    (identical to train.py), then return the test split.

    Returns
    -------
    test_graphs : list[HeteroData]   preprocessed, leakage-fixed
    label_vocab : list[str]          275 SNOMED concept ID strings
    vocab_sizes : dict               {'cond', 'proc', 'drug'} → int
    """
    log.info("Loading %s …", args.graphs_path)
    graphs = torch.load(args.graphs_path, weights_only=False)
    graphs = [g for g in graphs if hasattr(g, "y_diagnosis")]
    log.info("  %d graphs with y_diagnosis", len(graphs))

    graphs_dir = Path(args.graphs_path).parent

    # Label vocabulary
    with open(graphs_dir / "label_vocab.json") as f:
        label_vocab = json.load(f)
    log.info("  %d diagnosis labels", len(label_vocab))

    # Concept vocabulary sizes (embedding table dimensions)
    with open(graphs_dir / "concept_vocab.json") as f:
        concept_vocab = json.load(f)

    def _vocab_size(domain):
        d = concept_vocab.get(domain, {})
        return max(int(v) for v in d.values()) + 1 if d else 0

    vocab_sizes = {
        "cond": _vocab_size("condition"),
        "proc": _vocab_size("procedure"),
        "drug": _vocab_size("drug"),
    }
    log.info("  Vocab sizes — cond: %d  proc: %d  drug: %d",
             vocab_sizes["cond"], vocab_sizes["proc"], vocab_sizes["drug"])

    # Normalise visit features (columns 3-5 only, same as train.py)
    all_x    = torch.cat([g["visit"].x for g in graphs], dim=0)
    mean4, std4 = float(all_x[:, 4].mean()), float(all_x[:, 4].std()) + 1e-6
    mean5, std5 = float(all_x[:, 5].mean()), float(all_x[:, 5].std()) + 1e-6
    for g in graphs:
        x = g["visit"].x.clone()
        x[:, 3] = torch.log1p(x[:, 3])
        x[:, 4] = (x[:, 4] - mean4) / std4
        x[:, 5] = (x[:, 5] - mean5) / std5
        g["visit"].x = x

    # Leakage prevention: remove last-visit condition edges (same as train.py)
    for g in graphs:
        last_v = g["visit"].x.size(0) - 1
        ei = g["visit", "has_condition", "condition"].edge_index
        if ei.numel() > 0:
            g["visit", "has_condition", "condition"].edge_index = ei[:, ei[0] != last_v]

    # Deterministic test split (same RNG state as E11 training)
    rng    = np.random.default_rng(args.seed)
    idx    = rng.permutation(len(graphs))
    n_test = int(len(graphs) * args.test_frac)
    n_val  = int(len(graphs) * args.val_frac)
    test_idx    = idx[:n_test]
    test_graphs = [graphs[i] for i in test_idx]
    log.info("  Test split: %d patients", len(test_graphs))

    return test_graphs, label_vocab, vocab_sizes


# ---------------------------------------------------------------------------
# Graph truncation
# ---------------------------------------------------------------------------

def make_truncated_graph(g, n_keep):
    """
    Build a truncated copy of patient graph g, retaining only the first
    n_keep visit nodes.

    The new last visit (index n_keep-1) has its condition edges removed to
    prevent label leakage. Procedure and drug edges are retained for all
    n_keep visits (they are features, not labels).

    Concept node tensors (condition/procedure/drug .x) are kept intact —
    disconnected concept nodes have no effect on GNN message passing, and
    avoiding remapping keeps this fast.

    The y_diagnosis label (conditions at the ORIGINAL last visit) is carried
    over unchanged — it is the evaluation target across all lead times.

    Parameters
    ----------
    g      : HeteroData  already preprocessed (normalised + leakage-fixed)
    n_keep : int         number of visit nodes to retain (>= 2)

    Returns
    -------
    HeteroData  truncated graph ready for model inference
    """
    assert n_keep >= 2, f"n_keep must be >= 2, got {n_keep}"

    new_g  = HeteroData()
    last_v = n_keep - 1   # new last visit index (prediction target)

    # Visit node features — first n_keep rows (already normalised)
    new_g["visit"].x = g["visit"].x[:n_keep]

    # Temporal edges: 0→1, 1→2, …, n_keep-2 → n_keep-1
    if n_keep > 1:
        src = torch.arange(n_keep - 1, dtype=torch.long)
        new_g["visit", "temporal_next", "visit"].edge_index = torch.stack(
            [src, src + 1]
        )
    else:
        new_g["visit", "temporal_next", "visit"].edge_index = torch.zeros(
            2, 0, dtype=torch.long
        )

    # Concept edges — filter and apply leakage prevention
    for domain, edge_rel in [
        ("condition", "has_condition"),
        ("procedure", "has_procedure"),
        ("drug",      "has_drug"),
    ]:
        orig_ei = g["visit", edge_rel, domain].edge_index
        orig_cx = g[domain].x

        if orig_ei.numel() == 0:
            new_g[domain].x = orig_cx
            new_g["visit", edge_rel, domain].edge_index = torch.zeros(
                2, 0, dtype=torch.long
            )
            continue

        if edge_rel == "has_condition":
            # Retain condition edges from visits 0 .. last_v-1 only.
            # This simultaneously excludes visits >= n_keep (out of range)
            # and the new last visit's condition edges (leakage prevention).
            mask = orig_ei[0] < last_v
        else:
            # Procedures and drugs: retain edges from all n_keep visits
            mask = orig_ei[0] < n_keep

        new_g[domain].x = orig_cx           # concept features unchanged
        new_g["visit", edge_rel, domain].edge_index = orig_ei[:, mask]

    # Carry graph-level attributes
    new_g.y_diagnosis = g.y_diagnosis       # ORIGINAL last-visit label (evaluation target)
    if hasattr(g, "subject_id"):
        new_g.subject_id = g.subject_id

    return new_g


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def build_model(args, vocab_sizes, num_labels, device):
    """Load PatientGNN from E11 checkpoint."""
    gnn_src = Path(__file__).parent.parent / "src" / "gnn"
    if str(gnn_src) not in sys.path:
        sys.path.insert(0, str(gnn_src))
    from model import PatientGNN   # noqa: E402

    model = PatientGNN.from_config(
        no_kg=True,
        cond_vocab_size=vocab_sizes["cond"],
        proc_vocab_size=vocab_sizes["proc"],
        drug_vocab_size=vocab_sizes["drug"],
        kg_embed_dim=args.kg_embed_dim,
        hidden_dim=args.hidden_dim,
        num_gnn_layers=args.num_gnn_layers,
        num_transformer_layers=args.num_transformer_layers,
        nhead=args.nhead,
        pe_dim=args.pe_dim,
        alpha=args.alpha,
        dropout=args.dropout,
        visit_feat_dim=7,
        num_labels=num_labels,
    ).to(device)

    ckpt = torch.load(args.gnn_ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    log.info("Loaded checkpoint — epoch %d, val AUROC %.4f",
             ckpt.get("epoch", -1),
             ckpt.get("val_metrics", {}).get("auroc_macro", float("nan")))
    return model


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_inference(graphs_list, model, device, batch_size, desc="Inference"):
    """
    Run diagnosis inference on a list of HeteroData graphs.

    Returns
    -------
    y_true : np.ndarray  [N, num_labels]
    y_prob : np.ndarray  [N, num_labels]
    """
    loader = DataLoader(graphs_list, batch_size=batch_size,
                        shuffle=False, num_workers=0)
    y_true_all, y_prob_all = [], []

    for batch in tqdm(loader, desc=desc, leave=False):
        batch   = batch.to(device)
        out     = model(batch, task="diagnosis")
        logits  = out["diagnosis"]                          # [B, num_labels]
        labels  = batch.y_diagnosis.view(logits.shape)      # [B, num_labels]
        y_true_all.append(labels.cpu().float().numpy())
        y_prob_all.append(torch.sigmoid(logits).cpu().float().numpy())

    y_true = np.concatenate(y_true_all, axis=0)
    y_prob = np.concatenate(y_prob_all, axis=0)
    y_prob = np.nan_to_num(y_prob, nan=0.5)
    return y_true, y_prob


def per_label_auroc(y_true, y_prob, min_positives):
    """
    Per-label AUROC. Returns array [num_labels] with np.nan
    where a label has fewer than min_positives positive cases.
    """
    n_labels = y_true.shape[1]
    aurocs   = np.full(n_labels, np.nan)
    for i in range(n_labels):
        if y_true[:, i].sum() >= min_positives:
            try:
                aurocs[i] = roc_auc_score(y_true[:, i], y_prob[:, i])
            except Exception:
                pass
    return aurocs


# ---------------------------------------------------------------------------
# Lead-time analysis
# ---------------------------------------------------------------------------

def run_lead_analysis(test_graphs, model, device, args):
    """
    Evaluate PatientGNN at each lead time k = 0 .. max_lead.

    Returns
    -------
    results : dict  {k: {'auroc': np.ndarray, 'n_valid': int,
                          'y_true': np.ndarray, 'y_prob': np.ndarray}}
    """
    results = {}

    for k in range(args.max_lead + 1):
        log.info("── Lead k=%d ──", k)

        if k == 0:
            # Standard evaluation — use preprocessed graphs directly
            valid_graphs = test_graphs
            skip         = 0
        else:
            # Build truncated graphs: keep the first N-k visit nodes
            valid_graphs, skip = [], 0
            for g in test_graphs:
                n_visits = g["visit"].x.size(0)
                n_keep   = n_visits - k
                if n_keep < 2:          # need ≥2 visit nodes for meaningful input
                    skip += 1
                    continue
                valid_graphs.append(make_truncated_graph(g, n_keep))

            log.info("  %d / %d patients have enough visits (skipped %d with N ≤ %d)",
                     len(valid_graphs), len(test_graphs), skip, k + 1)

        if not valid_graphs:
            log.warning("  No valid graphs at lead k=%d — skipping", k)
            continue

        y_true, y_prob = run_inference(
            valid_graphs, model, device, args.batch_size,
            desc=f"Lead k={k}"
        )
        auroc = per_label_auroc(y_true, y_prob, args.min_positives)

        n_valid_auroc = int(np.sum(~np.isnan(auroc)))
        log.info("  Macro-AUROC: %.4f  (%d labels with ≥%d positives evaluated)",
                 np.nanmean(auroc), n_valid_auroc, args.min_positives)

        results[k] = {
            "auroc":   auroc,
            "n_valid": len(valid_graphs),
            "y_true":  y_true,
            "y_prob":  y_prob,
        }

    return results


# ---------------------------------------------------------------------------
# Output: CSV and figures
# ---------------------------------------------------------------------------

def save_outputs(results, label_vocab, args):
    """Write CSV and all figures to args.out_dir."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(args.out_dir, exist_ok=True)
    lead_times = sorted(results.keys())
    num_labels = len(label_vocab)

    # Summary stats for each lead
    macro_aurocs = [np.nanmean(results[k]["auroc"]) for k in lead_times]
    macro_stds   = [np.nanstd(results[k]["auroc"])  for k in lead_times]
    n_valids     = [results[k]["n_valid"]            for k in lead_times]
    n0           = results[0]["n_valid"]
    pct_retained = [100.0 * n / n0 for n in n_valids]

    # ── 1. CSV: per-label AUROC at each lead time ─────────────────────────
    csv_path = os.path.join(args.out_dir, "early_prediction_results.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["snomed_code"] + [f"auroc_k{k}" for k in lead_times]
                   + [f"n_patients_k{k}" for k in lead_times])
        for i in range(num_labels):
            auroc_row = [
                f"{results[k]['auroc'][i]:.4f}" if not np.isnan(results[k]["auroc"][i]) else "nan"
                for k in lead_times
            ]
            n_row = [str(results[k]["n_valid"]) for k in lead_times]
            w.writerow([label_vocab[i]] + auroc_row + n_row)
    log.info("Saved %s", csv_path)

    # ── 2. Primary figure: macro-AUROC vs lead time ───────────────────────
    fig, ax1 = plt.subplots(figsize=(8, 5))

    ax1.errorbar(lead_times, macro_aurocs, yerr=macro_stds,
                 marker="o", linewidth=2.5, markersize=8, capsize=6,
                 color="#2c7bb6", label="Macro-AUROC (±1 std across labels)")
    # Annotate each point
    for k, auroc in zip(lead_times, macro_aurocs):
        ax1.annotate(f"{auroc:.4f}",
                     xy=(k, auroc), xytext=(0, 10),
                     textcoords="offset points", ha="center", fontsize=10,
                     color="#2c7bb6", fontweight="bold")

    ax1.set_xlabel("Lead time k  (number of recent visits withheld from input)",
                   fontsize=12)
    ax1.set_ylabel("Macro-AUROC (mean over 275 labels)", fontsize=12,
                   color="#2c7bb6")
    ax1.tick_params(axis="y", labelcolor="#2c7bb6")
    ax1.set_xticks(lead_times)
    ax1.set_ylim(max(0.4, min(macro_aurocs) - 0.06),
                 min(1.0,  max(macro_aurocs) + 0.06))
    ax1.grid(True, alpha=0.3)

    # Secondary axis: patient retention
    ax2 = ax1.twinx()
    ax2.bar(lead_times, pct_retained, alpha=0.18, color="steelblue",
            width=0.35, label=f"% patients with N > k+1 visits")
    ax2.set_ylabel("% test patients retained at each lead", fontsize=11,
                   color="steelblue")
    ax2.tick_params(axis="y", labelcolor="steelblue")
    ax2.set_ylim(0, 130)

    lines1, lbl1 = ax1.get_legend_handles_labels()
    lines2, lbl2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, lbl1 + lbl2, loc="lower left", fontsize=9)

    ax1.set_title(
        "PatientGNN E11 — Predictive Performance vs Lead Time\n"
        "k=0: standard evaluation  |  k>0: k most-recent visits withheld",
        fontsize=11
    )

    plt.tight_layout()
    path = os.path.join(args.out_dir, "fig_lead_vs_auroc.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info("Saved %s", path)

    # ── 3. Heatmap: per-condition AUROC decay (top-50 by frequency) ────────
    freq      = results[0]["y_true"].mean(axis=0)          # [275]
    top50_idx = np.argsort(-freq)[:50]

    heatmap = np.full((len(top50_idx), len(lead_times)), np.nan)
    for col_j, k in enumerate(lead_times):
        auroc_k = results[k]["auroc"]
        for row_i, label_i in enumerate(top50_idx):
            if label_i < len(auroc_k):
                heatmap[row_i, col_j] = auroc_k[label_i]

    fig_h = max(12, len(top50_idx) * 0.28)
    fig_w = max(5, len(lead_times) * 2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(heatmap, aspect="auto", cmap="RdYlGn",
                   vmin=0.5, vmax=1.0, interpolation="nearest")
    plt.colorbar(im, ax=ax, label="AUROC", fraction=0.025, pad=0.02)

    ax.set_xticks(range(len(lead_times)))
    ax.set_xticklabels([f"k={k}" for k in lead_times], fontsize=12)
    ax.set_yticks(range(len(top50_idx)))
    ax.set_yticklabels(
        [f"{label_vocab[idx]}  (prev={freq[idx]:.3f})" for idx in top50_idx],
        fontsize=7
    )
    ax.set_xlabel("Lead time k", fontsize=12)
    ax.set_ylabel("SNOMED condition code  (label prevalence in test set)", fontsize=11)
    ax.set_title("Per-condition AUROC vs Lead Time  —  top-50 conditions by prevalence",
                 fontsize=12)

    # Cell annotations
    for row_i in range(len(top50_idx)):
        for col_j in range(len(lead_times)):
            val = heatmap[row_i, col_j]
            if not np.isnan(val):
                txt_col = "white" if (val < 0.62 or val > 0.96) else "black"
                ax.text(col_j, row_i, f"{val:.2f}",
                        ha="center", va="center", fontsize=6, color=txt_col)

    plt.tight_layout()
    path = os.path.join(args.out_dir, "fig_lead_heatmap.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info("Saved %s", path)

    # ── 4. Line plot: AUROC decay for top-15 individual conditions ────────
    top15_idx = top50_idx[:15]
    cmap15    = plt.cm.get_cmap("tab20", len(top15_idx))

    fig, ax = plt.subplots(figsize=(9, 6))
    for row_i, label_i in enumerate(top15_idx):
        ks, vals = [], []
        for k in lead_times:
            v = results[k]["auroc"][label_i]
            if not np.isnan(v):
                ks.append(k)
                vals.append(v)
        if vals:
            ax.plot(ks, vals, marker="o", linewidth=1.8,
                    color=cmap15(row_i), alpha=0.85,
                    label=f"{label_vocab[label_i][:24]}  ({freq[label_i]:.3f})")

    ax.set_xlabel("Lead time k", fontsize=12)
    ax.set_ylabel("Per-label AUROC", fontsize=12)
    ax.set_xticks(lead_times)
    ax.set_title("AUROC Decay by Lead Time — Top 15 Conditions by Prevalence",
                 fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7)
    plt.tight_layout()
    path = os.path.join(args.out_dir, "fig_lead_decay_top15.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", path)

    # ── Console summary ───────────────────────────────────────────────────
    auroc_0 = macro_aurocs[0]
    log.info("\n=== Early Prediction Summary ===")
    log.info("  %-6s  %-12s  %-8s  %-12s  %-16s  %-18s",
             "Lead k", "Macro-AUROC", "Std", "N patients",
             "% retained", "% of k=0 AUROC")
    for i, k in enumerate(lead_times):
        pct_auroc = 100.0 * macro_aurocs[i] / auroc_0 if auroc_0 > 0 else 0.0
        log.info("  k=%-4d  %-12.4f  %-8.4f  %-12d  %-16.1f  %-18.1f",
                 k, macro_aurocs[i], macro_stds[i],
                 n_valids[i], pct_retained[i], pct_auroc)

    log.info("\n  Interpretation:")
    for i, k in enumerate(lead_times[1:], start=1):
        drop = auroc_0 - macro_aurocs[i]
        log.info("  k=%d: AUROC drops by %.4f from standard (%.1f%% of k=0 retained)",
                 k, drop, 100.0 * macro_aurocs[i] / auroc_0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    # Load and preprocess
    test_graphs, label_vocab, vocab_sizes = load_and_preprocess(args)
    num_labels = len(label_vocab)

    # Load model
    model = build_model(args, vocab_sizes, num_labels, device)

    # Run lead-time analysis
    results = run_lead_analysis(test_graphs, model, device, args)

    if not results:
        log.error("No results produced — check that patient graphs have enough visits.")
        sys.exit(1)

    # Save outputs
    save_outputs(results, label_vocab, args)

    log.info("\nAll outputs written to: %s", args.out_dir)


if __name__ == "__main__":
    main()
