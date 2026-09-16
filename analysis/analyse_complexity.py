"""
analyse_complexity.py
=====================
Analysis 3: Patient complexity subgroup — does PatientGNN's advantage
scale with how complex a patient's clinical history is?

Complexity is defined as the total number of unique concept nodes in a
patient's instance graph:
    complexity = |condition nodes| + |procedure nodes| + |drug nodes|

This is a direct measure of how "personalised" the PatientGNN graph is.
Simple patients (Q1) have few unique codes → small, sparse graph.
Complex patients (Q4) have many unique codes → large, dense graph.

Hypothesis: PatientGNN's AUROC advantage over TRANS grows with complexity.
TRANS uses the same vocabulary-level graph for every patient — it does not
benefit from a patient having more unique codes. PatientGNN builds a
patient-specific instance graph that gets richer the more codes a patient has.

Outputs (in --out_dir):
  fig_complexity_auroc.png       Main result: AUROC by quartile, both models
  fig_complexity_distribution.png Distribution of complexity and visits per quartile
  fig_complexity_scatter.png     Scatter: complexity vs per-patient recall@10
  complexity_results.csv         Quartile-level summary statistics
  complexity_per_patient.csv     Per-patient: complexity, n_visits, recall@10

Run AFTER analyse_per_label.py (uses its CSV for TRANS reference line):
  python analyse_complexity.py \\
    --gnn_ckpt      logs/e11/checkpoints/best_model.pt \\
    --graphs_path   data/processed/patient_graphs.pt \\
    --per_label_csv analysis/per_label/per_label_results.csv \\
    --out_dir       analysis/complexity \\
    --device        cuda:1

Can also run standalone (TRANS reference line omitted if CSV absent):
  python analyse_complexity.py \\
    --gnn_ckpt    logs/e11/checkpoints/best_model.pt \\
    --graphs_path data/processed/patient_graphs.pt \\
    --out_dir     analysis/complexity \\
    --device      cuda:1
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
from sklearn.metrics import roc_auc_score
from torch_geometric.data import DataLoader
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
    p = argparse.ArgumentParser(description="Patient complexity subgroup analysis")

    p.add_argument("--gnn_ckpt",    required=True,
                   help="PatientGNN E11 checkpoint: logs/e11/checkpoints/best_model.pt")
    p.add_argument("--graphs_path", required=True,
                   help="patient_graphs.pt")
    p.add_argument("--per_label_csv", default=None,
                   help="Optional: per_label_results.csv from analyse_per_label.py "
                        "(used to draw TRANS AUROC reference line)")

    p.add_argument("--out_dir",    default="analysis/complexity")
    p.add_argument("--device",     default="cuda:0")
    p.add_argument("--batch_size", type=int,   default=128)
    p.add_argument("--seed",       type=int,   default=42)
    p.add_argument("--val_frac",   type=float, default=0.1)
    p.add_argument("--test_frac",  type=float, default=0.15)

    # PatientGNN hparams — must match E11 training
    p.add_argument("--hidden_dim",             type=int,   default=128)
    p.add_argument("--kg_embed_dim",           type=int,   default=128)
    p.add_argument("--num_gnn_layers",         type=int,   default=2)
    p.add_argument("--num_transformer_layers", type=int,   default=2)
    p.add_argument("--nhead",                  type=int,   default=4)
    p.add_argument("--pe_dim",                 type=int,   default=4)
    p.add_argument("--alpha",                  type=float, default=0.8)
    p.add_argument("--dropout",                type=float, default=0.3)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Complexity helpers
# ---------------------------------------------------------------------------

def graph_complexity(g) -> int:
    """Total unique concept nodes: conditions + procedures + drugs."""
    total = 0
    for ntype in ("condition", "procedure", "drug"):
        if ntype in g.node_types:
            total += g[ntype].x.size(0)
    return total


def graph_n_visits(g) -> int:
    return g["visit"].x.size(0)


# ---------------------------------------------------------------------------
# GNN inference
# ---------------------------------------------------------------------------

def run_gnn_inference(args, device):
    """
    Load PatientGNN E11, run on test split.
    Returns:
      y_true       [N, 275]  ground-truth labels
      y_prob       [N, 275]  predicted probabilities
      complexity   [N]       concept node count per patient
      n_visits     [N]       visit count per patient
      label_vocab  [275]     SNOMED code strings
    """
    log.info("=== PatientGNN E11 inference (complexity analysis) ===")

    gnn_src = Path(__file__).parent.parent / "src" / "gnn"
    if str(gnn_src) not in sys.path:
        sys.path.insert(0, str(gnn_src))
    from model import PatientGNN  # noqa

    # ---- Load graphs ---------------------------------------------------------
    log.info("Loading patient_graphs.pt …")
    graphs = torch.load(args.graphs_path, weights_only=False)
    graphs = [g for g in graphs if hasattr(g, "y_diagnosis")]
    log.info("  %d graphs with y_diagnosis", len(graphs))

    graphs_dir = Path(args.graphs_path).parent

    with open(graphs_dir / "label_vocab.json") as f:
        label_vocab = json.load(f)
    num_labels = len(label_vocab)

    with open(graphs_dir / "concept_vocab.json") as f:
        concept_vocab = json.load(f)

    def _vocab_size(domain):
        d = concept_vocab.get(domain, {})
        return max(int(v) for v in d.values()) + 1 if d else 0

    vocab_sizes = {k: _vocab_size(k) for k in ("condition", "procedure", "drug")}

    # ---- Normalise visit features (identical to train.py) -------------------
    all_x   = torch.cat([g["visit"].x for g in graphs], dim=0)
    mean4, std4 = all_x[:, 4].mean().item(), all_x[:, 4].std().item() + 1e-6
    mean5, std5 = all_x[:, 5].mean().item(), all_x[:, 5].std().item() + 1e-6
    for g in graphs:
        x = g["visit"].x.clone()
        x[:, 3] = torch.log1p(x[:, 3])
        x[:, 4] = (x[:, 4] - mean4) / std4
        x[:, 5] = (x[:, 5] - mean5) / std5
        g["visit"].x = x

    # Remove last-visit condition edges (prevent leakage)
    for g in graphs:
        n_v = g["visit"].x.size(0)
        ei  = g["visit", "has_condition", "condition"].edge_index
        if ei.numel() > 0:
            g["visit", "has_condition", "condition"].edge_index = ei[:, ei[0] != (n_v - 1)]

    # ---- Deterministic test split (seed=42, identical to E11 training) ------
    rng    = np.random.default_rng(args.seed)
    idx    = rng.permutation(len(graphs))
    n_test = int(len(graphs) * args.test_frac)
    n_val  = int(len(graphs) * args.val_frac)
    test_idx    = idx[:n_test]
    test_graphs = [graphs[i] for i in test_idx]
    log.info("  test=%d", len(test_graphs))

    # ---- Compute complexity and visit count BEFORE batching -----------------
    complexity_arr = np.array([graph_complexity(g) for g in test_graphs], dtype=np.int32)
    n_visits_arr   = np.array([graph_n_visits(g)   for g in test_graphs], dtype=np.int32)

    log.info("  Complexity — min=%d  median=%d  max=%d  mean=%.1f",
             complexity_arr.min(), int(np.median(complexity_arr)),
             complexity_arr.max(), complexity_arr.mean())
    log.info("  Visits     — min=%d  median=%d  max=%d  mean=%.1f",
             n_visits_arr.min(), int(np.median(n_visits_arr)),
             n_visits_arr.max(), n_visits_arr.mean())

    # ---- Build model ---------------------------------------------------------
    model = PatientGNN.from_config(
        no_kg=True,
        cond_vocab_size=vocab_sizes["condition"],
        proc_vocab_size=vocab_sizes["procedure"],
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
    log.info("  Loaded checkpoint epoch=%d  val_AUROC=%.4f",
             ckpt["epoch"], ckpt["val_metrics"].get("auroc_macro", float("nan")))

    # ---- Inference -----------------------------------------------------------
    loader = DataLoader(test_graphs, batch_size=args.batch_size,
                        shuffle=False, num_workers=4)
    y_true_list, y_prob_list = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="GNN inference"):
            batch  = batch.to(device)
            out    = model(batch, task="diagnosis")
            logits = out["diagnosis"]
            labels = batch.y_diagnosis.view(logits.shape)
            y_true_list.append(labels.cpu().float().numpy())
            y_prob_list.append(torch.sigmoid(logits).cpu().float().numpy())

    y_true = np.concatenate(y_true_list, axis=0)   # [N, 275]
    y_prob = np.nan_to_num(np.concatenate(y_prob_list, axis=0), nan=0.5)
    log.info("  Inference done: %d patients", y_true.shape[0])

    return y_true, y_prob, complexity_arr, n_visits_arr, label_vocab


# ---------------------------------------------------------------------------
# Per-patient metrics
# ---------------------------------------------------------------------------

def recall_at_k(y_true_row, y_prob_row, k=10):
    """Recall@K for a single patient."""
    n_pos = y_true_row.sum()
    if n_pos == 0:
        return np.nan
    top_k = np.argsort(-y_prob_row)[:k]
    return float(y_true_row[top_k].sum()) / float(n_pos)


def per_patient_recall(y_true, y_prob, k=10):
    """[N] recall@k, NaN where patient has no positive labels."""
    return np.array([recall_at_k(y_true[i], y_prob[i], k)
                     for i in range(len(y_true))])


def quartile_macro_auroc(y_true, y_prob, mask, min_positives=3):
    """
    Macro AUROC computed only over patients in mask.
    Returns (auroc, n_labels_evaluated).
    """
    yt = y_true[mask]
    yp = y_prob[mask]
    if len(yt) < 10:
        return float("nan"), 0
    aurocs = []
    for i in range(yt.shape[1]):
        pos = yt[:, i].sum()
        if pos >= min_positives and pos < len(yt):
            aurocs.append(roc_auc_score(yt[:, i], yp[:, i]))
    return (float(np.mean(aurocs)) if aurocs else float("nan")), len(aurocs)


# ---------------------------------------------------------------------------
# Analysis + plots
# ---------------------------------------------------------------------------

def analyse_and_plot(y_true, y_prob, complexity, n_visits,
                     label_vocab, per_label_csv, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    N = len(complexity)

    # ---- Quartile boundaries ------------------------------------------------
    q25, q50, q75 = np.percentile(complexity, [25, 50, 75])
    quartile_masks = [
        complexity <= q25,
        (complexity > q25) & (complexity <= q50),
        (complexity > q50) & (complexity <= q75),
        complexity > q75,
    ]
    quartile_labels = [
        f"Q1 simple\n(≤{int(q25)} nodes)",
        f"Q2\n({int(q25)+1}–{int(q50)} nodes)",
        f"Q3\n({int(q50)+1}–{int(q75)} nodes)",
        f"Q4 complex\n(>{int(q75)} nodes)",
    ]

    # ---- Per-patient recall@10 -----------------------------------------------
    recall10 = per_patient_recall(y_true, y_prob, k=10)

    # ---- Per-quartile stats --------------------------------------------------
    log.info("\n=== Complexity Quartile Summary ===")
    log.info("  %-22s  %7s  %10s  %10s  %10s  %10s",
             "Quartile", "N", "Mean cplx", "Mean vis", "AUROC", "Recall@10")

    rows = []
    gnn_aurocs = []
    for mask, qlabel in zip(quartile_masks, quartile_labels):
        n_q     = mask.sum()
        mc      = complexity[mask].mean()
        mv      = n_visits[mask].mean()
        auroc, n_eval = quartile_macro_auroc(y_true, y_prob, mask)
        r10     = np.nanmean(recall10[mask])
        gnn_aurocs.append(auroc)
        rows.append({
            "quartile":        qlabel.replace("\n", " "),
            "n_patients":      int(n_q),
            "mean_complexity": round(float(mc), 1),
            "mean_visits":     round(float(mv), 2),
            "gnn_auroc":       round(auroc, 4) if not np.isnan(auroc) else "nan",
            "gnn_recall_at10": round(r10,  4) if not np.isnan(r10)   else "nan",
            "n_labels_eval":   n_eval,
        })
        log.info("  %-22s  %7d  %10.1f  %10.2f  %10.4f  %10.4f",
                 qlabel.replace("\n", " "), n_q, mc, mv, auroc, r10)

    # ---- Load TRANS AUROC reference (from Analysis 1 CSV) -------------------
    trans_macro_auroc = None
    if per_label_csv and os.path.exists(per_label_csv):
        try:
            trans_aurocs = []
            with open(per_label_csv, newline="") as f:
                for row in csv.DictReader(f):
                    if row["trans_auroc"] != "nan":
                        trans_aurocs.append(float(row["trans_auroc"]))
            if trans_aurocs:
                trans_macro_auroc = float(np.mean(trans_aurocs))
                log.info("  TRANS macro AUROC (from per_label_results.csv): %.4f",
                         trans_macro_auroc)
        except Exception as e:
            log.warning("  Could not read TRANS AUROC from CSV: %s", e)

    # ---- Save CSV -----------------------------------------------------------
    csv_path = os.path.join(out_dir, "complexity_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    log.info("Saved %s", csv_path)

    # Per-patient CSV
    pp_csv = os.path.join(out_dir, "complexity_per_patient.csv")
    with open(pp_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["patient_idx", "complexity", "n_visits",
                    "recall_at10", "quartile"])
        q_labels_short = ["Q1", "Q2", "Q3", "Q4"]
        patient_q = np.zeros(N, dtype=int)
        for qi, mask in enumerate(quartile_masks):
            patient_q[mask] = qi + 1
        for i in range(N):
            w.writerow([i, int(complexity[i]), int(n_visits[i]),
                        f"{recall10[i]:.4f}" if not np.isnan(recall10[i]) else "nan",
                        f"Q{patient_q[i]}"])
    log.info("Saved %s", pp_csv)

    # =========================================================================
    # Figure 1: Main result — GNN AUROC by complexity quartile
    # =========================================================================
    fig, ax = plt.subplots(figsize=(9, 6))
    x     = np.arange(4)
    colors = ["#91bfdb", "#74add1", "#4575b4", "#313695"]

    bars = ax.bar(x, gnn_aurocs, color=colors, alpha=0.85, edgecolor="k",
                  linewidth=0.5, width=0.55, label="PatientGNN E11")

    # Annotate bar values
    for xi, v in enumerate(gnn_aurocs):
        if not np.isnan(v):
            ax.text(xi, v + 0.001, f"{v:.4f}", ha="center", va="bottom",
                    fontsize=10, fontweight="bold")

    # TRANS reference line
    if trans_macro_auroc is not None:
        ax.axhline(trans_macro_auroc, color="#d73027", linewidth=2,
                   linestyle="--",
                   label=f"TRANS E7 overall  ({trans_macro_auroc:.4f})")

    ax.set_xticks(x)
    ax.set_xticklabels(quartile_labels, fontsize=10)
    ax.set_ylabel("Macro AUROC (per-label mean)", fontsize=12)
    ax.set_title(
        "PatientGNN AUROC by patient complexity quartile\n"
        "Complexity = unique concept nodes (conditions + procedures + drugs)",
        fontsize=11, fontweight="bold"
    )
    ax.set_ylim(max(0.5, min(v for v in gnn_aurocs if not np.isnan(v)) - 0.02), 1.0)
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)

    # Annotate patient counts
    for xi, row in enumerate(rows):
        ax.text(xi, ax.get_ylim()[0] + 0.003,
                f"n={row['n_patients']:,}", ha="center", va="bottom",
                fontsize=8, color="grey")

    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_complexity_auroc.png"), dpi=150)
    plt.close(fig)
    log.info("Saved fig_complexity_auroc.png")

    # =========================================================================
    # Figure 2: Distribution — complexity and visit count per quartile
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, arr, ylabel, title in [
        (axes[0], complexity, "Concept nodes (conditions+procedures+drugs)",
         "Patient complexity distribution by quartile"),
        (axes[1], n_visits,  "Number of visits",
         "Visit count distribution by quartile"),
    ]:
        groups = [arr[mask] for mask in quartile_masks]
        bp = ax.boxplot(groups, labels=[f"Q{i+1}" for i in range(4)],
                        patch_artist=True, medianprops={"color": "black", "linewidth": 2},
                        showfliers=False)
        palette = ["#91bfdb", "#74add1", "#4575b4", "#313695"]
        for patch, c in zip(bp["boxes"], palette):
            patch.set_facecolor(c)
            patch.set_alpha(0.7)
        ax.set_xlabel("Complexity quartile", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=11)
        ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_complexity_distribution.png"), dpi=150)
    plt.close(fig)
    log.info("Saved fig_complexity_distribution.png")

    # =========================================================================
    # Figure 3: Scatter — complexity vs recall@10 (per patient)
    # =========================================================================
    valid = ~np.isnan(recall10)
    # Subsample for plotting if large
    if valid.sum() > 5000:
        rng_plot = np.random.default_rng(0)
        plot_idx = rng_plot.choice(np.where(valid)[0], size=5000, replace=False)
    else:
        plot_idx = np.where(valid)[0]

    fig, ax = plt.subplots(figsize=(9, 6))
    sc = ax.scatter(
        complexity[plot_idx], recall10[plot_idx],
        c=n_visits[plot_idx], cmap="viridis",
        alpha=0.3, s=10, linewidths=0,
    )
    plt.colorbar(sc, ax=ax, label="Number of visits")

    # Trend line
    from scipy import stats as scipy_stats
    slope, intercept, r, p, _ = scipy_stats.linregress(
        complexity[plot_idx], recall10[plot_idx]
    )
    xfit = np.linspace(complexity[plot_idx].min(), complexity[plot_idx].max(), 200)
    ax.plot(xfit, slope * xfit + intercept, "r-", linewidth=2,
            label=f"Trend (r={r:.3f}, p={p:.3f})")

    ax.set_xlabel("Patient complexity (unique concept nodes)", fontsize=12)
    ax.set_ylabel("Recall@10 (per patient)", fontsize=12)
    ax.set_title(
        "Per-patient Recall@10 vs graph complexity\n"
        "(colour = number of visits; trend line across all test patients)",
        fontsize=11
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_complexity_scatter.png"), dpi=150)
    plt.close(fig)
    log.info("Saved fig_complexity_scatter.png")

    # ---- Summary ------------------------------------------------------------
    log.info("\n=== Key result ===")
    if not any(np.isnan(v) for v in gnn_aurocs):
        q1_q4_gap = gnn_aurocs[-1] - gnn_aurocs[0]
        log.info("  GNN AUROC Q1→Q4 gain: %+.4f", q1_q4_gap)
        if q1_q4_gap > 0:
            log.info("  ✓ Hypothesis supported: PatientGNN improves with complexity")
        else:
            log.info("  ✗ No complexity benefit detected — inspect distribution")
    if trans_macro_auroc:
        log.info("  GNN Q4 vs TRANS overall: %+.4f",
                 gnn_aurocs[-1] - trans_macro_auroc)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    y_true, y_prob, complexity, n_visits, label_vocab = \
        run_gnn_inference(args, device)

    analyse_and_plot(
        y_true, y_prob, complexity, n_visits,
        label_vocab,
        per_label_csv=args.per_label_csv,
        out_dir=args.out_dir,
    )

    log.info("\nAll outputs written to: %s", args.out_dir)


if __name__ == "__main__":
    main()
