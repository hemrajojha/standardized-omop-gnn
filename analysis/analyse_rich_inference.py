#!/usr/bin/env python3
"""
analyse_rich_inference.py
=========================
Single-script comprehensive analysis combining:

  PATH A  — Training dynamics from all saved metrics.json files
            (cap10, sqrt5, none variants of Proposed GNN; TRANS)

  PATH B  — Full model inference on best-model test set
            Saves y_true / y_prob arrays, then computes:
              • Per-label AUROC, AUPRC, TP/FP/FN/TN, F1, optimal threshold
              • Calibration (reliability diagram, ECE)
              • Per-patient top-K accuracy and precision
              • Rare vs common label group breakdowns
              • Precision-Recall curves (per label group)
              • Cross-reference with TRANS per-label CSV for head-to-head comparison

Mirrors the EXACT same data pipeline as src/gnn/train.py
(same seed, split fractions, normalisation, last-visit edge removal).

Run from OMOP_GNN_PROJECT root on HPC:
    python analysis/analyse_rich_inference.py \\
        --graphs_path data/processed/patient_graphs_diag.pt \\
        --vocab_path  data/processed/concept_vocab.json \\
        --ckpt_path   logs/e11/checkpoints/best_model.pt \\
        --out_dir     analysis/rich_inference \\
        --device      cuda:1

Optional flags:
    --e10_metrics  logs/e10/results/metrics.json
    --e11_metrics  logs/e11/results/metrics.json
    --e12_metrics  logs/e12/results/metrics.json
    --trans_results /path/to/TRANS/logs/results_TRANS_omop.json
    --per_label_csv analysis/per_label/per_label_results.csv
    --threshold 0.5
    --skip_inference   (skip GPU inference, load saved y_true/y_prob.npy if present)
"""

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ── add src/gnn to path so we can import PatientGNN and helpers ──────────────
SCRIPT_DIR   = Path(__file__).resolve().parent          # …/analysis/
PROJECT_ROOT = SCRIPT_DIR.parent                        # …/OMOP_GNN_PROJECT/
GNN_DIR      = PROJECT_ROOT / "src" / "gnn"
sys.path.insert(0, str(GNN_DIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve,
)
from torch_geometric.data import DataLoader


# ── Lazy import model (only needed for inference) ────────────────────────────
def import_model():
    from model import PatientGNN
    return PatientGNN


# ── helpers ──────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    # Data
    p.add_argument("--graphs_path",   required=True)
    p.add_argument("--vocab_path",    default=None)
    p.add_argument("--ckpt_path",     default=None,
                   help="E11 checkpoint; auto-detected from logs/e11 if not given")
    p.add_argument("--out_dir",       default="analysis/rich_inference")
    # Training history (Path A)
    p.add_argument("--e10_metrics",   default="logs/e10/results/metrics.json")
    p.add_argument("--e11_metrics",   default="logs/e11/results/metrics.json")
    p.add_argument("--e12_metrics",   default="logs/e12/results/metrics.json")
    p.add_argument("--trans_results", default=None,
                   help="Path to results_TRANS_omop.json")
    p.add_argument("--extra_metrics", action="append", default=[],
                   metavar="LABEL:PATH",
                   help="Additional experiment metrics, e.g. "
                        "'Proposed GNN (KG):logs/e13/results/metrics.json'. "
                        "Repeat for multiple experiments.")
    # Per-label CSV (from existing Analysis 1)
    p.add_argument("--per_label_csv", default="analysis/per_label/per_label_results.csv")
    # Inference
    p.add_argument("--device",        default="cuda")
    p.add_argument("--batch_size",    type=int, default=128)
    p.add_argument("--threshold",     type=float, default=0.5)
    p.add_argument("--seed",          type=int, default=42)
    p.add_argument("--val_frac",      type=float, default=0.10)
    p.add_argument("--test_frac",     type=float, default=0.15)
    p.add_argument("--skip_inference", action="store_true",
                   help="Skip GPU pass; load saved y_true.npy / y_prob.npy")
    return p.parse_args()


def load_json(path):
    path = Path(path)
    if not path.exists():
        log.warning("File not found: %s", path)
        return None
    with open(path) as f:
        return json.load(f)


def load_per_label_csv(path):
    rows = []
    path = Path(path)
    if not path.exists():
        log.warning("Per-label CSV not found: %s", path)
        return rows
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append({
                "code":     r["snomed_code"],
                "freq":     float(r["label_freq"]),
                "trans":    float(r["trans_auroc"]),
                "gnn":      float(r["gnn_auroc"]),
                "gap":      float(r["gap"]),
                "gnn_wins": int(r["gnn_wins"]),
                "q":        int(r["freq_quartile"]),
            })
    return rows


# ── DATA PIPELINE (mirrors train.py exactly) ──────────────────────────────────

def load_and_prepare_graphs(graphs_path, vocab_path, seed, val_frac, test_frac):
    """Reproduce the EXACT load + normalise + split pipeline from train.py."""
    graphs_path = Path(graphs_path)
    graphs_dir  = graphs_path.parent

    log.info("Loading graphs from %s ...", graphs_path)
    graphs = torch.load(graphs_path, weights_only=False)
    log.info("  Loaded %d graphs total", len(graphs))

    # Filter to graphs with y_diagnosis
    graphs = [g for g in graphs if hasattr(g, "y_diagnosis")]
    log.info("  %d graphs have y_diagnosis", len(graphs))

    # Label vocab
    label_vocab_path = graphs_dir / "label_vocab.json"
    with open(label_vocab_path) as f:
        label_vocab = json.load(f)
    num_labels = len(label_vocab)
    log.info("  Labels: %d", num_labels)

    # Concept vocab
    vp = Path(vocab_path) if vocab_path else graphs_dir / "concept_vocab.json"
    with open(vp) as f:
        concept_vocab = json.load(f)
    def _vsz(domain):
        d = concept_vocab.get(domain, {})
        return max(int(v) for v in d.values()) + 1 if d else 0
    vocab_sizes = {
        "cond": _vsz("condition"),
        "proc": _vsz("procedure"),
        "drug": _vsz("drug"),
    }

    # Normalise visit features (cols 3-5) — same as train.py
    log.info("  Normalising visit features ...")
    all_x = torch.cat([g["visit"].x for g in graphs], dim=0)
    mean4, std4 = all_x[:, 4].mean().item(), all_x[:, 4].std().item() + 1e-6
    mean5, std5 = all_x[:, 5].mean().item(), all_x[:, 5].std().item() + 1e-6
    for g in graphs:
        x = g["visit"].x.clone()
        x[:, 3] = torch.log1p(x[:, 3])
        x[:, 4] = (x[:, 4] - mean4) / std4
        x[:, 5] = (x[:, 5] - mean5) / std5
        g["visit"].x = x

    # Remove last-visit->condition edges (label leakage prevention)
    log.info("  Removing last-visit condition edges ...")
    for g in graphs:
        n_visits = g["visit"].x.size(0)
        ei = g["visit", "has_condition", "condition"].edge_index
        if ei.numel() > 0:
            mask = ei[0] != (n_visits - 1)
            g["visit", "has_condition", "condition"].edge_index = ei[:, mask]

    # Deterministic split — MUST match train.py
    rng   = np.random.default_rng(seed)
    n     = len(graphs)
    idx   = rng.permutation(n)
    n_test = int(n * test_frac)
    n_val  = int(n * val_frac)
    test_idx  = idx[:n_test]
    val_idx   = idx[n_test:n_test + n_val]
    train_idx = idx[n_test + n_val:]
    train_graphs = [graphs[i] for i in train_idx]
    val_graphs   = [graphs[i] for i in val_idx]
    test_graphs  = [graphs[i] for i in test_idx]
    log.info("  Split -> train=%d  val=%d  test=%d", len(train_graphs), len(val_graphs), len(test_graphs))

    return train_graphs, val_graphs, test_graphs, label_vocab, vocab_sizes, num_labels


# ── INFERENCE ─────────────────────────────────────────────────────────────────

def run_inference(test_graphs, label_vocab, vocab_sizes, num_labels, ckpt_path, device, batch_size):
    """Load checkpoint and run test inference. Returns (y_true, y_prob)."""
    PatientGNN = import_model()

    ckpt = torch.load(ckpt_path, weights_only=False, map_location=device)
    saved_args = ckpt.get("args", {})
    log.info("  Checkpoint from epoch %d (val_auroc=%.4f)",
             ckpt.get("epoch", -1),
             ckpt.get("val_metrics", {}).get("auroc_macro", float("nan")))

    model = PatientGNN.from_config(
        no_kg=True,
        cond_vocab_size=vocab_sizes["cond"],
        proc_vocab_size=vocab_sizes["proc"],
        drug_vocab_size=vocab_sizes["drug"],
        kg_embed_dim=saved_args.get("kg_embed_dim", 128),
        hidden_dim=saved_args.get("hidden_dim", 128),
        num_gnn_layers=saved_args.get("num_gnn_layers", 2),
        num_transformer_layers=saved_args.get("num_transformer_layers", 2),
        nhead=saved_args.get("nhead", 4),
        pe_dim=saved_args.get("pe_dim", 4),
        alpha=saved_args.get("alpha", 0.8),
        dropout=saved_args.get("dropout", 0.3),
        visit_feat_dim=7,
        num_labels=num_labels,
    )
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device)
    model.eval()
    log.info("  Model loaded. Params: %s", f"{sum(p.numel() for p in model.parameters()):,}")

    loader = DataLoader(test_graphs, batch_size=batch_size, shuffle=False, num_workers=4)
    y_true_all, y_prob_all = [], []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch, task="diagnosis")
            logits = out["diagnosis"]
            labels = batch.y_diagnosis.view(logits.shape)
            y_true_all.append(labels.cpu().float().numpy())
            y_prob_all.append(torch.sigmoid(logits).cpu().float().numpy())

    y_true = np.concatenate(y_true_all, axis=0)
    y_prob = np.concatenate(y_prob_all, axis=0)
    y_prob = np.nan_to_num(y_prob, nan=0.5)
    log.info("  Inference done: y_true=%s  y_prob=%s", y_true.shape, y_prob.shape)
    return y_true, y_prob


# ── PER-LABEL METRICS ─────────────────────────────────────────────────────────

def compute_per_label_metrics(y_true, y_prob, label_vocab, threshold=0.5):
    """For each label: AUROC, AUPRC, TP/FP/FN/TN, F1 at threshold and optimal threshold."""
    n_labels = y_true.shape[1]
    results = []
    for i in range(n_labels):
        yt = y_true[:, i]
        yp = y_prob[:, i]
        n_pos = int(yt.sum())
        n_neg = int((1 - yt).sum())
        freq  = float(n_pos / len(yt))

        if n_pos == 0:
            results.append(None)
            continue

        auroc = float(roc_auc_score(yt, yp))
        auprc = float(average_precision_score(yt, yp))

        # At fixed threshold
        pred_t  = (yp >= threshold).astype(float)
        tp_t  = int((pred_t * yt).sum())
        fp_t  = int((pred_t * (1 - yt)).sum())
        fn_t  = int(((1 - pred_t) * yt).sum())
        tn_t  = int(((1 - pred_t) * (1 - yt)).sum())
        prec_t = tp_t / max(tp_t + fp_t, 1)
        rec_t  = tp_t / max(tp_t + fn_t, 1)
        f1_t   = 2 * prec_t * rec_t / max(prec_t + rec_t, 1e-9)

        # Optimal threshold (Youden J)
        fpr, tpr, thrs = roc_curve(yt, yp)
        j_idx    = int(np.argmax(tpr - fpr))
        opt_thr  = float(thrs[j_idx]) if len(thrs) > j_idx else threshold
        pred_opt = (yp >= opt_thr).astype(float)
        tp_o  = int((pred_opt * yt).sum())
        fp_o  = int((pred_opt * (1 - yt)).sum())
        fn_o  = int(((1 - pred_opt) * yt).sum())
        tn_o  = int(((1 - pred_opt) * (1 - yt)).sum())
        prec_o = tp_o / max(tp_o + fp_o, 1)
        rec_o  = tp_o / max(tp_o + fn_o, 1)
        f1_o   = 2 * prec_o * rec_o / max(prec_o + rec_o, 1e-9)

        results.append({
            "label_idx": i,
            "omop_code": label_vocab[i] if i < len(label_vocab) else str(i),
            "n_pos": n_pos,
            "n_neg": n_neg,
            "freq":  round(freq, 6),
            "auroc": round(auroc, 4),
            "auprc": round(auprc, 4),
            # fixed threshold
            "threshold": threshold,
            "tp": tp_t, "fp": fp_t, "fn": fn_t, "tn": tn_t,
            "precision": round(prec_t, 4),
            "recall":    round(rec_t, 4),
            "f1":        round(f1_t, 4),
            # optimal threshold
            "opt_threshold": round(opt_thr, 4),
            "opt_tp": tp_o, "opt_fp": fp_o, "opt_fn": fn_o, "opt_tn": tn_o,
            "opt_precision": round(prec_o, 4),
            "opt_recall":    round(rec_o, 4),
            "opt_f1":        round(f1_o, 4),
        })
    return results


# ── PER-PATIENT METRICS ───────────────────────────────────────────────────────

def compute_per_patient_metrics(y_true, y_prob, test_graphs, ks=(5, 10, 20)):
    """Per-patient: top-K hits, precision, recall, graph complexity."""
    n = len(y_true)
    records = []

    for i in range(n):
        yt = y_true[i]
        yp = y_prob[i]
        n_pos = int(yt.sum())
        if n_pos == 0:
            continue

        # Graph complexity (total unique concept nodes)
        g = test_graphs[i]
        n_visits    = int(g["visit"].x.size(0))
        n_cond_nodes = int(g["condition"].x.size(0)) if "condition" in g.node_types else 0
        n_proc_nodes = int(g["procedure"].x.size(0)) if "procedure" in g.node_types else 0
        n_drug_nodes = int(g["drug"].x.size(0))      if "drug"      in g.node_types else 0
        complexity   = n_cond_nodes + n_proc_nodes + n_drug_nodes

        top_idxs = np.argsort(-yp)
        rec = {"graph_idx": i, "n_pos": n_pos, "n_visits": n_visits, "complexity": complexity}

        for k in ks:
            top_k = top_idxs[:k]
            hits  = int(yt[top_k].sum())
            rec[f"hits@{k}"]      = hits
            rec[f"precision@{k}"] = round(hits / k, 4)
            rec[f"recall@{k}"]    = round(hits / n_pos, 4)

        # Max predicted probability for any true positive
        rec["max_tp_prob"]   = round(float(yp[yt == 1].max()), 4)
        # Mean predicted probability for true positives vs negatives
        rec["mean_tp_prob"]  = round(float(yp[yt == 1].mean()), 4)
        rec["mean_tn_prob"]  = round(float(yp[yt == 0].mean()), 4)

        records.append(rec)

    return records


# ── CALIBRATION ───────────────────────────────────────────────────────────────

def compute_calibration(y_true, y_prob, n_bins=15):
    """Reliability diagram data and Expected Calibration Error."""
    bins = np.linspace(0, 1, n_bins + 1)
    bin_centers, bin_accs, bin_confs, bin_counts = [], [], [], []

    flat_true = y_true.flatten()
    flat_prob = y_prob.flatten()

    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (flat_prob >= lo) & (flat_prob < hi)
        if mask.sum() == 0:
            continue
        acc  = float(flat_true[mask].mean())
        conf = float(flat_prob[mask].mean())
        cnt  = int(mask.sum())
        bin_centers.append(round((lo + hi) / 2, 4))
        bin_accs.append(round(acc, 4))
        bin_confs.append(round(conf, 4))
        bin_counts.append(cnt)

    total = sum(bin_counts)
    ece = float(sum(
        bin_counts[j] / total * abs(bin_accs[j] - bin_confs[j])
        for j in range(len(bin_counts))
    ))
    return {
        "ece":         round(ece, 5),
        "bin_centers": bin_centers,
        "bin_accs":    bin_accs,
        "bin_confs":   bin_confs,
        "bin_counts":  bin_counts,
    }


# ── PLOTS ─────────────────────────────────────────────────────────────────────

C_GNN   = "#2196F3"
C_TRANS = "#FF5722"
C_E10   = "#9C27B0"
C_E11   = "#2196F3"
C_E12   = "#FF9800"
C_IDEAL = "#888888"
EXTRA_COLORS = ["#4CAF50", "#00BCD4", "#795548", "#607D8B", "#E91E63", "#009688", "#FF5722", "#8BC34A"]

plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})


def plot_training_curves(e10, e11, e12, trans, out_dir, extra_expts=None):
    """4-panel training curves: loss and val AUROC for GNN variants vs TRANS.

    extra_expts: list of (data_dict, label, color) tuples for additional experiments.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("")

    expts = [
        (e10, "Proposed GNN (cap10)", C_E10),
        (e11, "Proposed GNN (sqrt5)", C_E11),
        (e12, "Proposed GNN (none)",  C_E12),
    ]
    if extra_expts:
        expts = expts + extra_expts
    all_curves = expts + [(trans, "TRANS", C_TRANS)]

    # ── (0,0) Val AUROC per epoch ──────────────────────────────────────────
    ax = axes[0, 0]
    for data, label, color in all_curves:
        if data is None:
            continue
        hist = data.get("history", [])
        if not hist or "val_score" not in hist[0]:
            continue  # TRANS has val_loss only, not val_score
        epochs = [h["epoch"] for h in hist]
        vals   = [h.get("val_score", float("nan")) for h in hist]
        ax.plot(epochs, vals, color=color, linewidth=2, label=label)
        best_e = data.get("best_epoch", -1)
        if best_e is not None and best_e >= 0:
            ax.axvline(best_e, color=color, linestyle=":", alpha=0.5)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Val AUROC")
    ax.set_title("")
    ax.legend(fontsize=9, loc="lower right")

    # ── (0,1) Train loss per epoch ─────────────────────────────────────────
    ax = axes[0, 1]
    for data, label, color in all_curves:
        if data is None:
            continue
        hist = data.get("history", [])
        if not hist:
            continue
        epochs = [h["epoch"] for h in hist]
        losses = [h.get("train_loss", float("nan")) for h in hist]
        ax.plot(epochs, losses, color=color, linewidth=2, label=label)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Train Loss (BCE)")
    ax.set_title("")
    ax.legend(fontsize=9, loc="upper right")

    # ── (1,0) Best epoch bar chart ────────────────────────────────────────
    ax = axes[1, 0]
    best_epochs = []
    for data, label, color in all_curves:
        if data is None:
            continue
        best_e = data.get("best_epoch", None)
        # Derive best epoch from min val_loss if not stored (e.g. TRANS)
        if best_e is None:
            hist = data.get("history", [])
            if hist and "val_loss" in hist[0]:
                best_e = min(hist, key=lambda h: h["val_loss"])["epoch"]
        if best_e is not None:
            best_epochs.append((label, best_e, color))
    names  = [b[0] for b in best_epochs]
    vals   = [b[1] for b in best_epochs]
    colors = [b[2] for b in best_epochs]
    bars = ax.bar(range(len(names)), vals, color=colors, alpha=0.85, width=0.5)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.3, str(v),
                ha="center", fontsize=10, fontweight="bold")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=9, rotation=15, ha="right")
    ax.set_ylabel("Best Epoch (early stopping)")
    ax.set_title("")

    # ── (1,1) Test metrics comparison bar chart ────────────────────────────
    ax = axes[1, 1]
    labels_bar, aurocs_bar, auprcs_bar, colors_bar = [], [], [], []
    for data, label, color in expts:
        if data is None:
            continue
        labels_bar.append(label)
        aurocs_bar.append(data.get("test", {}).get("auroc_macro", 0))
        auprcs_bar.append(data.get("test", {}).get("auprc_macro", 0))
        colors_bar.append(color)

    if trans:
        tm = trans.get("test_metrics", {})
        labels_bar.append("TRANS")
        aurocs_bar.append(tm.get("auroc_macro", 0))
        auprcs_bar.append(tm.get("auprc_macro", 0))
        colors_bar.append(C_TRANS)

    x = np.arange(len(labels_bar))
    w = 0.35
    b1 = ax.bar(x - w / 2, aurocs_bar, w, label="AUROC", alpha=0.85, color=colors_bar)
    b2 = ax.bar(x + w / 2, auprcs_bar, w, label="AUPRC", alpha=0.55, color=colors_bar, hatch="//")
    for bar, v in zip(list(b1) + list(b2), aurocs_bar + auprcs_bar):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels_bar, fontsize=9, rotation=15, ha="right")
    ax.set_ylim(0.28, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("")
    ax.legend(loc="upper left", bbox_to_anchor=(0, 1.12), ncol=2, fontsize=9, frameon=True)

    plt.tight_layout()
    fig.savefig(out_dir / "fig_training_curves.png", dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Saved fig_training_curves.png")


def plot_calibration(cal_data, out_dir):
    fig, ax = plt.subplots(figsize=(7, 6))
    bins_c = cal_data["bin_centers"]
    accs   = cal_data["bin_accs"]
    confs  = cal_data["bin_confs"]
    counts = cal_data["bin_counts"]

    total = sum(counts)
    widths = [0.065] * len(bins_c)
    ax.bar(confs, accs, width=widths, alpha=0.7, color=C_GNN,
           label="Actual accuracy", align="center")
    ax.plot([0, 1], [0, 1], linestyle="--", color=C_IDEAL, linewidth=1.5, label="Perfect calibration")

    ece = cal_data["ece"]
    ax.text(0.65, 0.10, f"ECE = {ece:.4f}", transform=ax.transAxes, fontsize=12,
            bbox=dict(facecolor="white", edgecolor="grey", boxstyle="round"))
    ax.set_xlabel("Mean Predicted Probability (confidence)")
    ax.set_ylabel("Fraction of Positives (accuracy)")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title("Calibration — Proposed GNN\n(Reliability Diagram, all labels flattened)",
                 fontweight="bold")
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_dir / "fig_calibration.png", dpi=150)
    plt.close()
    log.info("Saved fig_calibration.png")


def plot_per_label_tp_fp(per_label, per_label_csv_rows, out_dir):
    """Scatter: TP rate vs FP rate, coloured by frequency quartile."""
    csv_map = {r["code"]: r for r in per_label_csv_rows} if per_label_csv_rows else {}

    valid = [r for r in per_label if r is not None]
    freq_vals = [r["freq"] for r in valid]
    freqs_sorted = sorted(freq_vals)
    n = len(freqs_sorted)
    q25 = freqs_sorted[n // 4]
    q75 = freqs_sorted[3 * n // 4]

    def quartile(f):
        if f <= q25: return 1
        if f <= freqs_sorted[n // 2]: return 2
        if f <= q75: return 3
        return 4

    q_colors = {1: "#9C27B0", 2: "#2196F3", 3: "#4CAF50", 4: "#FF9800"}

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Per-Label TP/FP Analysis — Proposed GNN\n(fixed threshold = 0.5)",
                 fontweight="bold")

    ax = axes[0]
    for r in valid:
        q = csv_map.get(r["omop_code"], {}).get("q", quartile(r["freq"]))
        ax.scatter(r["freq"], r["recall"], color=q_colors.get(q, "grey"),
                   alpha=0.6, s=20, linewidths=0)
    ax.set_xlabel("Label Frequency")
    ax.set_ylabel("Recall (TP / (TP+FN)) @ threshold 0.5")
    ax.set_title("Recall vs Label Frequency")
    handles = [mpatches.Patch(color=q_colors[q], label=f"Q{q}") for q in [1, 2, 3, 4]]
    ax.legend(handles=handles, fontsize=9)

    ax = axes[1]
    for r in valid:
        q = csv_map.get(r["omop_code"], {}).get("q", quartile(r["freq"]))
        ax.scatter(r["recall"], r["precision"], color=q_colors.get(q, "grey"),
                   alpha=0.6, s=20, linewidths=0)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision vs Recall per Label")
    ax.legend(handles=handles, fontsize=9)

    plt.tight_layout()
    fig.savefig(out_dir / "fig_per_label_tp_fp.png", dpi=150)
    plt.close()
    log.info("Saved fig_per_label_tp_fp.png")


def plot_pr_curves_by_group(y_true, y_prob, per_label, out_dir):
    """Aggregate P-R curves for rare, mid, and common label groups."""
    valid = [r for r in per_label if r is not None]
    freqs = sorted(r["freq"] for r in valid)
    n = len(freqs)
    q25 = freqs[n // 4]
    q75 = freqs[3 * n // 4]

    rare_idx   = [r["label_idx"] for r in valid if r["freq"] <= q25]
    common_idx = [r["label_idx"] for r in valid if r["freq"] > q75]
    mid_idx    = [r["label_idx"] for r in valid if q25 < r["freq"] <= q75]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    fig.suptitle("Precision-Recall Curves by Label Frequency Group — Proposed GNN",
                 fontweight="bold")

    groups = [
        (rare_idx,   "Rare (Q1, N={})".format(len(rare_idx)),    "#9C27B0"),
        (mid_idx,    "Mid (Q2-Q3, N={})".format(len(mid_idx)),   "#2196F3"),
        (common_idx, "Common (Q4, N={})".format(len(common_idx)), "#FF9800"),
    ]

    for ax, (idxs, title, color) in zip(axes, groups):
        auprcs = []
        for idx in idxs:
            yt = y_true[:, idx]
            yp = y_prob[:, idx]
            if yt.sum() == 0:
                continue
            prec, rec, _ = precision_recall_curve(yt, yp)
            ax.plot(rec, prec, color=color, alpha=0.15, linewidth=0.8)
            auprcs.append(average_precision_score(yt, yp))

        if auprcs:
            mean_auprc = np.mean(auprcs)
            ax.text(0.5, 0.05, f"Mean AUPRC = {mean_auprc:.3f}",
                    transform=ax.transAxes, ha="center", fontsize=10,
                    bbox=dict(facecolor="white", edgecolor=color, boxstyle="round"))

        ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
        ax.set_xlabel("Recall")
        if ax == axes[0]:
            ax.set_ylabel("Precision")
        ax.set_title(title, fontweight="bold")

    plt.tight_layout()
    fig.savefig(out_dir / "fig_pr_curves_by_group.png", dpi=150)
    plt.close()
    log.info("Saved fig_pr_curves_by_group.png")


def plot_per_patient_analysis(patient_records, out_dir):
    """Two-panel: Precision@10 distribution + complexity vs Precision@10."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Per-Patient Analysis — Proposed GNN", fontweight="bold")

    prec10 = [r["precision@10"] for r in patient_records]
    comp   = [r["complexity"]   for r in patient_records]
    visits = [r["n_visits"]     for r in patient_records]

    ax = axes[0]
    ax.hist(prec10, bins=30, color=C_GNN, alpha=0.8, edgecolor="white")
    mean_p = np.mean(prec10)
    ax.axvline(mean_p, color="black", linestyle="--", linewidth=1.5, label=f"Mean={mean_p:.3f}")
    ax.set_xlabel("Precision@10")
    ax.set_ylabel("Number of Patients")
    ax.set_title("Precision@10 Distribution")
    ax.legend()

    ax = axes[1]
    sc = ax.scatter(comp, prec10, c=visits, cmap="viridis", alpha=0.4, s=8, linewidths=0)
    z = np.polyfit(comp, prec10, 1)
    p_line = np.poly1d(z)
    xfit = np.linspace(min(comp), max(comp), 200)
    ax.plot(xfit, p_line(xfit), color="black", linewidth=1.5, alpha=0.7, label="Trend")
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("# Visits")
    r_val = float(np.corrcoef(comp, prec10)[0, 1])
    ax.set_xlabel("Graph Complexity (total concept nodes)")
    ax.set_ylabel("Precision@10")
    ax.set_title(f"Complexity vs Precision@10  (r={r_val:.3f})")
    ax.legend()

    plt.tight_layout()
    fig.savefig(out_dir / "fig_per_patient_analysis.png", dpi=150)
    plt.close()
    log.info("Saved fig_per_patient_analysis.png")


def plot_f1_by_group(per_label, per_label_csv_rows, out_dir):
    """Box-like violin of F1 (optimal threshold) by frequency quartile."""
    valid  = [r for r in per_label if r is not None]
    freqs  = sorted(r["freq"] for r in valid)
    n = len(freqs)
    q25 = freqs[n // 4]; q50 = freqs[n // 2]; q75 = freqs[3 * n // 4]

    groups = {1: [], 2: [], 3: [], 4: []}
    for r in valid:
        f = r["freq"]
        if f <= q25:  groups[1].append(r["opt_f1"])
        elif f <= q50: groups[2].append(r["opt_f1"])
        elif f <= q75: groups[3].append(r["opt_f1"])
        else:          groups[4].append(r["opt_f1"])

    fig, ax = plt.subplots(figsize=(8, 5))
    q_labels = ["Q1\nRare\n(<1.0%)", "Q2\n(1.0-1.5%)", "Q3\n(1.5-3.1%)", "Q4\nCommon\n(>3.1%)"]
    q_colors = ["#9C27B0", "#2196F3", "#4CAF50", "#FF9800"]
    data = [groups[q] for q in [1, 2, 3, 4]]

    parts = ax.violinplot(data, positions=[1, 2, 3, 4], showmedians=True, showextrema=True)
    for pc, color in zip(parts["bodies"], q_colors):
        pc.set_facecolor(color); pc.set_alpha(0.7)
    for partname in ("cbars", "cmins", "cmaxes", "cmedians"):
        if partname in parts:
            parts[partname].set_edgecolor("black")

    for i, (d, color) in enumerate(zip(data, q_colors)):
        ax.scatter([i + 1] * len(d), d, color=color, alpha=0.25, s=8, linewidths=0)
        ax.text(i + 1, np.median(d) + 0.02, f"{np.median(d):.3f}", ha="center", fontsize=9)

    ax.set_xticks([1, 2, 3, 4])
    ax.set_xticklabels(q_labels)
    ax.set_ylabel("F1 Score @ Optimal Threshold")
    ax.set_title("Per-Label F1 Distribution by Frequency Quartile\nProposed GNN",
                 fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_dir / "fig_f1_by_group.png", dpi=150)
    plt.close()
    log.info("Saved fig_f1_by_group.png")


def plot_tp_fp_stacked(per_label, per_label_csv_rows, out_dir):
    """Stacked bar: mean TP/FP/FN per quartile at threshold 0.5."""
    valid  = [r for r in per_label if r is not None]
    freqs  = sorted(r["freq"] for r in valid)
    n = len(freqs)
    q25 = freqs[n // 4]; q50 = freqs[n // 2]; q75 = freqs[3 * n // 4]

    def q(f):
        if f <= q25: return 1
        if f <= q50: return 2
        if f <= q75: return 3
        return 4

    groups = {1: [], 2: [], 3: [], 4: []}
    for r in valid:
        groups[q(r["freq"])].append(r)

    fig, ax = plt.subplots(figsize=(9, 5))
    q_names = ["Q1 Rare", "Q2 Mid-low", "Q3 Mid-high", "Q4 Common"]
    x = np.arange(4)
    w = 0.55

    mean_tp    = [np.mean([r["tp"]    for r in groups[q]]) for q in [1, 2, 3, 4]]
    mean_fp    = [np.mean([r["fp"]    for r in groups[q]]) for q in [1, 2, 3, 4]]
    mean_fn    = [np.mean([r["fn"]    for r in groups[q]]) for q in [1, 2, 3, 4]]
    mean_n_pos = [np.mean([r["n_pos"] for r in groups[q]]) for q in [1, 2, 3, 4]]

    tp_rate = [mean_tp[i] / max(mean_n_pos[i], 1) for i in range(4)]
    fp_rate = [mean_fp[i] / max(mean_n_pos[i], 1) for i in range(4)]
    fn_rate = [mean_fn[i] / max(mean_n_pos[i], 1) for i in range(4)]

    ax.bar(x, tp_rate, w, label="TP rate",     color="#4CAF50", alpha=0.85)
    ax.bar(x, fp_rate, w, label="FP / n_pos",  color=C_TRANS,   alpha=0.65, bottom=tp_rate)
    ax.bar(x, fn_rate, w, label="FN rate",     color="#FF5722", alpha=0.65,
           bottom=[tp_rate[i] + fp_rate[i] for i in range(4)])

    ax.set_xticks(x)
    ax.set_xticklabels(q_names)
    ax.set_ylabel("Rate relative to mean n_pos per label")
    ax.set_title("Mean TP / FP / FN Rates by Frequency Quartile\n(threshold = 0.5)",
                 fontweight="bold")
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_dir / "fig_tp_fp_stacked.png", dpi=150)
    plt.close()
    log.info("Saved fig_tp_fp_stacked.png")


# ── CSV WRITERS ───────────────────────────────────────────────────────────────

def save_per_label_csv(per_label, out_dir):
    path = out_dir / "per_label_full.csv"
    fieldnames = [
        "label_idx", "omop_code", "n_pos", "n_neg", "freq",
        "auroc", "auprc",
        "threshold", "tp", "fp", "fn", "tn", "precision", "recall", "f1",
        "opt_threshold", "opt_tp", "opt_fp", "opt_fn", "opt_tn",
        "opt_precision", "opt_recall", "opt_f1",
    ]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in per_label:
            if r is not None:
                w.writerow(r)
    log.info("Saved per_label_full.csv (%d rows)", sum(1 for r in per_label if r))


def save_per_patient_csv(records, out_dir):
    if not records:
        return
    path = out_dir / "per_patient_full.csv"
    fieldnames = list(records[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(records)
    log.info("Saved per_patient_full.csv (%d rows)", len(records))


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("Output directory: %s", out_dir)

    # ──────────────────────────────────────────────────────────────────────
    # PATH A: Training histories
    # ──────────────────────────────────────────────────────────────────────
    log.info("\n=== PATH A: Loading training histories ===")
    e10_data   = load_json(PROJECT_ROOT / args.e10_metrics)
    e11_data   = load_json(PROJECT_ROOT / args.e11_metrics)
    e12_data   = load_json(PROJECT_ROOT / args.e12_metrics)
    trans_data = load_json(args.trans_results) if args.trans_results else None

    for label, d in [("E10", e10_data), ("E11", e11_data), ("E12", e12_data)]:
        if d:
            log.info("  %s: best_epoch=%d  test_AUROC=%.4f  test_AUPRC=%.4f",
                     label, d.get("best_epoch", -1),
                     d.get("test", {}).get("auroc_macro", float("nan")),
                     d.get("test", {}).get("auprc_macro", float("nan")))

    # Parse --extra_metrics flags (format "Label:path")
    extra_expts = []
    for i, spec in enumerate(args.extra_metrics):
        if ":" not in spec:
            log.warning("Skipping malformed --extra_metrics entry (expected 'Label:path'): %s", spec)
            continue
        lbl, path = spec.split(":", 1)
        data = load_json(path)
        color = EXTRA_COLORS[i % len(EXTRA_COLORS)]
        extra_expts.append((data, lbl.strip(), color))
        if data:
            log.info("  Extra [%s]: best_epoch=%s  test_AUROC=%s",
                     lbl.strip(),
                     data.get("best_epoch", "?"),
                     data.get("test", {}).get("auroc_macro", "?"))

    plot_training_curves(e10_data, e11_data, e12_data, trans_data, out_dir,
                         extra_expts=extra_expts if extra_expts else None)

    # ──────────────────────────────────────────────────────────────────────
    # PATH B: Inference
    # ──────────────────────────────────────────────────────────────────────
    log.info("\n=== PATH B: Model inference ===")

    ckpt_path = args.ckpt_path
    if not ckpt_path:
        ckpt_path = str(PROJECT_ROOT / "logs" / "e11" / "checkpoints" / "best_model.pt")
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        log.error("Checkpoint not found: %s", ckpt_path)
        sys.exit(1)

    y_true_path = out_dir / "y_true.npy"
    y_prob_path = out_dir / "y_prob.npy"

    if args.skip_inference and y_true_path.exists() and y_prob_path.exists():
        log.info("  Loading saved y_true/y_prob arrays ...")
        y_true = np.load(y_true_path)
        y_prob = np.load(y_prob_path)
        (train_graphs, val_graphs, test_graphs,
         label_vocab, vocab_sizes, num_labels) = load_and_prepare_graphs(
            args.graphs_path, args.vocab_path,
            args.seed, args.val_frac, args.test_frac
        )
    else:
        (train_graphs, val_graphs, test_graphs,
         label_vocab, vocab_sizes, num_labels) = load_and_prepare_graphs(
            args.graphs_path, args.vocab_path,
            args.seed, args.val_frac, args.test_frac
        )

        device = args.device
        if device.startswith("cuda") and not torch.cuda.is_available():
            log.warning("CUDA not available, using CPU")
            device = "cpu"

        y_true, y_prob = run_inference(
            test_graphs, label_vocab, vocab_sizes, num_labels,
            ckpt_path, device, args.batch_size
        )
        np.save(y_true_path, y_true)
        np.save(y_prob_path, y_prob)
        log.info("  Saved y_true.npy and y_prob.npy")

    log.info("  y_true shape: %s  (N_patients=%d, N_labels=%d)",
             y_true.shape, y_true.shape[0], y_true.shape[1])

    # ── Per-label metrics ────────────────────────────────────────────────
    log.info("Computing per-label metrics ...")
    per_label = compute_per_label_metrics(y_true, y_prob, label_vocab, args.threshold)
    valid     = [r for r in per_label if r is not None]
    log.info("  Evaluable labels: %d / %d", len(valid), len(per_label))

    # ── Per-patient metrics ──────────────────────────────────────────────
    log.info("Computing per-patient metrics ...")
    patient_records = compute_per_patient_metrics(y_true, y_prob, test_graphs)
    log.info("  Patients with >=1 positive label: %d", len(patient_records))

    # ── Calibration ──────────────────────────────────────────────────────
    log.info("Computing calibration ...")
    cal_data = compute_calibration(y_true, y_prob)
    log.info("  ECE = %.5f", cal_data["ece"])

    per_label_csv_rows = load_per_label_csv(PROJECT_ROOT / args.per_label_csv)

    # ── Plots ─────────────────────────────────────────────────────────────
    log.info("Generating plots ...")
    plot_calibration(cal_data, out_dir)
    plot_per_label_tp_fp(per_label, per_label_csv_rows, out_dir)
    plot_pr_curves_by_group(y_true, y_prob, per_label, out_dir)
    plot_per_patient_analysis(patient_records, out_dir)
    plot_f1_by_group(per_label, per_label_csv_rows, out_dir)
    plot_tp_fp_stacked(per_label, per_label_csv_rows, out_dir)

    # ── CSV outputs ───────────────────────────────────────────────────────
    save_per_label_csv(per_label, out_dir)
    save_per_patient_csv(patient_records, out_dir)

    # ── Summary JSON ──────────────────────────────────────────────────────
    def q_stats(grp_rows):
        if not grp_rows: return {}
        return {
            "n":             len(grp_rows),
            "mean_auroc":    round(float(np.mean([r["auroc"]     for r in grp_rows])), 4),
            "mean_auprc":    round(float(np.mean([r["auprc"]     for r in grp_rows])), 4),
            "mean_f1_opt":   round(float(np.mean([r["opt_f1"]    for r in grp_rows])), 4),
            "mean_recall":   round(float(np.mean([r["recall"]    for r in grp_rows])), 4),
            "mean_precision":round(float(np.mean([r["precision"] for r in grp_rows])), 4),
        }

    freqs_valid = sorted(r["freq"] for r in valid)
    n_v  = len(freqs_valid)
    q25v = freqs_valid[n_v // 4]
    q75v = freqs_valid[3 * n_v // 4]

    rare_rows   = [r for r in valid if r["freq"] <= q25v]
    mid_rows    = [r for r in valid if q25v < r["freq"] <= q75v]
    common_rows = [r for r in valid if r["freq"] > q75v]

    summary = {
        "n_test_patients":    int(y_true.shape[0]),
        "n_labels":           int(y_true.shape[1]),
        "n_evaluable_labels": len(valid),
        "ece":                cal_data["ece"],
        "threshold":          args.threshold,
        "macro_auroc":        round(float(np.mean([r["auroc"]        for r in valid])), 4),
        "macro_auprc":        round(float(np.mean([r["auprc"]        for r in valid])), 4),
        "macro_f1_opt":       round(float(np.mean([r["opt_f1"]       for r in valid])), 4),
        "mean_precision@10":  round(float(np.mean([r["precision@10"] for r in patient_records])), 4),
        "mean_recall@10":     round(float(np.mean([r["recall@10"]    for r in patient_records])), 4),
        "group_stats": {
            "rare (Q1)":   q_stats(rare_rows),
            "mid (Q2-Q3)": q_stats(mid_rows),
            "common (Q4)": q_stats(common_rows),
        },
        "worst_10_labels": sorted(valid, key=lambda r: r["auroc"])[:10],
        "best_10_labels":  sorted(valid, key=lambda r: -r["auroc"])[:10],
        "training": {
            "e10_best_epoch": e10_data.get("best_epoch") if e10_data else None,
            "e11_best_epoch": e11_data.get("best_epoch") if e11_data else None,
            "e12_best_epoch": e12_data.get("best_epoch") if e12_data else None,
            "e10_test_auroc": e10_data.get("test", {}).get("auroc_macro") if e10_data else None,
            "e11_test_auroc": e11_data.get("test", {}).get("auroc_macro") if e11_data else None,
            "e12_test_auroc": e12_data.get("test", {}).get("auroc_macro") if e12_data else None,
        },
    }

    with open(out_dir / "rich_inference_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    log.info("Saved rich_inference_summary.json")

    print("\n" + "=" * 65)
    print("RICH INFERENCE SUMMARY — Proposed GNN")
    print("=" * 65)
    print(f"Test patients:       {summary['n_test_patients']}")
    print(f"Evaluable labels:    {summary['n_evaluable_labels']}")
    print(f"Macro AUROC:         {summary['macro_auroc']:.4f}")
    print(f"Macro AUPRC:         {summary['macro_auprc']:.4f}")
    print(f"Macro F1 (opt thr):  {summary['macro_f1_opt']:.4f}")
    print(f"Mean Precision@10:   {summary['mean_precision@10']:.4f}")
    print(f"Mean Recall@10:      {summary['mean_recall@10']:.4f}")
    print(f"ECE (calibration):   {summary['ece']:.5f}")
    print()
    for grp, st in summary["group_stats"].items():
        print(f"  {grp:14s}  AUROC={st['mean_auroc']:.4f}  AUPRC={st['mean_auprc']:.4f}"
              f"  F1(opt)={st['mean_f1_opt']:.4f}  Prec={st['mean_precision']:.4f}"
              f"  Rec={st['mean_recall']:.4f}")
    print()
    print(f"Outputs: {out_dir}")
    print("=" * 65)


if __name__ == "__main__":
    main()
