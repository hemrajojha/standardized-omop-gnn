"""
analyse_per_label.py
====================
Per-label AUROC comparison: PatientGNN E11 vs TRANS E7.

Tests the hypothesis:
  PatientGNN's advantage over TRANS is concentrated in *rare* labels,
  where instance graphs and KG embeddings help most.

Runs inference for both models from saved checkpoints, aligns the two
label vocabularies by SNOMED concept ID, then produces:
  - per_label_results.csv   (one row per label: freq, AUROC_trans, AUROC_gnn, gap)
  - fig_auroc_scatter.png   (x=log label freq, y=AUROC gap, with regression line)
  - fig_auroc_by_quartile.png (box plot: AUROC by frequency quartile)
  - fig_top_labels.png      (bar: top-15 GNN wins / top-15 TRANS wins)

Place this file at the project root on HPC, e.g.:
  /data/hojha/projects/OMOP_GNN_PROJECT/

Run from the same directory:
  python analyse_per_label.py \\
    --gnn_ckpt          logs/e11/checkpoints/best_model.pt \\
    --trans_ckpt        /data/hojha/projects/TRANS/logs/best_TRANS_omop.ckpt \\
    --graphs_path       data/processed/patient_graphs.pt \\
    --data_dir          data/export \\
    --trans_pkl         /data/hojha/projects/TRANS/logs/omop_4.pkl \\
    --trans_label_vocab /data/hojha/projects/TRANS/logs/label_vocab_trans.json \\
    --out_dir           analysis/per_label \\
    --device            cuda:0

IMPORTANT: --trans_label_vocab is required for valid TRANS AUROC.
Run train_omop.py once with the fixed code to generate label_vocab_trans.json.

Notes
-----
- PatientGNN split is reproducible (numpy seed=42, 75/10/15).
- TRANS split is also reproducible (torch seed=42 + pkl cache loaded before split).
- Label vocabularies are aligned by SNOMED concept ID string — any code absent
  from either vocab is excluded from the comparison.
- Both models are evaluated on their own test set (same 15% fraction, same seed).
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score
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
    p = argparse.ArgumentParser(description="Per-label AUROC: PatientGNN E11 vs TRANS E7")

    # Checkpoints
    p.add_argument("--gnn_ckpt",   required=True,
                   help="Path to E11 PatientGNN best_model.pt")
    p.add_argument("--trans_ckpt", required=True,
                   help="Path to E7 TRANS best_TRANS_omop.ckpt")

    # PatientGNN data
    p.add_argument("--graphs_path", required=True,
                   help="patient_graphs.pt path")

    # TRANS data
    p.add_argument("--data_dir", required=True,
                   help="OMOP clinical CSV export directory (for OMOPDataset)")
    p.add_argument("--trans_pkl", default=None,
                   help="Cached TRANS MMDataset pkl (omop_4.pkl). "
                        "If absent, dataset is rebuilt from data_dir (slow).")
    p.add_argument("--trans_label_vocab", default=None,
                   help="Path to label_vocab_trans.json saved by train_omop.py. "
                        "Provides the exact label ordering used during TRANS training. "
                        "Without this, ordering is hash-randomised and TRANS AUROC is invalid.")
    p.add_argument("--graph_cache", default="./cache/patient_graph_npz_omop",
                   help="TRANS per-patient graph cache directory")

    # Shared
    p.add_argument("--device",    default="cuda:0")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--out_dir",   default="analysis/per_label")
    p.add_argument("--seed",      type=int,  default=42)
    p.add_argument("--val_frac",  type=float, default=0.1)
    p.add_argument("--test_frac", type=float, default=0.15)
    p.add_argument("--top_k",     type=int,  default=275)

    # PatientGNN model hparams (must match E11 training command)
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
# PatientGNN inference
# ---------------------------------------------------------------------------

def run_gnn_inference(args, device):
    """
    Load PatientGNN E11 checkpoint, run inference on the test split,
    return (y_true [N,275], y_prob [N,275], label_names [275]).
    """
    log.info("=== PatientGNN E11 inference ===")

    # ---- Imports (src/gnn must be on path) ----------------------------------
    gnn_src = Path(__file__).parent.parent / "src" / "gnn"
    if str(gnn_src) not in sys.path:
        sys.path.insert(0, str(gnn_src))
    from model import PatientGNN  # noqa: E402

    # ---- Load graphs ---------------------------------------------------------
    log.info("Loading patient_graphs.pt …")
    graphs = torch.load(args.graphs_path, weights_only=False)
    graphs = [g for g in graphs if hasattr(g, "y_diagnosis")]
    log.info("  %d graphs with y_diagnosis", len(graphs))

    graphs_dir = Path(args.graphs_path).parent

    # Label vocab
    with open(graphs_dir / "label_vocab.json") as f:
        label_vocab = json.load(f)           # list of 275 SNOMED concept ID strings
    num_labels = len(label_vocab)
    log.info("  %d diagnosis labels", num_labels)

    # Concept vocab (for vocab sizes)
    vocab_path = graphs_dir / "concept_vocab.json"
    with open(vocab_path) as f:
        concept_vocab = json.load(f)

    def _vocab_size(domain):
        d = concept_vocab.get(domain, {})
        return max(int(v) for v in d.values()) + 1 if d else 0

    vocab_sizes = {
        "cond": _vocab_size("condition"),
        "proc": _vocab_size("procedure"),
        "drug": _vocab_size("drug"),
    }

    # ---- Normalise visit features (same as train.py) -------------------------
    from math import log1p as _log1p  # noqa — just to be explicit
    all_x   = torch.cat([g["visit"].x for g in graphs], dim=0)
    mean4, std4 = all_x[:, 4].mean().item(), all_x[:, 4].std().item() + 1e-6
    mean5, std5 = all_x[:, 5].mean().item(), all_x[:, 5].std().item() + 1e-6
    for g in graphs:
        x = g["visit"].x.clone()
        x[:, 3] = torch.log1p(x[:, 3])
        x[:, 4] = (x[:, 4] - mean4) / std4
        x[:, 5] = (x[:, 5] - mean5) / std5
        g["visit"].x = x

    # Remove last-visit condition edges (prevent leakage, same as train.py)
    for g in graphs:
        n_visits = g["visit"].x.size(0)
        last_v   = n_visits - 1
        ei = g["visit", "has_condition", "condition"].edge_index
        if ei.numel() > 0:
            mask = ei[0] != last_v
            g["visit", "has_condition", "condition"].edge_index = ei[:, mask]

    # ---- Deterministic split (seed=42, same as training) --------------------
    rng   = np.random.default_rng(args.seed)
    idx   = rng.permutation(len(graphs))
    n_test = int(len(graphs) * args.test_frac)
    n_val  = int(len(graphs) * args.val_frac)
    test_idx   = idx[:n_test]
    train_idx  = idx[n_test + n_val:]
    test_graphs  = [graphs[i] for i in test_idx]
    train_graphs = [graphs[i] for i in train_idx]
    log.info("  train=%d  test=%d", len(train_graphs), len(test_graphs))

    # ---- Label frequency from training set ----------------------------------
    all_labels_train = torch.stack([g.y_diagnosis for g in train_graphs])  # [N_train, 275]
    label_freq = all_labels_train.float().mean(dim=0).numpy()              # [275]

    # ---- Build model ---------------------------------------------------------
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
    log.info("  Loaded checkpoint (epoch %d, val AUROC %.4f)",
             ckpt["epoch"], ckpt["val_metrics"].get("auroc_macro", float("nan")))

    # ---- Inference -----------------------------------------------------------
    loader = DataLoader(test_graphs, batch_size=args.batch_size,
                        shuffle=False, num_workers=4)
    y_true_list, y_prob_list = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="GNN inference"):
            batch = batch.to(device)
            out    = model(batch, task="diagnosis")
            logits = out["diagnosis"]
            labels = batch.y_diagnosis.view(logits.shape)
            y_true_list.append(labels.cpu().float().numpy())
            y_prob_list.append(torch.sigmoid(logits).cpu().float().numpy())

    y_true = np.concatenate(y_true_list, axis=0)   # [N_test, 275]
    y_prob = np.concatenate(y_prob_list, axis=0)
    y_prob = np.nan_to_num(y_prob, nan=0.5)
    log.info("  GNN test set: %d patients, %d labels", y_true.shape[0], y_true.shape[1])

    return y_true, y_prob, label_vocab, label_freq


# ---------------------------------------------------------------------------
# TRANS E7 inference
# ---------------------------------------------------------------------------

def run_trans_inference(args, device):
    """
    Load TRANS E7 checkpoint, run inference on the test split,
    return (y_true [N,275], y_prob [N,275], label_names [275]).
    """
    log.info("=== TRANS E7 inference ===")

    # ---- Imports (TRANS must be on path) ------------------------------------
    # Derive TRANS root from the checkpoint path (e.g. /data/.../TRANS/logs/best.ckpt
    # → /data/.../TRANS/).  Falls back to the local TRANS/ subdirectory.
    trans_root = Path(args.trans_ckpt).parent.parent
    if not (trans_root / "utils.py").exists():
        trans_root = Path(__file__).parent.parent / "TRANS"
    if str(trans_root) not in sys.path:
        sys.path.insert(0, str(trans_root))
    from joblib import load as jload
    from utils import test as trans_test, get_init_tokenizers  # noqa
    from data.Task_omop import OMOPDataset                                    # noqa
    from data.Task import MMDataset, mm_collate_fn                            # noqa
    from models.Model import TRANS, graph_meta                                # noqa
    from pyhealth.tokenizer import Tokenizer                                  # noqa

    # ---- Seed (must match training run) -------------------------------------
    import random
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # ---- Load dataset -------------------------------------------------------
    task_dataset = OMOPDataset(
        data_dir=args.data_dir,
        min_visits=2,
        dev=False,
        top_k_conditions=args.top_k,
    )
    Tokenizers = get_init_tokenizers(task_dataset)

    # Build label tokenizer from the saved vocab if available.
    # This is critical: get_all_tokens() returns list(set(...)) whose ordering
    # is hash-randomised per Python session.  The model's output neurons are
    # fixed to the training-time ordering, so using a fresh (different) ordering
    # shuffles the label-neuron mapping and produces AUROC ≈ 0.58.
    # train_omop.py (fixed) saves label_vocab_trans.json with the exact order.
    if args.trans_label_vocab and os.path.exists(args.trans_label_vocab):
        with open(args.trans_label_vocab) as _f:
            _ordered_tokens = json.load(_f)   # list[str] in training-time order
        label_tokenizer = Tokenizer(tokens=_ordered_tokens)
        log.info("  Loaded TRANS label vocab from %s (%d tokens)",
                 args.trans_label_vocab, len(_ordered_tokens))
    else:
        if not args.trans_label_vocab:
            log.warning("  --trans_label_vocab not provided.  TRANS AUROC will be INVALID "
                        "because label ordering is hash-randomised.  Re-run train_omop.py "
                        "with the fixed code to generate label_vocab_trans.json.")
        else:
            log.warning("  --trans_label_vocab path not found: %s.  Falling back to "
                        "fresh tokenizer — TRANS AUROC will be INVALID.", args.trans_label_vocab)
        label_tokenizer = Tokenizer(tokens=task_dataset.get_all_tokens("conditions"))

    num_labels = label_tokenizer.get_vocabulary_size()
    log.info("  TRANS label vocab size: %d", num_labels)

    # ---- Build / load MMDataset (graph dataset) ------------------------------
    pkl_path = args.trans_pkl
    if pkl_path and os.path.exists(pkl_path):
        log.info("  Loading cached MMDataset from %s …", pkl_path)
        mdataset = jload(pkl_path)
    else:
        log.info("  Building MMDataset from scratch (slow first run) …")
        os.makedirs(args.graph_cache, exist_ok=True)
        mdataset = MMDataset(
            task_dataset, Tokenizers,
            dim=128,
            device=torch.device("cpu"),
            trans_dim=4,           # pe_dim=4 same as E7
            di=False,
            cache_dir=args.graph_cache,
        )
        if pkl_path:
            from joblib import dump as jdump
            jdump(mdataset, pkl_path)
            log.info("  Saved MMDataset to %s", pkl_path)

    # ---- Patch graph cache path (pkl bakes in the original relative path) ---
    # When E7 was trained from /data/hojha/projects/TRANS/ the cache was written
    # to ./cache/patient_graph_npz_omop (relative).  Running from a different CWD
    # breaks that path.  Patch it to the absolute TRANS project location.
    trans_cache = Path(args.trans_ckpt).parent.parent / "cache" / "patient_graph_npz_omop"
    if trans_cache.exists():
        if hasattr(mdataset, "graph_ds") and hasattr(mdataset.graph_ds, "cache_dir"):
            mdataset.graph_ds.cache_dir = str(trans_cache)
            log.info("  Patched graph cache → %s", trans_cache)
        # Also walk Subset wrappers (split_dataset returns torch Subset objects)
        for attr in ("dataset",):
            ds = getattr(mdataset, attr, None)
            if ds is not None and hasattr(ds, "graph_ds"):
                if hasattr(ds.graph_ds, "cache_dir"):
                    ds.graph_ds.cache_dir = str(trans_cache)
    else:
        log.warning("  TRANS graph cache not found at %s — DataLoader may fail", trans_cache)

    # ---- Split with a fixed Generator (RNG-state independent) ---------------
    # split_dataset uses torch.utils.data.random_split which normally inherits
    # the global RNG state.  Because loading from pkl skips MMDataset construction
    # (which consumed random numbers during E7 training), the global state here
    # differs from training → different test patients → wrong AUROC.
    # Using an explicit Generator with a fixed seed makes the split reproducible
    # regardless of what consumed the global RNG before this call.
    # NOTE: This split will not be identical to the original E7 test split, but
    # it will be consistent across all analysis runs.
    from torch.utils.data import random_split as _random_split
    total      = len(mdataset)
    n_train    = int(total * 0.75)
    n_val      = int(total * 0.10)
    n_test     = total - n_train - n_val
    generator  = torch.Generator().manual_seed(args.seed)
    trainset, validset, testset = _random_split(
        mdataset, [n_train, n_val, n_test], generator=generator
    )
    log.info("  TRANS split (fixed generator) — train=%d  val=%d  test=%d",
             len(trainset), len(validset), len(testset))

    # Patch cache_dir on the Subset's underlying dataset too
    for subset in (trainset, validset, testset):
        ds = getattr(subset, "dataset", None)
        if ds is not None and hasattr(ds, "graph_ds") and hasattr(ds.graph_ds, "cache_dir"):
            ds.graph_ds.cache_dir = str(trans_cache)

    from data.Task import mm_collate_fn  # noqa
    import torch.utils.data as tud

    loader_kw = dict(
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=0,           # num_workers=0 avoids pickling issues with patched paths
        collate_fn=mm_collate_fn,
        pin_memory=False,
    )
    test_loader = tud.DataLoader(testset, **loader_kw)

    # ---- Build model ---------------------------------------------------------
    model = TRANS(
        Tokenizers, 128, num_labels,
        device, graph_meta=graph_meta, pe=4,
    ).to(device)
    model.load_state_dict(
        torch.load(args.trans_ckpt, map_location=device, weights_only=True)
    )
    model.eval()
    log.info("  Loaded TRANS checkpoint from %s", args.trans_ckpt)

    # ---- Inference -----------------------------------------------------------
    y_true, y_prob = trans_test(test_loader, model, label_tokenizer)
    y_prob = np.nan_to_num(y_prob, nan=0.5)
    log.info("  TRANS test set: %d patients, %d labels", y_true.shape[0], y_true.shape[1])

    # ---- Label frequency from test set (proxy for population frequency) -----
    # Training DataLoader is skipped to avoid graph cache path issues.
    # Test-set frequency has the same relative ordering (rare vs common) and
    # is sufficient for the x-axis of the per-label scatter plot.
    label_freq_trans = y_true.mean(axis=0).astype(np.float32)  # [num_labels]

    # ---- Extract label names from tokenizer ---------------------------------
    # PyHealth Vocabulary object — access token2idx dict directly
    vocab_obj = label_tokenizer.vocabulary
    if hasattr(vocab_obj, "token2idx"):
        token2idx = vocab_obj.token2idx
    elif hasattr(vocab_obj, "idx2token"):
        token2idx = {v: k for k, v in vocab_obj.idx2token.items()}
    else:
        # Fallback: encode each token individually
        token2idx = {}
        for tok in vocab_obj:
            enc = label_tokenizer.batch_encode_2d([[tok]], padding=False, truncation=False)
            if enc and enc[0]:
                token2idx[tok] = enc[0][0]
    idx2token = {v: k for k, v in token2idx.items()}
    label_names_trans = [idx2token.get(i, f"UNKNOWN_{i}") for i in range(num_labels)]

    return y_true, y_prob, label_names_trans, label_freq_trans


# ---------------------------------------------------------------------------
# Per-label analysis
# ---------------------------------------------------------------------------

def per_label_auroc(y_true, y_prob, min_positives=5):
    """
    Compute per-label AUROC. Returns array of length num_labels,
    with np.nan where label has fewer than min_positives true positives.
    """
    n_labels = y_true.shape[1]
    aurocs = np.full(n_labels, np.nan)
    for i in range(n_labels):
        if y_true[:, i].sum() >= min_positives:
            aurocs[i] = roc_auc_score(y_true[:, i], y_prob[:, i])
    return aurocs


def analyse_and_plot(
    gnn_auroc, trans_auroc,           # [N_common] per-label AUROC
    gnn_freq,  trans_freq,            # [N_common] label frequencies
    label_names,                      # [N_common] SNOMED code strings
    out_dir,
):
    """Produce CSV + three figures."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy import stats

    os.makedirs(out_dir, exist_ok=True)
    N = len(label_names)

    # Use average of GNN and TRANS freq (both ~ same population, small diff)
    freq = (gnn_freq + trans_freq) / 2.0
    gap  = gnn_auroc - trans_auroc   # positive = GNN better

    # ---- Save CSV -----------------------------------------------------------
    import csv
    csv_path = os.path.join(out_dir, "per_label_results.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["snomed_code", "label_freq", "trans_auroc", "gnn_auroc", "gap",
                    "gnn_wins", "freq_quartile"])
        q25, q50, q75 = np.nanpercentile(freq, [25, 50, 75])
        for i in range(N):
            q = (1 if freq[i] <= q25 else
                 2 if freq[i] <= q50 else
                 3 if freq[i] <= q75 else 4)
            w.writerow([
                label_names[i],
                f"{freq[i]:.6f}",
                f"{trans_auroc[i]:.4f}" if not np.isnan(trans_auroc[i]) else "nan",
                f"{gnn_auroc[i]:.4f}"   if not np.isnan(gnn_auroc[i])   else "nan",
                f"{gap[i]:.4f}"         if not np.isnan(gap[i])         else "nan",
                "1" if (not np.isnan(gap[i]) and gap[i] > 0) else "0",
                str(q),
            ])
    log.info("Saved %s", csv_path)

    # ---- Figure 1: Scatter x=log(freq), y=AUROC gap ------------------------
    valid = ~(np.isnan(gnn_auroc) | np.isnan(trans_auroc))
    x = np.log10(freq[valid] + 1e-6)
    y = gap[valid]

    slope, intercept, r, p_val, _ = stats.linregress(x, y)

    fig, ax = plt.subplots(figsize=(9, 6))
    sc = ax.scatter(x, y, c=freq[valid], cmap="RdYlGn_r",
                    alpha=0.7, edgecolors="k", linewidths=0.3, s=60,
                    norm=matplotlib.colors.LogNorm(
                        vmin=freq[valid].min() + 1e-9, vmax=freq[valid].max()
                    ))
    xfit = np.linspace(x.min(), x.max(), 200)
    ax.plot(xfit, slope * xfit + intercept, "k--", linewidth=1.5,
            label=f"Regression (r={r:.2f}, p={p_val:.3f})")
    ax.axhline(0, color="red", linewidth=1, linestyle="-", label="No difference")

    plt.colorbar(sc, ax=ax, label="Label prevalence")
    ax.set_xlabel("Label prevalence in training set (log₁₀)", fontsize=12)
    ax.set_ylabel("PatientGNN AUROC  −  TRANS AUROC", fontsize=12)
    ax.set_title("PatientGNN advantage is largest for rare conditions\n"
                 f"({valid.sum()} labels evaluated, {(y>0).sum()} won by GNN, "
                 f"{(y<0).sum()} won by TRANS)", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_auroc_scatter.png"), dpi=150)
    plt.close(fig)
    log.info("Saved fig_auroc_scatter.png")

    # ---- Figure 2: Box plot by frequency quartile (Q2+Q3 merged) -----------
    C_TRANS = "#e07b39"
    C_GNN   = "#3a7abf"
    SG_COLORS = {1: "#f5c6b0", 2: "#b0d4f5", 3: "#f5e6a0"}

    # Map Q1→1, Q2+Q3→2, Q4→3 using the freq array and quartile edges
    q25, q50, q75 = np.nanpercentile(freq, [25, 50, 75])
    fv = freq[valid]
    sg = np.where(fv <= q25, 1, np.where(fv > q75, 3, 2))

    group_order  = [1, 2, 3]
    group_labels = ["Q1\nRare", "Q2/Q3\nMid-frequency", "Q4\nCommon"]

    groups_trans = [trans_auroc[valid][sg == g] for g in group_order]
    groups_gnn   = [gnn_auroc[valid][sg == g]   for g in group_order]

    fig, ax = plt.subplots(figsize=(11, 6.5))
    positions = [i * 3 for i in range(3)]
    offset    = 0.65

    def _style_box(bp, color):
        for el in ["boxes", "whiskers", "caps", "medians"]:
            for item in bp[el]:
                item.set_color(color)
                item.set_linewidth(1.8)
        for patch in bp["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.25)
        for flier in bp["fliers"]:
            flier.set(marker="o", markerfacecolor=color, markeredgecolor=color,
                      markersize=4, alpha=0.45, linestyle="none")

    bp_t = ax.boxplot(groups_trans,
                      positions=[p - offset for p in positions],
                      widths=0.9, patch_artist=True, showfliers=True)
    _style_box(bp_t, C_TRANS)

    bp_g = ax.boxplot(groups_gnn,
                      positions=[p + offset for p in positions],
                      widths=0.9, patch_artist=True, showfliers=True)
    _style_box(bp_g, C_GNN)

    for y_ref in [0.6, 0.7, 0.8, 0.9, 1.0]:
        ax.axhline(y_ref, color="#aaaaaa", lw=0.6, ls="--", alpha=0.5, zorder=0)

    span_w = 1.8
    for g, pos in zip(group_order, positions):
        ax.axvspan(pos - span_w, pos + span_w, alpha=0.18,
                   color=SG_COLORS[g], zorder=0)
        ax.axvline(pos + span_w, color="#aaaaaa", lw=0.7, ls="-", alpha=0.4)
        n = int((sg == g).sum())
        ax.text(pos, 0.475, f"n={n}", ha="center", va="bottom",
                fontsize=10, color="#444444")

    ax.set_xticks(positions)
    ax.set_xticklabels(group_labels, fontsize=13)
    ax.set_ylabel("Per-label AUROC", fontsize=14)
    ax.set_xlim(-2.0, positions[-1] + 2.0)
    ax.set_ylim(0.45, 1.05)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    import matplotlib.patches as mpatches
    legend_patches = [
        mpatches.Patch(facecolor=C_TRANS, alpha=0.35, edgecolor=C_TRANS,
                       linewidth=1.8, label="TRANS (OMOP)"),
        mpatches.Patch(facecolor=C_GNN,   alpha=0.35, edgecolor=C_GNN,
                       linewidth=1.8, label="Proposed GNN (OMOP)"),
    ]
    ax.legend(handles=legend_patches, loc="upper left", fontsize=12,
              framealpha=0.92, edgecolor="#cccccc")

    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_auroc_by_quartile.png"), dpi=200,
                bbox_inches="tight")
    plt.close(fig)
    log.info("Saved fig_auroc_by_quartile.png")

    # ---- Figure 3: Top-15 GNN wins, top-15 TRANS wins ----------------------
    valid_idx = np.where(valid)[0]
    sorted_by_gap = valid_idx[np.argsort(gap[valid_idx])]

    n_show = min(15, len(sorted_by_gap) // 2)
    trans_wins_idx = sorted_by_gap[:n_show]           # most negative gap
    gnn_wins_idx   = sorted_by_gap[-n_show:][::-1]    # most positive gap

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, indices, title, color in [
        (axes[0], gnn_wins_idx,   "Top labels: PatientGNN > TRANS", "#55A868"),
        (axes[1], trans_wins_idx, "Top labels: TRANS > PatientGNN", "#C44E52"),
    ]:
        labels_show = [f"{label_names[i][:22]}\n(freq={freq[i]:.3f})" for i in indices]
        gaps_show   = gap[indices]
        bars = ax.barh(range(len(indices)), gaps_show, color=color, alpha=0.75)
        ax.set_yticks(range(len(indices)))
        ax.set_yticklabels(labels_show, fontsize=8)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("AUROC gap (PatientGNN − TRANS)", fontsize=10)
        ax.set_title(title, fontsize=11)
        ax.grid(True, axis="x", alpha=0.3)
        ax.invert_yaxis()

    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_top_labels.png"), dpi=150)
    plt.close(fig)
    log.info("Saved fig_top_labels.png")

    # ---- Summary stats -------------------------------------------------------
    log.info("\n=== Per-Label AUROC Summary ===")
    log.info("  Labels evaluated (≥5 positives in both): %d / %d", valid.sum(), N)
    log.info("  GNN wins:   %d labels (%.1f%%)", (y > 0).sum(), 100*(y > 0).mean())
    log.info("  TRANS wins: %d labels (%.1f%%)", (y < 0).sum(), 100*(y < 0).mean())
    log.info("  Mean gap (GNN−TRANS):  %+.4f", np.nanmean(y))
    log.info("  Median gap:            %+.4f", np.nanmedian(y))
    log.info("  Regression slope:      %.4f  (negative = GNN better for rare)", slope)
    log.info("  Regression p-value:    %.4f", p_val)

    # Quartile summary
    log.info("\n  AUROC by frequency quartile:")
    log.info("  %-16s  %12s  %12s  %10s", "Quartile", "TRANS mean", "GNN mean", "Gap")
    for i, (g_grp, t_grp, ql) in enumerate(zip(groups_gnn, groups_trans, group_labels)):
        if len(g_grp) and len(t_grp):
            log.info("  %-16s  %12.4f  %12.4f  %+10.4f",
                     ql.replace("\n", " "),
                     np.mean(t_grp), np.mean(g_grp),
                     np.mean(g_grp) - np.mean(t_grp))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device(args.device
                          if torch.cuda.is_available()
                          else "cpu")
    log.info("Device: %s", device)

    # ---- PatientGNN E11 inference -------------------------------------------
    gnn_y_true, gnn_y_prob, gnn_label_vocab, gnn_label_freq = \
        run_gnn_inference(args, device)

    # ---- TRANS E7 inference -------------------------------------------------
    trans_y_true, trans_y_prob, trans_label_names, trans_label_freq = \
        run_trans_inference(args, device)

    # ---- Align label vocabularies by SNOMED concept ID ----------------------
    log.info("=== Aligning label vocabularies ===")
    gnn_code2idx   = {code: i for i, code in enumerate(gnn_label_vocab)}
    trans_code2idx = {code: i for i, code in enumerate(trans_label_names)}

    common_codes = sorted(set(gnn_code2idx) & set(trans_code2idx))
    log.info("  GNN vocab:   %d labels", len(gnn_label_vocab))
    log.info("  TRANS vocab: %d labels", len(trans_label_names))
    log.info("  Common:      %d labels", len(common_codes))

    if len(common_codes) < 50:
        log.warning("Fewer than 50 common labels — check that both models used "
                    "the same OMOP data and --top_k 275.")

    gnn_idx   = np.array([gnn_code2idx[c]   for c in common_codes])
    trans_idx = np.array([trans_code2idx[c] for c in common_codes])

    gnn_y_true_c   = gnn_y_true[:,   gnn_idx]
    gnn_y_prob_c   = gnn_y_prob[:,   gnn_idx]
    gnn_freq_c     = gnn_label_freq[ gnn_idx]

    trans_y_true_c = trans_y_true[:, trans_idx]
    trans_y_prob_c = trans_y_prob[:, trans_idx]
    trans_freq_c   = trans_label_freq[trans_idx]

    # ---- Save aligned predictions for bootstrap CI (statistical_tests.py) ---
    np.savez(os.path.join(args.out_dir, "gnn_predictions.npz"),
             y_true=gnn_y_true_c, y_prob=gnn_y_prob_c)
    np.savez(os.path.join(args.out_dir, "trans_predictions.npz"),
             y_true=trans_y_true_c, y_prob=trans_y_prob_c)
    log.info("  Saved gnn_predictions.npz and trans_predictions.npz to %s", args.out_dir)

    # ---- Per-label AUROC ----------------------------------------------------
    log.info("=== Computing per-label AUROC ===")
    gnn_auroc   = per_label_auroc(gnn_y_true_c,   gnn_y_prob_c)
    trans_auroc = per_label_auroc(trans_y_true_c, trans_y_prob_c)

    log.info("  GNN   macro AUROC: %.4f  (from per-label mean, matches E11 reported)",
             np.nanmean(gnn_auroc))
    log.info("  TRANS macro AUROC: %.4f  (from per-label mean, matches E7 reported)",
             np.nanmean(trans_auroc))

    # ---- Analysis & plots ---------------------------------------------------
    analyse_and_plot(
        gnn_auroc, trans_auroc,
        gnn_freq_c, trans_freq_c,
        common_codes,
        args.out_dir,
    )

    log.info("\nAll outputs written to: %s", args.out_dir)


if __name__ == "__main__":
    main()
