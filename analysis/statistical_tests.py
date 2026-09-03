"""
statistical_tests.py
====================
Standalone statistical significance tests for the paper results.

Three tests are implemented:

  1. Wilcoxon Signed-Rank Test  — non-parametric paired test on per-label AUROCs
                                   (recommended over t-test: no normality assumption,
                                    AUROC values are bounded [0, 1])

  2. Paired t-test              — parametric paired test on per-label AUROCs
                                   (complement to Wilcoxon; assumes normality)

  3. Bootstrap CI               — 95% confidence intervals on macro AUROC for each
                                   model and on the difference (E-TRANS minus TRANS),
                                   by resampling patients with replacement

Tests 1 and 2 require per_label_results.csv (output of analyse_per_label.py).
Test 3 requires saved prediction arrays (.npz) OR model checkpoints to re-run inference.

Usage — from per_label CSV only (tests 1 + 2):
----------------------------------------------
python statistical_tests.py \\
    --per_label_csv  analysis/per_label/per_label_results.csv \\
    --out_dir        analysis/statistical_tests

Usage — with saved predictions (all three tests):
--------------------------------------------------
# First save predictions from analyse_per_label.py (or run inference here):
python statistical_tests.py \\
    --per_label_csv   analysis/per_label/per_label_results.csv \\
    --gnn_preds       analysis/per_label/gnn_predictions.npz \\
    --trans_preds     analysis/per_label/trans_predictions.npz \\
    --out_dir         analysis/statistical_tests

Usage — re-run inference from checkpoints (all three tests):
-------------------------------------------------------------
python statistical_tests.py \\
    --per_label_csv   analysis/per_label/per_label_results.csv \\
    --gnn_ckpt        logs/e11/checkpoints/best_model.pt \\
    --graphs_path     data/processed/patient_graphs.pt \\
    --out_dir         analysis/statistical_tests \\
    --device          cuda:0

Outputs (written to --out_dir):
    statistical_results.json    all test results with statistics and p-values
    statistical_results.txt     human-readable summary report
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

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
    p = argparse.ArgumentParser(description="Statistical significance tests for paper results")

    # Required for tests 1 + 2
    p.add_argument("--per_label_csv", default=None,
                   help="per_label_results.csv from analyse_per_label.py "
                        "(required for Wilcoxon and paired t-test)")

    # Option A: pre-saved prediction arrays for bootstrap CI
    p.add_argument("--gnn_preds",   default=None,
                   help="Path to GNN prediction .npz file with keys: y_true, y_prob")
    p.add_argument("--trans_preds", default=None,
                   help="Path to TRANS prediction .npz file with keys: y_true, y_prob")

    # Option B: re-run GNN inference from checkpoint for bootstrap CI
    p.add_argument("--gnn_ckpt",    default=None,
                   help="PatientGNN checkpoint best_model.pt (used if --gnn_preds absent)")
    p.add_argument("--graphs_path", default=None,
                   help="patient_graphs.pt (required with --gnn_ckpt)")
    p.add_argument("--device",      default="cuda:0")
    p.add_argument("--batch_size",  type=int, default=128)

    # GNN model hparams (must match training)
    p.add_argument("--hidden_dim",             type=int,   default=128)
    p.add_argument("--kg_embed_dim",           type=int,   default=128)
    p.add_argument("--num_gnn_layers",         type=int,   default=2)
    p.add_argument("--num_transformer_layers", type=int,   default=2)
    p.add_argument("--nhead",                  type=int,   default=4)
    p.add_argument("--pe_dim",                 type=int,   default=4)
    p.add_argument("--alpha",                  type=float, default=0.8)
    p.add_argument("--dropout",                type=float, default=0.3)
    p.add_argument("--seed",                   type=int,   default=42)
    p.add_argument("--val_frac",               type=float, default=0.1)
    p.add_argument("--test_frac",              type=float, default=0.15)

    # Bootstrap settings
    p.add_argument("--n_bootstrap", type=int, default=1000,
                   help="Number of bootstrap resamples (default: 1000)")
    p.add_argument("--ci_level",    type=float, default=0.95,
                   help="Confidence interval level (default: 0.95)")

    p.add_argument("--out_dir", default="analysis/statistical_tests")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Test 1 + 2: Wilcoxon and paired t-test on per-label AUROCs
# ---------------------------------------------------------------------------

def run_paired_tests(csv_path: str) -> dict:
    """
    Load per_label_results.csv and run:
      - Wilcoxon signed-rank test (non-parametric, recommended)
      - Paired t-test (parametric, complementary)

    Both test whether per-label AUROC of E-TRANS is significantly
    different from TRANS across all evaluated labels.

    Returns a dict with statistics and p-values for both tests.
    """
    log.info("=== Paired Tests (Wilcoxon + t-test) on per-label AUROCs ===")
    log.info("  Loading %s", csv_path)

    df = pd.read_csv(csv_path)

    # Drop labels where either model has no valid AUROC
    df = df.dropna(subset=["gnn_auroc", "trans_auroc"])
    df = df[(df["gnn_auroc"] != "nan") & (df["trans_auroc"] != "nan")]
    df["gnn_auroc"]   = df["gnn_auroc"].astype(float)
    df["trans_auroc"] = df["trans_auroc"].astype(float)

    n_labels  = len(df)
    gnn_vals  = df["gnn_auroc"].values
    trans_vals = df["trans_auroc"].values
    diff       = gnn_vals - trans_vals

    log.info("  Labels evaluated: %d", n_labels)
    log.info("  Mean AUROC — GNN: %.4f  TRANS: %.4f", gnn_vals.mean(), trans_vals.mean())
    log.info("  Mean difference (GNN − TRANS): %+.4f", diff.mean())
    log.info("  GNN wins: %d / %d labels (%.1f%%)",
             (diff > 0).sum(), n_labels, 100 * (diff > 0).mean())

    # ---- Wilcoxon signed-rank test ------------------------------------------
    # Non-parametric: tests whether the distribution of differences is symmetric
    # around zero. More appropriate than t-test for bounded AUROC values.
    w_stat, w_p = stats.wilcoxon(gnn_vals, trans_vals, alternative="greater")
    log.info("\n  Wilcoxon signed-rank test (one-sided: GNN > TRANS)")
    log.info("    Statistic = %.4f  p-value = %.6f", w_stat, w_p)
    log.info("    Significant (p < 0.05): %s", "YES" if w_p < 0.05 else "NO")

    # ---- Paired t-test -------------------------------------------------------
    # Parametric: assumes differences are normally distributed.
    # Complementary to Wilcoxon; useful for reporting alongside.
    t_stat, t_p_twosided = stats.ttest_rel(gnn_vals, trans_vals)
    t_p_onesided = t_p_twosided / 2 if t_stat > 0 else 1 - t_p_twosided / 2
    log.info("\n  Paired t-test (one-sided: GNN > TRANS)")
    log.info("    t-statistic = %.4f  p-value (one-sided) = %.6f", t_stat, t_p_onesided)
    log.info("    Significant (p < 0.05): %s", "YES" if t_p_onesided < 0.05 else "NO")

    # ---- Per frequency quartile breakdown ------------------------------------
    quartile_results = {}
    if "freq_quartile" in df.columns:
        log.info("\n  Wilcoxon per frequency quartile:")
        for q in sorted(df["freq_quartile"].unique()):
            mask = df["freq_quartile"] == q
            sub_gnn   = gnn_vals[mask.values]
            sub_trans = trans_vals[mask.values]
            if len(sub_gnn) < 5:
                continue
            try:
                wq_stat, wq_p = stats.wilcoxon(sub_gnn, sub_trans, alternative="greater")
                sig = "YES" if wq_p < 0.05 else "NO"
                log.info("    Q%d (n=%d): stat=%.4f  p=%.4f  sig=%s",
                         q, len(sub_gnn), wq_stat, wq_p, sig)
                quartile_results[f"Q{int(q)}"] = {
                    "n_labels":   int(len(sub_gnn)),
                    "mean_gnn":   float(sub_gnn.mean()),
                    "mean_trans": float(sub_trans.mean()),
                    "wilcoxon_stat": float(wq_stat),
                    "wilcoxon_p":    float(wq_p),
                    "significant":   bool(wq_p < 0.05),
                }
            except Exception as e:
                log.warning("    Q%d: could not run Wilcoxon (%s)", q, e)

    return {
        "test": "paired_label_auroc",
        "n_labels": int(n_labels),
        "mean_gnn_auroc":   float(gnn_vals.mean()),
        "mean_trans_auroc": float(trans_vals.mean()),
        "mean_diff":        float(diff.mean()),
        "gnn_wins_count":   int((diff > 0).sum()),
        "gnn_wins_pct":     float(100 * (diff > 0).mean()),
        "wilcoxon": {
            "statistic": float(w_stat),
            "p_value":   float(w_p),
            "alternative": "greater",
            "significant_at_0.05": bool(w_p < 0.05),
        },
        "paired_ttest": {
            "statistic":   float(t_stat),
            "p_value_onesided": float(t_p_onesided),
            "p_value_twosided": float(t_p_twosided),
            "alternative": "greater",
            "significant_at_0.05": bool(t_p_onesided < 0.05),
        },
        "per_quartile": quartile_results,
    }


# ---------------------------------------------------------------------------
# Test 3: Bootstrap confidence intervals on macro AUROC
# ---------------------------------------------------------------------------

def _macro_auroc(y_true: np.ndarray, y_prob: np.ndarray,
                 min_positives: int = 3) -> float:
    """Macro AUROC over labels with at least min_positives true positives."""
    aurocs = []
    for i in range(y_true.shape[1]):
        pos = y_true[:, i].sum()
        if pos >= min_positives and pos < len(y_true):
            aurocs.append(roc_auc_score(y_true[:, i], y_prob[:, i]))
    return float(np.mean(aurocs)) if aurocs else float("nan")


def bootstrap_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
    seed: int = 42,
    label: str = "model",
) -> dict:
    """
    Bootstrap 95% CI on macro AUROC by resampling patients with replacement.

    Parameters
    ----------
    y_true      : [N, num_labels] ground truth
    y_prob      : [N, num_labels] predicted probabilities
    n_bootstrap : number of resamples
    ci_level    : confidence level (default 0.95)
    seed        : random seed for reproducibility
    label       : model name for logging

    Returns
    -------
    dict with observed AUROC and CI bounds
    """
    rng = np.random.default_rng(seed)
    N   = len(y_true)

    observed = _macro_auroc(y_true, y_prob)
    log.info("  %s — observed macro AUROC: %.4f", label, observed)

    boot_scores = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, N, N)          # resample patients with replacement
        yt  = y_true[idx]
        yp  = y_prob[idx]
        boot_scores.append(_macro_auroc(yt, yp))

    boot_scores = np.array(boot_scores)
    alpha = 1 - ci_level
    lo, hi = np.percentile(boot_scores, [100 * alpha / 2, 100 * (1 - alpha / 2)])

    log.info("    %d%% CI: [%.4f, %.4f]", int(ci_level * 100), lo, hi)

    return {
        "model":           label,
        "observed_auroc":  float(observed),
        "ci_lower":        float(lo),
        "ci_upper":        float(hi),
        "ci_level":        ci_level,
        "n_bootstrap":     n_bootstrap,
        "boot_mean":       float(boot_scores.mean()),
        "boot_std":        float(boot_scores.std()),
    }


def bootstrap_difference_ci(
    gnn_y_true: np.ndarray,  gnn_y_prob: np.ndarray,
    trans_y_true: np.ndarray, trans_y_prob: np.ndarray,
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
    seed: int = 42,
) -> dict:
    """
    Bootstrap CI on the AUROC difference (E-TRANS minus TRANS).

    Each resample draws from each model's test set independently
    (both sets may differ in patient composition if test splits differ).

    Returns dict with observed difference and CI bounds.
    """
    rng     = np.random.default_rng(seed)
    N_gnn   = len(gnn_y_true)
    N_trans = len(trans_y_true)

    observed_gnn   = _macro_auroc(gnn_y_true,   gnn_y_prob)
    observed_trans = _macro_auroc(trans_y_true, trans_y_prob)
    observed_diff  = observed_gnn - observed_trans

    log.info("  Bootstrap CI on AUROC difference (GNN − TRANS)")
    log.info("    Observed difference: %+.4f  (GNN=%.4f  TRANS=%.4f)",
             observed_diff, observed_gnn, observed_trans)

    diff_scores = []
    for _ in range(n_bootstrap):
        idx_g = rng.integers(0, N_gnn,   N_gnn)
        idx_t = rng.integers(0, N_trans, N_trans)
        a_g = _macro_auroc(gnn_y_true[idx_g],   gnn_y_prob[idx_g])
        a_t = _macro_auroc(trans_y_true[idx_t], trans_y_prob[idx_t])
        diff_scores.append(a_g - a_t)

    diff_scores = np.array(diff_scores)
    alpha = 1 - ci_level
    lo, hi = np.percentile(diff_scores, [100 * alpha / 2, 100 * (1 - alpha / 2)])

    # One-sided p-value: proportion of bootstrap samples where diff <= 0
    p_value = float((diff_scores <= 0).mean())

    log.info("    %d%% CI on difference: [%+.4f, %+.4f]", int(ci_level * 100), lo, hi)
    log.info("    Bootstrap p-value (GNN > TRANS): %.4f", p_value)
    log.info("    Significant (p < 0.05): %s", "YES" if p_value < 0.05 else "NO")

    return {
        "observed_gnn_auroc":   float(observed_gnn),
        "observed_trans_auroc": float(observed_trans),
        "observed_diff":        float(observed_diff),
        "ci_lower":             float(lo),
        "ci_upper":             float(hi),
        "ci_level":             ci_level,
        "n_bootstrap":          n_bootstrap,
        "p_value_gnn_gt_trans": p_value,
        "significant_at_0.05":  bool(p_value < 0.05),
    }


def run_bootstrap_tests(args, gnn_y_true, gnn_y_prob,
                        trans_y_true, trans_y_prob) -> dict:
    """Run all bootstrap CI tests and return results dict."""
    log.info("=== Bootstrap Confidence Intervals (n=%d) ===", args.n_bootstrap)

    gnn_ci   = bootstrap_ci(gnn_y_true,   gnn_y_prob,
                            n_bootstrap=args.n_bootstrap,
                            ci_level=args.ci_level, seed=args.seed,
                            label="E-TRANS (GNN)")
    trans_ci = bootstrap_ci(trans_y_true, trans_y_prob,
                            n_bootstrap=args.n_bootstrap,
                            ci_level=args.ci_level, seed=args.seed,
                            label="TRANS")
    diff_ci  = bootstrap_difference_ci(
        gnn_y_true, gnn_y_prob,
        trans_y_true, trans_y_prob,
        n_bootstrap=args.n_bootstrap,
        ci_level=args.ci_level,
        seed=args.seed,
    )

    return {
        "test":       "bootstrap_macro_auroc_ci",
        "gnn_ci":     gnn_ci,
        "trans_ci":   trans_ci,
        "difference": diff_ci,
    }


# ---------------------------------------------------------------------------
# GNN inference (used when --gnn_preds absent and --gnn_ckpt provided)
# ---------------------------------------------------------------------------

def run_gnn_inference(args):
    """
    Load PatientGNN checkpoint and run inference on the test split.
    Returns (y_true [N, num_labels], y_prob [N, num_labels]).
    Mirrors the inference logic in analyse_per_label.py.
    """
    import torch
    from torch_geometric.data import DataLoader
    from tqdm import tqdm

    gnn_src = Path(__file__).parent.parent / "src" / "gnn"
    if str(gnn_src) not in sys.path:
        sys.path.insert(0, str(gnn_src))
    from model import PatientGNN  # noqa

    log.info("Loading patient graphs from %s …", args.graphs_path)
    graphs = torch.load(args.graphs_path, weights_only=False)
    graphs = [g for g in graphs if hasattr(g, "y_diagnosis")]
    log.info("  %d graphs with y_diagnosis", len(graphs))

    graphs_dir = Path(args.graphs_path).parent
    import json as _json
    with open(graphs_dir / "label_vocab.json") as f:
        label_vocab = _json.load(f)
    num_labels = len(label_vocab)

    with open(graphs_dir / "concept_vocab.json") as f:
        concept_vocab = _json.load(f)

    def _vocab_size(domain):
        d = concept_vocab.get(domain, {})
        return max(int(v) for v in d.values()) + 1 if d else 0

    vocab_sizes = {k: _vocab_size(k) for k in ("condition", "procedure", "drug")}

    # Normalise visit features (identical to train.py)
    all_x = torch.cat([g["visit"].x for g in graphs], dim=0)
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

    rng    = np.random.default_rng(args.seed)
    idx    = rng.permutation(len(graphs))
    n_test = int(len(graphs) * args.test_frac)
    test_graphs = [graphs[i] for i in idx[:n_test]]
    log.info("  Test split: %d patients", len(test_graphs))

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model  = PatientGNN.from_config(
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
    log.info("  Loaded checkpoint (epoch %d, val AUROC %.4f)",
             ckpt["epoch"], ckpt["val_metrics"].get("auroc_macro", float("nan")))

    import torch.nn.functional as F
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
            y_prob_list.append(F.sigmoid(logits).cpu().float().numpy())

    y_true = np.concatenate(y_true_list, axis=0)
    y_prob = np.nan_to_num(np.concatenate(y_prob_list, axis=0), nan=0.5)
    log.info("  Inference done: %d patients  %d labels", *y_true.shape)
    return y_true, y_prob


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------

def write_report(results: dict, out_dir: Path, ci_level: float):
    """Write a human-readable plain-text summary report."""
    lines = []
    lines.append("=" * 65)
    lines.append("  STATISTICAL SIGNIFICANCE REPORT")
    lines.append("  E-TRANS (PatientGNN OMOP) vs TRANS (ICD baseline)")
    lines.append("=" * 65)

    # ---- Paired tests -------------------------------------------------------
    if "paired_tests" in results:
        pt = results["paired_tests"]
        lines.append("\n--- Test 1: Wilcoxon Signed-Rank Test (per-label AUROC) ---")
        lines.append(f"  Labels evaluated : {pt['n_labels']}")
        lines.append(f"  Mean GNN AUROC   : {pt['mean_gnn_auroc']:.4f}")
        lines.append(f"  Mean TRANS AUROC : {pt['mean_trans_auroc']:.4f}")
        lines.append(f"  Mean difference  : {pt['mean_diff']:+.4f}")
        lines.append(f"  GNN wins         : {pt['gnn_wins_count']} / {pt['n_labels']} "
                     f"labels ({pt['gnn_wins_pct']:.1f}%)")
        w = pt["wilcoxon"]
        lines.append(f"  Statistic        : {w['statistic']:.4f}")
        lines.append(f"  p-value          : {w['p_value']:.6f}")
        lines.append(f"  Significant      : {'YES ✓' if w['significant_at_0.05'] else 'NO'}")

        lines.append("\n--- Test 2: Paired t-test (per-label AUROC) ---")
        t = pt["paired_ttest"]
        lines.append(f"  t-statistic      : {t['statistic']:.4f}")
        lines.append(f"  p-value (1-sided): {t['p_value_onesided']:.6f}")
        lines.append(f"  Significant      : {'YES ✓' if t['significant_at_0.05'] else 'NO'}")

        if pt.get("per_quartile"):
            lines.append("\n  Wilcoxon per frequency quartile:")
            for qname, qr in pt["per_quartile"].items():
                lines.append(
                    f"    {qname} (n={qr['n_labels']:3d}): "
                    f"GNN={qr['mean_gnn']:.4f}  TRANS={qr['mean_trans']:.4f}  "
                    f"p={qr['wilcoxon_p']:.4f}  "
                    f"{'sig ✓' if qr['significant'] else '    '}"
                )

    # ---- Bootstrap ----------------------------------------------------------
    if "bootstrap" in results:
        bt = results["bootstrap"]
        ci_pct = int(ci_level * 100)
        lines.append(f"\n--- Test 3: Bootstrap {ci_pct}% Confidence Intervals ---")
        g = bt["gnn_ci"]
        lines.append(f"  E-TRANS macro AUROC : {g['observed_auroc']:.4f} "
                     f"[{g['ci_lower']:.4f}, {g['ci_upper']:.4f}]")
        t = bt["trans_ci"]
        lines.append(f"  TRANS   macro AUROC : {t['observed_auroc']:.4f} "
                     f"[{t['ci_lower']:.4f}, {t['ci_upper']:.4f}]")
        d = bt["difference"]
        lines.append(f"  Difference (GNN−TRANS): {d['observed_diff']:+.4f} "
                     f"[{d['ci_lower']:+.4f}, {d['ci_upper']:+.4f}]")
        lines.append(f"  Bootstrap p-value   : {d['p_value_gnn_gt_trans']:.4f}")
        lines.append(f"  Significant         : "
                     f"{'YES ✓' if d['significant_at_0.05'] else 'NO'}")

    lines.append("\n" + "=" * 65)
    report = "\n".join(lines)
    print(report)

    report_path = out_dir / "statistical_results.txt"
    report_path.write_text(report, encoding="utf-8")
    log.info("Report saved to %s", report_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    # ---- Tests 1 + 2: Wilcoxon and paired t-test ----------------------------
    if args.per_label_csv and Path(args.per_label_csv).exists():
        paired_results = run_paired_tests(args.per_label_csv)
        all_results["paired_tests"] = paired_results
    else:
        if args.per_label_csv:
            log.warning("per_label_csv not found: %s — skipping paired tests",
                        args.per_label_csv)
        else:
            log.info("--per_label_csv not provided — skipping Wilcoxon and t-test")

    # ---- Test 3: Bootstrap CI -----------------------------------------------
    gnn_y_true = gnn_y_prob = trans_y_true = trans_y_prob = None

    # Option A: load pre-saved prediction arrays
    if args.gnn_preds and Path(args.gnn_preds).exists():
        log.info("Loading GNN predictions from %s", args.gnn_preds)
        npz = np.load(args.gnn_preds)
        gnn_y_true, gnn_y_prob = npz["y_true"], npz["y_prob"]

    if args.trans_preds and Path(args.trans_preds).exists():
        log.info("Loading TRANS predictions from %s", args.trans_preds)
        npz = np.load(args.trans_preds)
        trans_y_true, trans_y_prob = npz["y_true"], npz["y_prob"]

    # Option B: re-run GNN inference from checkpoint
    if gnn_y_true is None and args.gnn_ckpt and args.graphs_path:
        log.info("Running GNN inference from checkpoint …")
        gnn_y_true, gnn_y_prob = run_gnn_inference(args)

        # Save for reuse (avoids re-running inference next time)
        save_path = out_dir / "gnn_predictions.npz"
        np.savez(save_path, y_true=gnn_y_true, y_prob=gnn_y_prob)
        log.info("Saved GNN predictions to %s (reuse with --gnn_preds)", save_path)

    if gnn_y_true is not None and trans_y_true is not None:
        bootstrap_results = run_bootstrap_tests(
            args, gnn_y_true, gnn_y_prob, trans_y_true, trans_y_prob
        )
        all_results["bootstrap"] = bootstrap_results

    elif gnn_y_true is not None and trans_y_true is None:
        # Run bootstrap CI for GNN alone (no TRANS predictions available)
        log.info("=== Bootstrap CI for GNN only (TRANS predictions not provided) ===")
        gnn_ci = bootstrap_ci(
            gnn_y_true, gnn_y_prob,
            n_bootstrap=args.n_bootstrap,
            ci_level=args.ci_level,
            seed=args.seed,
            label="E-TRANS (GNN)",
        )
        all_results["bootstrap"] = {
            "test": "bootstrap_macro_auroc_ci",
            "gnn_ci": gnn_ci,
            "note": "TRANS predictions not provided — difference CI skipped",
        }

    else:
        log.info("No prediction arrays available — skipping bootstrap CI")
        log.info("  Provide --gnn_preds / --trans_preds  OR  --gnn_ckpt + --graphs_path")

    # ---- Save JSON results --------------------------------------------------
    if all_results:
        json_path = out_dir / "statistical_results.json"
        with open(json_path, "w") as f:
            json.dump(all_results, f, indent=2)
        log.info("Results saved to %s", json_path)

        write_report(all_results, out_dir, args.ci_level)
    else:
        log.warning("No tests were run — check that input files are provided and exist.")


if __name__ == "__main__":
    main()
