"""
analyse_subgroup.py
====================
Clinical subgroup analysis for PatientGNN E11.

Evaluates PatientGNN performance on a specific disease patient subgroup
and compares it against the full test cohort. Also extracts per-label
AUROC for disease-related condition codes from the existing
per_label_results.csv.

Supports built-in disease groups:
  --disease diabetes        Type 1/2 diabetes and common complications
  --disease cancer          Malignant neoplasms and cancer complications
  --disease heart_failure   Heart failure and related cardiac conditions
  --disease ckd             Chronic kidney disease and renal conditions

A patient belongs to a disease subgroup if ANY of the disease's primary
SNOMED codes appears in their y_diagnosis label vector (last-visit labels).

Outputs (written to --out_dir)
-------------------------------
  subgroup_label_results.csv     per-label AUROC for disease labels (from CSV)
  fig_disease_labels.png         bar chart: GNN vs TRANS per disease label
  subgroup_gnn_results.json      GNN AUROC full cohort vs disease subgroup
  fig_subgroup_auroc.png         bar chart: full vs subgroup macro-AUROC

Usage
-----
python analyse_subgroup.py \\
  --disease         diabetes \\
  --per_label_csv   analysis/per_label/per_label_results.csv \\
  --gnn_ckpt        logs/e11/checkpoints/best_model.pt \\
  --graphs_path     data/processed/patient_graphs.pt \\
  --out_dir         analysis/diabetes_subgroup \\
  --device          cuda:0
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
# Disease SNOMED definitions
# ---------------------------------------------------------------------------

# NOTE: These are OMOP concept IDs (integers stored as strings), NOT SNOMED CT codes.
# OMOP maps SNOMED concepts to its own numeric concept_id space.
# e.g. SNOMED 44054006 (Type 2 DM) → OMOP concept_id 201826
#
# Run this to discover which disease codes are in your label_vocab.json:
#   python3 -c "
#   import json
#   with open('data/processed/label_vocab.json') as f: codes=json.load(f)
#   for i,c in enumerate(codes): print(i, c)
#   "
# Then cross-reference with the OMOP concept table or athena.ohdsi.org.

DISEASE_SNOMED = {
    "diabetes": {
        # OMOP concept IDs for diabetes primary diagnoses
        "primary": {
            "201826",   # Type 2 diabetes mellitus
            "443238",   # Diabetes mellitus (parent)
            "4099651",  # Type 1 diabetes mellitus
            "201254",   # Diabetes mellitus type 1 (alt)
            "380834",   # Diabetic ketoacidosis
            "376065",   # Diabetic foot
            "4159131",  # Diabetic retinopathy
            "201914",   # Diabetic peripheral neuropathy
            "433736",   # Hypoglycaemia
            "192279",   # Diabetic nephropathy
            "4306655",  # Gestational diabetes mellitus
            "4230254",  # Type 2 DM without complication
            "442793",   # Type 2 DM with neurological manifestation
            "4171755",  # Type 2 DM with renal manifestation
        },
        # Common comorbidities/complications in diabetic patients
        "complications": {
            "320128",   # Essential hypertension
            "432867",   # Hypertensive disorder
            "317576",   # Coronary arteriosclerosis
            "316139",   # Heart failure
            "319835",   # Systolic heart failure
            "313217",   # Atrial fibrillation
            "318800",   # Chronic kidney disease
            "197320",   # Proteinuria / renal disorder
            "440383",   # Anaemia
            "255573",   # Chronic obstructive lung disease
            "4208223",  # Peripheral vascular disease
            "4119134",  # Peripheral neuropathy
            "74285",    # Obesity
        },
    },
    "cancer": {
        # OMOP concept IDs for malignant neoplasms
        "primary": {
            "438112",   # Malignant neoplasm of large intestine
            "4248893",  # Primary malignant neoplasm of lung
            "432571",   # Neoplasm of large bowel
            "436313",   # Leukaemia
            "4112853",  # Carcinoma of lung
            "192855",   # Malignant neoplasm of colon
            "4180791",  # Malignant neoplasm of bronchus / lung
            "4179242",  # Malignant neoplasm of pancreas
            "4162252",  # Malignant neoplasm of bladder
            "198979",   # Malignant neoplasm of breast
            "4115276",  # Malignant neoplasm of prostate
            "433753",   # Malignant neoplasm of kidney
            "4032806",  # Secondary malignant neoplasm of liver
            "4152282",  # Multiple myeloma
            "4033240",  # Non-Hodgkin lymphoma
            "4180831",  # Acute myeloid leukaemia
        },
        "complications": {
            "432870",   # Anaemia (in context of cancer)
            "440383",   # Anaemia
            "439777",   # Neutropenia
            "4119134",  # Neuropathy (chemo-induced)
            "444132",   # Deep vein thrombosis
            "440417",   # Pulmonary embolism
            "132797",   # Sepsis
            "436965",   # Hypercalcaemia
        },
    },
    "heart_failure": {
        "primary": {
            "316139",   # Heart failure
            "319835",   # Chronic systolic heart failure
            "443580",   # Acute systolic heart failure
            "4229440",  # Left heart failure
            "4209497",  # Chronic right-sided heart failure
            "45766032", # Decompensated chronic heart failure
            "4108832",  # Biventricular heart failure
        },
        "complications": {
            "320128",   # Essential hypertension
            "432867",   # Hypertensive disorder
            "317576",   # Coronary artery disease
            "313217",   # Atrial fibrillation
            "318800",   # CKD
            "440383",   # Anaemia
            "255573",   # COPD
            "319049",   # Pulmonary oedema
            "40304217", # Respiratory failure
        },
    },
    "ckd": {
        "primary": {
            "318800",   # Chronic kidney disease (parent)
            "4030518",  # CKD stage 1
            "443597",   # CKD stage 2
            "443601",   # CKD stage 3
            "443612",   # CKD stage 4
            "443614",   # CKD stage 5
            "193782",   # End-stage renal disease
            "4189102",  # Acute-on-chronic renal failure
        },
        "complications": {
            "201826",   # Type 2 diabetes
            "320128",   # Hypertension
            "432867",   # Hypertensive disorder
            "316139",   # Heart failure
            "440383",   # Anaemia
            "4281516",  # Hyperphosphataemia
            "434821",   # Hyperkalaemia
        },
    },
}


# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Clinical subgroup analysis for PatientGNN E11"
    )
    p.add_argument("--disease", required=True,
                   choices=list(DISEASE_SNOMED.keys()),
                   help="Disease group to analyse")
    p.add_argument("--per_label_csv", default=None,
                   help="Path to per_label_results.csv from analyse_per_label.py "
                        "(used for Part 1 label-level analysis; optional)")
    p.add_argument("--gnn_ckpt",    required=True,
                   help="Path to PatientGNN E11 best_model.pt")
    p.add_argument("--graphs_path", required=True,
                   help="Path to patient_graphs.pt")
    p.add_argument("--out_dir",     default="analysis/subgroup")
    p.add_argument("--device",      default="cuda:0")
    p.add_argument("--batch_size",  type=int,   default=64)
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--val_frac",    type=float, default=0.1)
    p.add_argument("--test_frac",   type=float, default=0.15)
    p.add_argument("--min_positives", type=int, default=5,
                   help="Min positive cases per label for AUROC (default: 5)")

    # Model hparams — must match E11
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
# Part 1: Label-level analysis from per_label_results.csv
# ---------------------------------------------------------------------------

def part1_label_analysis(csv_path, disease, out_dir):
    """
    Filter per_label_results.csv to disease-related SNOMED codes.
    Produces a focused bar chart and CSV for those labels.
    """
    log.info("=== Part 1: %s label analysis from CSV ===", disease.upper())

    primary_codes     = DISEASE_SNOMED[disease]["primary"]
    complication_codes = DISEASE_SNOMED[disease]["complications"]
    all_disease_codes  = primary_codes | complication_codes

    rows = []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)

    primary_rows     = [r for r in rows if r["snomed_code"] in primary_codes]
    complication_rows = [r for r in rows if r["snomed_code"] in complication_codes]
    all_disease_rows  = primary_rows + complication_rows

    log.info("  Primary %s labels in top-275:      %d", disease, len(primary_rows))
    log.info("  Complication labels in top-275:    %d", len(complication_rows))

    if not all_disease_rows:
        log.warning("  No %s SNOMED codes found in per_label_results.csv. "
                    "The top-275 may not contain these exact codes.", disease)
        return

    # Save disease-specific CSV
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, f"{disease}_label_results.csv")
    fieldnames = list(rows[0].keys()) + ["category"]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in primary_rows:
            w.writerow({**r, "category": "primary"})
        for r in complication_rows:
            w.writerow({**r, "category": "complication"})
    log.info("  Saved %s", out_csv)

    # Print summary
    for name, group in [("Primary", primary_rows), ("Complications", complication_rows)]:
        valid = [(float(r["trans_auroc"]), float(r["gnn_auroc"]))
                 for r in group
                 if r["trans_auroc"] != "nan" and r["gnn_auroc"] != "nan"]
        if valid:
            t_mean = np.mean([v[0] for v in valid])
            g_mean = np.mean([v[1] for v in valid])
            log.info("  %-15s  TRANS=%.4f  GNN=%.4f  Gap=%+.4f",
                     name, t_mean, g_mean, g_mean - t_mean)

    # Plot
    _plot_disease_labels(primary_rows, complication_rows, disease, out_dir)


def _plot_disease_labels(primary, complication, disease, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = [
        (f"Primary {disease} diagnoses", primary,    "#C44E52"),
        ("Complications / comorbidities", complication, "#4C72B0"),
    ]

    n_panels = sum(1 for _, g, _ in groups if g)
    if n_panels == 0:
        return

    fig, axes = plt.subplots(1, max(n_panels, 2), figsize=(8 * n_panels, max(6, 8)))
    if n_panels == 1:
        axes = [axes, axes]

    ax_idx = 0
    for title, group, color in groups:
        if not group:
            continue
        ax = axes[ax_idx]
        ax_idx += 1

        valid = [(r["snomed_code"][:22], float(r["trans_auroc"]), float(r["gnn_auroc"]))
                 for r in group
                 if r["trans_auroc"] != "nan" and r["gnn_auroc"] != "nan"]
        if not valid:
            ax.set_title(f"{title}\n(no valid AUROC)")
            continue

        valid.sort(key=lambda x: x[2])   # sort by GNN AUROC
        codes  = [v[0] for v in valid]
        t_vals = [v[1] for v in valid]
        g_vals = [v[2] for v in valid]

        y = np.arange(len(codes))
        ax.barh(y - 0.2, t_vals, height=0.35, label="TRANS E7",
                color="#4C72B0", alpha=0.8)
        ax.barh(y + 0.2, g_vals, height=0.35, label="PatientGNN E11",
                color="#DD8452", alpha=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels(codes, fontsize=9)
        ax.set_xlabel("Per-label AUROC", fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlim(0.4, 1.0)
        ax.axvline(0.5, color="grey", linewidth=0.5, linestyle="--")
        ax.legend(fontsize=9)
        ax.grid(True, axis="x", alpha=0.3)
        ax.invert_yaxis()

    fig.suptitle(
        f"PatientGNN E11 vs TRANS E7 — {disease.capitalize()}-related SNOMED labels",
        fontsize=13, fontweight="bold"
    )
    plt.tight_layout()
    path = os.path.join(out_dir, "fig_disease_labels.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info("  Saved %s", path)


# ---------------------------------------------------------------------------
# Part 2: GNN inference on disease patient subgroup
# ---------------------------------------------------------------------------

def load_and_preprocess(args):
    """Load graphs, normalise, remove leakage, split. Same as train.py."""
    log.info("Loading %s …", args.graphs_path)
    graphs = torch.load(args.graphs_path, weights_only=False)
    graphs = [g for g in graphs if hasattr(g, "y_diagnosis")]
    log.info("  %d graphs with y_diagnosis", len(graphs))

    graphs_dir = Path(args.graphs_path).parent
    with open(graphs_dir / "label_vocab.json") as f:
        label_vocab = json.load(f)

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

    # Normalise visit features
    all_x    = torch.cat([g["visit"].x for g in graphs], dim=0)
    mean4, std4 = float(all_x[:, 4].mean()), float(all_x[:, 4].std()) + 1e-6
    mean5, std5 = float(all_x[:, 5].mean()), float(all_x[:, 5].std()) + 1e-6
    for g in graphs:
        x = g["visit"].x.clone()
        x[:, 3] = torch.log1p(x[:, 3])
        x[:, 4] = (x[:, 4] - mean4) / std4
        x[:, 5] = (x[:, 5] - mean5) / std5
        g["visit"].x = x

    # Leakage prevention
    for g in graphs:
        last_v = g["visit"].x.size(0) - 1
        ei = g["visit", "has_condition", "condition"].edge_index
        if ei.numel() > 0:
            g["visit", "has_condition", "condition"].edge_index = ei[:, ei[0] != last_v]

    # Test split (same seed as E11 training)
    rng    = np.random.default_rng(args.seed)
    idx    = rng.permutation(len(graphs))
    n_test = int(len(graphs) * args.test_frac)
    test_graphs = [graphs[i] for i in idx[:n_test]]
    log.info("  Test split: %d patients", len(test_graphs))

    return test_graphs, label_vocab, vocab_sizes


def identify_disease_patients(test_graphs, label_vocab, disease):
    """
    Return boolean mask: True for patients with any primary disease SNOMED code
    in their y_diagnosis label.
    """
    primary_codes = DISEASE_SNOMED[disease]["primary"]
    disease_label_indices = [i for i, code in enumerate(label_vocab)
                              if code in primary_codes]

    log.info("  Primary %s codes found in label_vocab: %d  (indices: %s)",
             disease, len(disease_label_indices),
             [label_vocab[i] for i in disease_label_indices[:5]])

    if not disease_label_indices:
        log.warning("  No primary %s SNOMED codes in label_vocab. "
                    "Falling back to ALL disease codes (primary + complications).", disease)
        all_codes = DISEASE_SNOMED[disease]["primary"] | DISEASE_SNOMED[disease]["complications"]
        disease_label_indices = [i for i, code in enumerate(label_vocab)
                                  if code in all_codes]

    mask = np.zeros(len(test_graphs), dtype=bool)
    for i, g in enumerate(test_graphs):
        if disease_label_indices:
            mask[i] = bool(g.y_diagnosis[disease_label_indices].any().item())

    log.info("  %s patients in test set: %d / %d  (%.1f%%)",
             disease.capitalize(), mask.sum(), len(test_graphs),
             100 * mask.mean())
    return mask, disease_label_indices


@torch.no_grad()
def run_gnn_inference(graphs_list, model, device, batch_size, desc):
    loader = DataLoader(graphs_list, batch_size=batch_size,
                        shuffle=False, num_workers=0)
    y_true_all, y_prob_all = [], []
    for batch in tqdm(loader, desc=desc, leave=False):
        batch   = batch.to(device)
        out     = model(batch, task="diagnosis")
        logits  = out["diagnosis"]
        labels  = batch.y_diagnosis.view(logits.shape)
        y_true_all.append(labels.cpu().float().numpy())
        y_prob_all.append(torch.sigmoid(logits).cpu().float().numpy())
    y_true = np.concatenate(y_true_all, axis=0)
    y_prob = np.nan_to_num(np.concatenate(y_prob_all, axis=0), nan=0.5)
    return y_true, y_prob


def per_label_auroc(y_true, y_prob, min_positives=5):
    n = y_true.shape[1]
    out = np.full(n, np.nan)
    for i in range(n):
        if y_true[:, i].sum() >= min_positives:
            try:
                out[i] = roc_auc_score(y_true[:, i], y_prob[:, i])
            except Exception:
                pass
    return out


def compute_metrics(y_true, y_prob, label_indices=None, min_positives=5):
    """Return macro AUROC, AUPRC, and per-label AUROC for label_indices."""
    auroc_all = per_label_auroc(y_true, y_prob, min_positives)
    macro_auroc = float(np.nanmean(auroc_all))

    # AUPRC (macro over labels with positives)
    auprcs = []
    for i in range(y_true.shape[1]):
        if y_true[:, i].sum() >= min_positives:
            try:
                auprcs.append(average_precision_score(y_true[:, i], y_prob[:, i]))
            except Exception:
                pass
    macro_auprc = float(np.mean(auprcs)) if auprcs else float("nan")

    # Subset AUROC for disease-specific labels
    if label_indices:
        subset_aurocs = [float(auroc_all[i]) if not np.isnan(auroc_all[i]) else float("nan")
                         for i in label_indices]
        subset_macro  = float(np.nanmean(subset_aurocs)) if subset_aurocs else float("nan")
    else:
        subset_aurocs, subset_macro = [], float("nan")

    return {
        "macro_auroc":      macro_auroc,
        "macro_auprc":      macro_auprc,
        "n_patients":       int(y_true.shape[0]),
        "n_labels_evaluated": int(np.sum(~np.isnan(auroc_all))),
        "disease_label_auroc_mean": subset_macro,
        "disease_label_aurocs":     subset_aurocs,
    }


def part2_subgroup_inference(args, device):
    """PatientGNN inference: full test cohort vs disease subgroup."""
    log.info("=== Part 2: PatientGNN E11 — %s subgroup inference ===",
             args.disease.upper())

    gnn_src = Path(__file__).parent.parent / "src" / "gnn"
    if not gnn_src.exists():
        gnn_src = Path(__file__).parent / "src" / "gnn"
    if str(gnn_src) not in sys.path:
        sys.path.insert(0, str(gnn_src))
    from model import PatientGNN  # noqa

    test_graphs, label_vocab, vocab_sizes = load_and_preprocess(args)
    num_labels = len(label_vocab)

    disease_mask, disease_label_indices = identify_disease_patients(
        test_graphs, label_vocab, args.disease
    )

    subgroup_graphs = [g for g, flag in zip(test_graphs, disease_mask) if flag]
    if len(subgroup_graphs) < 20:
        log.warning("  Only %d %s patients in test set — AUROC estimates may be "
                    "unreliable. Consider lowering --test_frac or using more data.",
                    len(subgroup_graphs), args.disease)

    # Build model
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
    log.info("  Loaded checkpoint — epoch %d, val AUROC %.4f",
             ckpt.get("epoch", -1),
             ckpt.get("val_metrics", {}).get("auroc_macro", float("nan")))

    # Inference
    results = {}
    for name, subset in [
        ("full_test",         test_graphs),
        (f"{args.disease}_subgroup", subgroup_graphs),
    ]:
        if not subset:
            log.warning("  Skipping %s — empty subset", name)
            continue
        y_true, y_prob = run_gnn_inference(
            subset, model, device, args.batch_size, desc=name
        )
        results[name] = compute_metrics(
            y_true, y_prob, disease_label_indices, args.min_positives
        )
        results[name]["label_vocab_subset"] = [label_vocab[i] for i in disease_label_indices]

    return results, label_vocab, disease_label_indices


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def save_outputs(results, label_vocab, disease_label_indices, args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(args.out_dir, exist_ok=True)
    disease = args.disease

    # JSON
    json_path = os.path.join(args.out_dir, "subgroup_gnn_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info("Saved %s", json_path)

    # Bar chart: full cohort vs subgroup macro-AUROC and AUPRC
    subgroup_key = f"{disease}_subgroup"
    if "full_test" not in results or subgroup_key not in results:
        log.warning("Missing inference results — skipping figure")
        return

    full = results["full_test"]
    sub  = results[subgroup_key]

    metrics = ["macro_auroc", "macro_auprc", "disease_label_auroc_mean"]
    labels  = ["Macro-AUROC\n(all 275 labels)",
               "Macro-AUPRC\n(all 275 labels)",
               f"Macro-AUROC\n({disease} labels only)"]
    full_vals = [full[m] for m in metrics]
    sub_vals  = [sub[m]  for m in metrics]

    x = np.arange(len(metrics))
    fig, ax = plt.subplots(figsize=(10, 6))
    w = 0.35
    b1 = ax.bar(x - w/2, full_vals, w, label="Full test cohort",
                color="#4C72B0", alpha=0.85)
    b2 = ax.bar(x + w/2, sub_vals,  w, label=f"{disease.capitalize()} subgroup",
                color="#DD8452", alpha=0.85)

    # Value labels on bars
    for bar in list(b1) + list(b2):
        h = bar.get_height()
        if not np.isnan(h):
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                    f"{h:.4f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.set_title(
        f"PatientGNN E11 — Full Cohort vs {disease.capitalize()} Subgroup\n"
        f"(n_full={full['n_patients']}, n_{disease}={sub['n_patients']})",
        fontsize=12
    )
    ax.legend(fontsize=11)
    ax.grid(True, axis="y", alpha=0.3)
    ax.axhline(0.5, color="red", linewidth=0.8, linestyle="--", alpha=0.5,
               label="Random baseline")
    plt.tight_layout()
    path = os.path.join(args.out_dir, "fig_subgroup_auroc.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info("Saved %s", path)

    # Console summary
    log.info("\n=== %s Subgroup Summary ===", disease.upper())
    log.info("  %-35s  %-12s  %-12s", "Metric", "Full cohort", f"{disease} subgroup")
    log.info("  %-35s  %-12.4f  %-12.4f", "Macro-AUROC (all labels)",
             full["macro_auroc"], sub["macro_auroc"])
    log.info("  %-35s  %-12.4f  %-12.4f", "Macro-AUPRC (all labels)",
             full["macro_auprc"], sub["macro_auprc"])
    log.info("  %-35s  %-12.4f  %-12.4f",
             f"Macro-AUROC ({disease} labels only)",
             full["disease_label_auroc_mean"], sub["disease_label_auroc_mean"])
    log.info("  %-35s  %-12d  %-12d", "Patients evaluated",
             full["n_patients"], sub["n_patients"])

    if disease_label_indices:
        log.info("\n  Per-label AUROC — %s labels (full cohort vs subgroup):", disease)
        log.info("  %-15s  %-12s  %-12s", "SNOMED code", "Full AUROC", "Subgroup AUROC")
        for code, f_v, s_v in zip(
            results[subgroup_key]["label_vocab_subset"],
            full["disease_label_aurocs"],
            sub["disease_label_aurocs"],
        ):
            f_str = f"{f_v:.4f}" if not np.isnan(f_v) else "n/a"
            s_str = f"{s_v:.4f}" if not np.isnan(s_v) else "n/a"
            log.info("  %-15s  %-12s  %-12s", code, f_str, s_str)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    log.info("Device: %s  |  Disease: %s", device, args.disease)

    # Part 1: label-level analysis from existing CSV (optional)
    if args.per_label_csv and os.path.exists(args.per_label_csv):
        part1_label_analysis(args.per_label_csv, args.disease, args.out_dir)
    else:
        log.info("Skipping Part 1 — --per_label_csv not provided or not found")

    # Part 2: GNN subgroup inference
    results, label_vocab, disease_label_indices = part2_subgroup_inference(args, device)

    save_outputs(results, label_vocab, disease_label_indices, args)

    log.info("\nAll outputs written to: %s", args.out_dir)


if __name__ == "__main__":
    main()
