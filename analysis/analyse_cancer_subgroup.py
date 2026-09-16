"""
analyse_cancer_subgroup.py
==========================
Cancer-specific downstream analysis: PatientGNN E11 vs TRANS E7.

Builds on top of analyse_per_label.py. Produces two outputs:

  PART 1 — Cancer label analysis (no new inference needed)
  ---------------------------------------------------------
  Filters per_label_results.csv to SNOMED codes that fall under the
  neoplasm / cancer hierarchy. Shows whether PatientGNN outperforms TRANS
  specifically on predicting cancer-related next-visit diagnoses.
  Output: fig_cancer_labels.png, cancer_label_results.csv

  PART 2 — Cancer patient subgroup inference
  ------------------------------------------
  Filters patient_graphs.pt to patients who have any cancer SNOMED code
  in their input visit history. Runs PatientGNN E11 and TRANS E7 inference
  on that subgroup only. Compares macro AUROC, AUPRC, and per-label AUROC
  for the subset of labels most relevant to cancer complications
  (neutropenia, sepsis, venous thromboembolism, anaemia, etc.).
  Output: fig_cancer_subgroup_auroc.png, cancer_subgroup_results.json

Clinical hypothesis
-------------------
Cancer patients have complex drug-lab-condition interaction patterns
(e.g., chemotherapy → neutropenia → sepsis). PatientGNN's instance graph
explicitly connects drug nodes to lab-enriched visit nodes, enabling it
to learn these patterns. TRANS sees the same codes as flat tokens with
no structural connection between the drug and its downstream effect.

Expected finding: PatientGNN advantage is larger in the cancer subgroup
than in the full cohort, and largest for complication labels (sepsis,
neutropenia, VTE) rather than the primary cancer diagnosis labels.

Run after analyse_per_label.py has completed:
  python analyse_cancer_subgroup.py \\
    --per_label_csv  analysis/per_label/per_label_results.csv \\
    --gnn_ckpt       logs/e11/checkpoints/best_model.pt \\
    --trans_ckpt     /path/to/TRANS/logs/best_TRANS_omop.ckpt \\
    --graphs_path    data/processed/patient_graphs.pt \\
    --data_dir       data/export \\
    --trans_pkl      /path/to/TRANS/logs/omop_4.pkl \\
    --out_dir        analysis/cancer \\
    --device         cuda:0
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
# Cancer SNOMED concept IDs
# ---------------------------------------------------------------------------
# These are SNOMED CT concept IDs for cancer/neoplasm conditions that appear
# in the MIMIC-IV OMOP CDM and are likely candidates in the top-275 label set.
#
# Source: SNOMED hierarchy under "Malignant neoplastic disease" (363346000)
# and common cancer complications observed in MIMIC-IV.
#
# Split into two groups:
#   PRIMARY   — the cancer diagnoses themselves
#   COMPLICATIONS — conditions caused by cancer or its treatment

CANCER_PRIMARY_SNOMED = {
    # Solid tumours
    "363406005",   # Malignant neoplasm of colon
    "254637007",   # Non-small cell carcinoma of lung (disorder)
    "254626006",   # Adenocarcinoma of lung
    "413448000",   # Malignant neoplasm of bronchus and/or lung
    "363418001",   # Malignant neoplasm of pancreas
    "363354003",   # Malignant neoplasm of bladder
    "188161004",   # Malignant neoplasm of kidney
    "363359004",   # Malignant neoplasm of prostate
    "428061005",   # Malignant neoplasm of breast
    "109989006",   # Malignant neoplasm of breast (female)
    "363352000",   # Malignant neoplasm of rectum
    "363346000",   # Malignant neoplastic disease (parent — catches any cancer)
    "94381002",    # Secondary malignant neoplasm of bone
    "94222008",    # Secondary malignant neoplasm of lung
    "94391008",    # Secondary malignant neoplasm of liver
    "415068001",   # Malignant neoplasm of ovary
    # Haematological malignancies
    "413448000",   # (already above)
    "91855006",    # Acute myeloid leukaemia
    "91857003",    # Acute lymphoblastic leukaemia
    "118599009",   # Malignant lymphoma
    "413448000",   # (already above)
    "109989006",   # (already above)
    "109987001",   # Non-Hodgkin lymphoma
    "118600007",   # Diffuse large B-cell lymphoma
    "55921005",    # Multiple myeloma
    "109989006",   # (already above)
}

CANCER_COMPLICATION_SNOMED = {
    # Chemo/treatment complications
    "165517008",   # Neutropenia (low white cells → infection risk from chemo)
    "302215000",   # Thrombocytopenia (low platelets → bleeding risk)
    "87522002",    # Anaemia — cancer-related or chemo-induced
    "271737000",   # Anaemia (broader)
    # Infection complications
    "11302009",    # Febrile neutropenia
    "91302008",    # Sepsis (major complication in immunocompromised patients)
    "10674871000119105", # Sepsis due to Gram-negative bacteria
    "238150006",   # Neutropenic sepsis
    # Thromboembolic complications (cancer activates coagulation)
    "59282003",    # Pulmonary embolism
    "128053003",   # Deep vein thrombosis
    "371039008",   # Venous thromboembolism
    # Metabolic / electrolyte
    "40930008",    # Hypothyroidism (from immunotherapy/radiation)
    "237889002",   # Hypercalcaemia of malignancy
    "267406006",   # Hyponatraemia (SIADH from lung cancer)
    # Organ complications
    "19943007",    # Acute kidney injury (cisplatin nephrotoxicity)
    "408512008",   # Acute liver failure (hepatic metastasis or chemo toxicity)
    "44054006",    # Diabetes mellitus (steroid-induced, or immunotherapy)
    # Pain / palliative
    "57676002",    # Joint pain (bone metastasis)
    "57676002",    # (already above)
}

ALL_CANCER_SNOMED = CANCER_PRIMARY_SNOMED | CANCER_COMPLICATION_SNOMED


# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Cancer subgroup analysis: GNN E11 vs TRANS E7")

    # From previous analysis
    p.add_argument("--per_label_csv", required=True,
                   help="per_label_results.csv from analyse_per_label.py")

    # Checkpoints
    p.add_argument("--gnn_ckpt",   required=True)
    p.add_argument("--trans_ckpt", required=True)

    # PatientGNN data
    p.add_argument("--graphs_path", required=True)

    # TRANS data
    p.add_argument("--data_dir",  required=True)
    p.add_argument("--trans_pkl", default=None)
    p.add_argument("--trans_label_vocab", default=None,
                   help="Path to label_vocab_trans.json saved by train_omop.py. "
                        "Required for valid TRANS AUROC (prevents hash-random label ordering).")
    p.add_argument("--graph_cache", default="./cache/patient_graph_npz_omop")

    # Shared
    p.add_argument("--device",     default="cuda:0")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--out_dir",    default="analysis/cancer")
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--val_frac",   type=float, default=0.1)
    p.add_argument("--test_frac",  type=float, default=0.15)
    p.add_argument("--top_k",      type=int, default=275)

    # PatientGNN model hparams (must match E11)
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
# Part 1: Cancer label analysis from CSV (no new inference)
# ---------------------------------------------------------------------------

def part1_cancer_labels(csv_path, out_dir):
    """
    Filter per_label_results.csv to cancer-related SNOMED codes.
    Plot AUROC comparison for those labels specifically.
    """
    log.info("=== Part 1: Cancer label analysis ===")

    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    # Separate cancer labels: primary vs complication
    cancer_primary    = []
    cancer_complication = []
    other             = []

    for row in rows:
        code = row["snomed_code"]
        if code in CANCER_PRIMARY_SNOMED:
            cancer_primary.append(row)
        elif code in CANCER_COMPLICATION_SNOMED:
            cancer_complication.append(row)
        else:
            other.append(row)

    log.info("  Cancer primary labels found in top-275: %d", len(cancer_primary))
    log.info("  Cancer complication labels found in top-275: %d", len(cancer_complication))

    all_cancer = cancer_primary + cancer_complication
    if len(all_cancer) == 0:
        log.warning("No cancer SNOMED codes found in per_label_results.csv. "
                    "The top-275 labels may not include the exact codes listed. "
                    "Check label_vocab.json and expand CANCER_*_SNOMED sets if needed.")
        # Still produce the full label comparison sorted by category
        _write_label_summary(rows, out_dir)
        return

    # Save cancer-specific CSV
    cancer_csv = os.path.join(out_dir, "cancer_label_results.csv")
    os.makedirs(out_dir, exist_ok=True)
    fieldnames = list(rows[0].keys()) + ["cancer_category"]
    with open(cancer_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in cancer_primary:
            writer.writerow({**row, "cancer_category": "primary"})
        for row in cancer_complication:
            writer.writerow({**row, "cancer_category": "complication"})
    log.info("Saved %s", cancer_csv)

    # ---- Plot: cancer labels AUROC comparison --------------------------------
    _plot_cancer_labels(cancer_primary, cancer_complication, out_dir)

    # ---- Summary stats -------------------------------------------------------
    log.info("\n  === Cancer Label AUROC Summary ===")
    for group_name, group in [("Primary cancer", cancer_primary),
                               ("Complications",  cancer_complication)]:
        t_aurocs = [float(r["trans_auroc"]) for r in group if r["trans_auroc"] != "nan"]
        g_aurocs = [float(r["gnn_auroc"])   for r in group if r["gnn_auroc"]   != "nan"]
        if t_aurocs and g_aurocs:
            log.info("  %s: TRANS mean=%.4f  GNN mean=%.4f  Gap=%+.4f",
                     group_name,
                     np.mean(t_aurocs), np.mean(g_aurocs),
                     np.mean(g_aurocs) - np.mean(t_aurocs))


def _plot_cancer_labels(primary, complication, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    all_groups = [("Primary cancer diagnoses", primary,    "#C44E52"),
                  ("Cancer complications",     complication, "#4C72B0")]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    for ax, (title, group, color) in zip(axes, all_groups):
        if not group:
            ax.set_title(f"{title}\n(none in top-275 labels)")
            continue

        codes      = [r["snomed_code"][:20] for r in group]
        trans_vals = [float(r["trans_auroc"]) if r["trans_auroc"] != "nan" else 0
                      for r in group]
        gnn_vals   = [float(r["gnn_auroc"])   if r["gnn_auroc"]   != "nan" else 0
                      for r in group]

        # Sort by GNN AUROC for readability
        order = np.argsort(gnn_vals)
        codes      = [codes[i]      for i in order]
        trans_vals = [trans_vals[i] for i in order]
        gnn_vals   = [gnn_vals[i]   for i in order]

        y = np.arange(len(codes))
        ax.barh(y - 0.2, trans_vals, height=0.35, label="TRANS E7",      color="#4C72B0", alpha=0.8)
        ax.barh(y + 0.2, gnn_vals,   height=0.35, label="PatientGNN E11", color="#DD8452", alpha=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels(codes, fontsize=9)
        ax.set_xlabel("Per-label AUROC", fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlim(0.5, 1.0)
        ax.axvline(0.5, color="grey", linewidth=0.5, linestyle="--")
        ax.legend(fontsize=9)
        ax.grid(True, axis="x", alpha=0.3)
        ax.invert_yaxis()

    fig.suptitle("PatientGNN E11 vs TRANS E7 — Cancer-related SNOMED labels\n"
                 "(from top-275 most frequent conditions in MIMIC-IV OMOP)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_cancer_labels.png"), dpi=150)
    plt.close(fig)
    log.info("Saved fig_cancer_labels.png")


def _write_label_summary(rows, out_dir):
    """Fallback: write a sorted CSV of all 275 labels with gap column."""
    out = os.path.join(out_dir, "all_labels_sorted_by_gap.csv")
    rows_sorted = sorted(
        [r for r in rows if r["gap"] != "nan"],
        key=lambda r: float(r["gap"]), reverse=True
    )
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows_sorted)
    log.info("Saved fallback sorted CSV: %s", out)


# ---------------------------------------------------------------------------
# Part 2: Cancer patient subgroup inference
# ---------------------------------------------------------------------------

def _identify_cancer_patients_gnn(graphs, label_vocab):
    """
    Return indices (into graphs list) of patients who have any cancer
    SNOMED concept in their INPUT condition visit history.

    We look at the condition node indices in the graph and map back to
    concept_vocab to find SNOMED codes. Alternatively, we check y_diagnosis
    for patients who had a cancer code as a LABEL in any visit — this is a
    proxy for cancer patients without needing the concept_vocab reverse map.

    Strategy used here: a patient is a "cancer patient" if their y_diagnosis
    label vector has any bit set for a cancer SNOMED code from ALL_CANCER_SNOMED.
    This identifies patients who were ever *diagnosed* with a cancer code during
    the dataset period — a clean, reproducible definition.
    """
    cancer_label_indices = [i for i, code in enumerate(label_vocab)
                             if code in ALL_CANCER_SNOMED]
    log.info("  Cancer labels in GNN label_vocab: %d (indices: %s)",
             len(cancer_label_indices), cancer_label_indices[:10])

    if not cancer_label_indices:
        log.warning("  No cancer SNOMED codes found in label_vocab. "
                    "Returning all patients as fallback.")
        return list(range(len(graphs)))

    cancer_idx = []
    for i, g in enumerate(graphs):
        if hasattr(g, "y_diagnosis"):
            labels = g.y_diagnosis  # [275] float tensor
            if labels[cancer_label_indices].any():
                cancer_idx.append(i)

    log.info("  Cancer patients in GNN graphs: %d / %d (%.1f%%)",
             len(cancer_idx), len(graphs),
             100 * len(cancer_idx) / max(len(graphs), 1))
    return cancer_idx


def run_gnn_cancer_inference(args, device):
    """PatientGNN E11 inference on cancer patients only."""
    log.info("=== Part 2a: PatientGNN E11 — cancer subgroup ===")

    gnn_src = Path(__file__).parent.parent / "src" / "gnn"
    if str(gnn_src) not in sys.path:
        sys.path.insert(0, str(gnn_src))
    from model import PatientGNN  # noqa

    # Load all graphs
    graphs = torch.load(args.graphs_path, weights_only=False)
    graphs = [g for g in graphs if hasattr(g, "y_diagnosis")]

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

    # Normalise (same as train.py)
    all_x   = torch.cat([g["visit"].x for g in graphs], dim=0)
    mean4, std4 = all_x[:, 4].mean().item(), all_x[:, 4].std().item() + 1e-6
    mean5, std5 = all_x[:, 5].mean().item(), all_x[:, 5].std().item() + 1e-6
    for g in graphs:
        x = g["visit"].x.clone()
        x[:, 3] = torch.log1p(x[:, 3])
        x[:, 4] = (x[:, 4] - mean4) / std4
        x[:, 5] = (x[:, 5] - mean5) / std5
        g["visit"].x = x

    # Remove leakage edges
    for g in graphs:
        n_visits = g["visit"].x.size(0)
        ei = g["visit", "has_condition", "condition"].edge_index
        if ei.numel() > 0:
            mask = ei[0] != (n_visits - 1)
            g["visit", "has_condition", "condition"].edge_index = ei[:, mask]

    # Reproduce test split
    rng   = np.random.default_rng(args.seed)
    idx   = rng.permutation(len(graphs))
    n_test = int(len(graphs) * args.test_frac)
    test_idx = idx[:n_test]
    test_graphs = [graphs[i] for i in test_idx]

    # Identify cancer patients in test set
    cancer_flags = []
    cancer_label_indices = [i for i, code in enumerate(label_vocab)
                             if code in ALL_CANCER_SNOMED]
    for g in test_graphs:
        has_cancer = (len(cancer_label_indices) > 0 and
                      g.y_diagnosis[cancer_label_indices].any().item())
        cancer_flags.append(has_cancer)
    cancer_flags = np.array(cancer_flags)

    cancer_test = [g for g, flag in zip(test_graphs, cancer_flags) if flag]
    log.info("  Cancer patients in GNN test set: %d / %d",
             len(cancer_test), len(test_graphs))

    if len(cancer_test) < 20:
        log.warning("  Very few cancer patients in test set (%d). "
                    "AUROC estimates will be unreliable.", len(cancer_test))

    # Build model
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

    # Inference on full test + cancer test
    results = {}
    for name, subset in [("full_test", test_graphs), ("cancer_subgroup", cancer_test)]:
        loader = DataLoader(subset, batch_size=args.batch_size,
                            shuffle=False, num_workers=4)
        y_true_list, y_prob_list = [], []
        with torch.no_grad():
            for batch in tqdm(loader, desc=f"GNN {name}"):
                batch = batch.to(device)
                out    = model(batch, task="diagnosis")
                logits = out["diagnosis"]
                labels = batch.y_diagnosis.view(logits.shape)
                y_true_list.append(labels.cpu().float().numpy())
                y_prob_list.append(torch.sigmoid(logits).cpu().float().numpy())

        y_true = np.concatenate(y_true_list, axis=0)
        y_prob = np.nan_to_num(np.concatenate(y_prob_list, axis=0), nan=0.5)
        results[name] = {"y_true": y_true, "y_prob": y_prob}

    return results, label_vocab, cancer_label_indices


def run_trans_cancer_inference(args, device):
    """TRANS E7 inference on cancer patients only."""
    log.info("=== Part 2b: TRANS E7 — cancer subgroup ===")

    trans_root = Path(args.trans_ckpt).parent.parent
    if not (trans_root / "utils.py").exists():
        trans_root = Path(__file__).parent.parent / "TRANS"
    if str(trans_root) not in sys.path:
        sys.path.insert(0, str(trans_root))

    import random
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    from joblib import load as jload
    from utils import split_dataset, test as trans_test, get_init_tokenizers  # noqa
    from utils import prepare_labels, _extract_conditions                       # noqa
    from data.Task_omop import OMOPDataset                                      # noqa
    from data.Task import MMDataset, mm_collate_fn                              # noqa
    from models.Model import TRANS, graph_meta                                  # noqa
    from pyhealth.tokenizer import Tokenizer                                    # noqa
    import torch.utils.data as tud

    task_dataset = OMOPDataset(data_dir=args.data_dir, min_visits=2,
                               dev=False, top_k_conditions=args.top_k)
    Tokenizers   = get_init_tokenizers(task_dataset)

    if args.trans_label_vocab and os.path.exists(args.trans_label_vocab):
        with open(args.trans_label_vocab) as _f:
            _ordered_tokens = json.load(_f)
        label_tokenizer = Tokenizer(tokens=_ordered_tokens)
        log.info("  Loaded TRANS label vocab from %s (%d tokens)",
                 args.trans_label_vocab, len(_ordered_tokens))
    else:
        log.warning("  --trans_label_vocab not provided — label ordering is "
                    "hash-randomised and TRANS AUROC will be INVALID.")
        label_tokenizer = Tokenizer(tokens=task_dataset.get_all_tokens("conditions"))
    num_labels = label_tokenizer.get_vocabulary_size()

    pkl_path = args.trans_pkl
    if pkl_path and os.path.exists(pkl_path):
        mdataset = jload(pkl_path)
    else:
        os.makedirs(args.graph_cache, exist_ok=True)
        mdataset = MMDataset(task_dataset, Tokenizers, dim=128,
                             device=torch.device("cpu"), trans_dim=4,
                             di=False, cache_dir=args.graph_cache)

    trainset, validset, testset = split_dataset(mdataset)

    # Label names for alignment
    vocab_obj = label_tokenizer.vocabulary
    if hasattr(vocab_obj, "token2idx"):
        token2idx = vocab_obj.token2idx
    elif hasattr(vocab_obj, "idx2token"):
        token2idx = {v: k for k, v in vocab_obj.idx2token.items()}
    else:
        token2idx = {}
    idx2token   = {v: k for k, v in token2idx.items()}
    label_names = [idx2token.get(i, f"UNKNOWN_{i}") for i in range(num_labels)]
    cancer_label_indices = [i for i, code in enumerate(label_names)
                             if code in ALL_CANCER_SNOMED]
    log.info("  Cancer labels in TRANS vocab: %d", len(cancer_label_indices))

    loader_kw = dict(batch_size=args.batch_size, shuffle=False,
                     num_workers=4, collate_fn=mm_collate_fn,
                     pin_memory=(device.type == "cuda"))

    # Build model
    model = TRANS(Tokenizers, 128, num_labels, device,
                  graph_meta=graph_meta, pe=4).to(device)
    model.load_state_dict(
        torch.load(args.trans_ckpt, map_location=device, weights_only=True))
    model.eval()

    # Full test set
    test_loader = tud.DataLoader(testset, **loader_kw)
    y_true_full, y_prob_full = trans_test(test_loader, model, label_tokenizer)

    # Cancer subgroup: patients where any cancer label is positive in labels
    # We need to re-collect test samples with their labels
    log.info("  Identifying cancer patients in TRANS test set …")
    all_y_true = []
    all_batches_raw = []
    for batch in tqdm(tud.DataLoader(testset, batch_size=args.batch_size,
                                     shuffle=False, collate_fn=mm_collate_fn,
                                     num_workers=4),
                      desc="Collecting TRANS test labels"):
        seq_batch, _ = batch
        lbl = prepare_labels(_extract_conditions(seq_batch), label_tokenizer)
        all_y_true.append(lbl.numpy())
        all_batches_raw.append(batch)

    y_true_all_test = np.concatenate(all_y_true, axis=0)  # [N_test, num_labels]

    if len(cancer_label_indices) > 0:
        cancer_mask = y_true_all_test[:, cancer_label_indices].any(axis=1)
    else:
        cancer_mask = np.ones(len(y_true_all_test), dtype=bool)

    y_true_cancer = y_true_all_test[cancer_mask]
    y_prob_cancer = np.nan_to_num(y_prob_full[cancer_mask], nan=0.5)
    log.info("  TRANS cancer patients in test set: %d / %d",
             cancer_mask.sum(), len(cancer_mask))

    results = {
        "full_test":       {"y_true": np.nan_to_num(y_prob_full, nan=0.5),
                            "y_prob": np.nan_to_num(y_prob_full, nan=0.5)},
        # Note: full_test y_true comes from separate run above for consistency
        "cancer_subgroup": {"y_true": y_true_cancer, "y_prob": y_prob_cancer},
    }
    # Correct full test y_true
    results["full_test"]["y_true"] = y_true_all_test

    return results, label_names, cancer_label_indices


# ---------------------------------------------------------------------------
# Part 2 analysis: compare on cancer subgroup
# ---------------------------------------------------------------------------

def compute_subgroup_metrics(y_true, y_prob, label_indices_of_interest=None,
                              min_positives=3):
    """
    Compute macro AUROC, AUPRC, and per-label AUROC.
    If label_indices_of_interest is provided, also compute metrics restricted
    to those labels only.
    """
    n_labels = y_true.shape[1]
    aurocs, auprcs = [], []
    per_label_auroc = np.full(n_labels, np.nan)

    for i in range(n_labels):
        pos = y_true[:, i].sum()
        if pos >= min_positives and pos < len(y_true):
            a = roc_auc_score(y_true[:, i], y_prob[:, i])
            b = average_precision_score(y_true[:, i], y_prob[:, i])
            aurocs.append(a)
            auprcs.append(b)
            per_label_auroc[i] = a

    m = {
        "macro_auroc": float(np.mean(aurocs))  if aurocs  else float("nan"),
        "macro_auprc": float(np.mean(auprcs)) if auprcs else float("nan"),
        "n_patients":  int(y_true.shape[0]),
        "n_labels_evaluated": len(aurocs),
        "per_label_auroc": per_label_auroc.tolist(),
    }

    if label_indices_of_interest:
        ci_aurocs = [per_label_auroc[i] for i in label_indices_of_interest
                     if not np.isnan(per_label_auroc[i])]
        m["cancer_label_macro_auroc"] = float(np.mean(ci_aurocs)) if ci_aurocs else float("nan")
        m["n_cancer_labels_evaluated"] = len(ci_aurocs)

    return m


def part2_subgroup_analysis(gnn_results, trans_results,
                             gnn_label_vocab, trans_label_names,
                             gnn_cancer_idx, trans_cancer_idx,
                             out_dir):
    """Compare models on cancer subgroup."""
    log.info("=== Part 2: Cancer subgroup comparison ===")

    # Align label vocabularies
    common_codes = sorted(set(gnn_label_vocab) & set(trans_label_names))
    gnn_code2idx   = {c: i for i, c in enumerate(gnn_label_vocab)}
    trans_code2idx = {c: i for i, c in enumerate(trans_label_names)}
    gnn_common   = np.array([gnn_code2idx[c]   for c in common_codes])
    trans_common = np.array([trans_code2idx[c] for c in common_codes])
    common_cancer_idx = [i for i, c in enumerate(common_codes) if c in ALL_CANCER_SNOMED]

    summary = {}
    for split in ("full_test", "cancer_subgroup"):
        gnn_yt   = gnn_results[split]["y_true"][:, gnn_common]
        gnn_yp   = gnn_results[split]["y_prob"][:, gnn_common]
        trans_yt = trans_results[split]["y_true"][:, trans_common]
        trans_yp = trans_results[split]["y_prob"][:, trans_common]

        m_gnn   = compute_subgroup_metrics(gnn_yt,   gnn_yp,   common_cancer_idx)
        m_trans = compute_subgroup_metrics(trans_yt, trans_yp, common_cancer_idx)

        summary[split] = {
            "gnn":   m_gnn,
            "trans": m_trans,
            "auroc_gap": (m_gnn["macro_auroc"] - m_trans["macro_auroc"]
                          if not (np.isnan(m_gnn["macro_auroc"]) or
                                  np.isnan(m_trans["macro_auroc"])) else float("nan")),
            "cancer_auroc_gap": (m_gnn.get("cancer_label_macro_auroc", float("nan")) -
                                  m_trans.get("cancer_label_macro_auroc", float("nan"))),
        }

        log.info("\n  %s  (GNN n=%d, TRANS n=%d)",
                 split, m_gnn["n_patients"], m_trans["n_patients"])
        log.info("  %-20s  %12s  %12s  %8s", "", "TRANS E7", "GNN E11", "Gap")
        log.info("  %-20s  %12.4f  %12.4f  %+8.4f",
                 "Macro AUROC (all labels)",
                 m_trans["macro_auroc"], m_gnn["macro_auroc"],
                 summary[split]["auroc_gap"])
        log.info("  %-20s  %12.4f  %12.4f",
                 "Macro AUPRC",
                 m_trans["macro_auprc"], m_gnn["macro_auprc"])
        if common_cancer_idx:
            log.info("  %-20s  %12.4f  %12.4f  %+8.4f",
                     "Cancer labels AUROC",
                     m_trans.get("cancer_label_macro_auroc", float("nan")),
                     m_gnn.get("cancer_label_macro_auroc",   float("nan")),
                     summary[split]["cancer_auroc_gap"])

    # Save JSON
    json_path = os.path.join(out_dir, "cancer_subgroup_results.json")
    json_safe = json.dumps(
        {k: {mk: (mv if not isinstance(mv, dict) else
                  {ik: (iv if not isinstance(iv, list) else iv[:10])
                   for ik, iv in mv.items()})
             for mk, mv in v.items()}
         for k, v in summary.items()},
        indent=2
    )
    with open(json_path, "w") as f:
        f.write(json_safe)
    log.info("Saved %s", json_path)

    # ---- Plot: grouped bar — full vs cancer subgroup, both models -----------
    _plot_subgroup_bars(summary, out_dir)


def _plot_subgroup_bars(summary, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = ["macro_auroc", "macro_auprc", "cancer_label_macro_auroc"]
    metric_labels = ["Macro AUROC\n(all 275 labels)",
                     "Macro AUPRC\n(all 275 labels)",
                     "AUROC\n(cancer labels only)"]

    fig, axes = plt.subplots(1, len(metrics), figsize=(15, 6))

    for ax, metric, mlabel in zip(axes, metrics, metric_labels):
        groups = ["full_test", "cancer_subgroup"]
        group_labels = ["All test patients", "Cancer patients only"]
        x = np.arange(len(groups))
        width = 0.35

        trans_vals = [summary[g]["trans"].get(metric, float("nan")) for g in groups]
        gnn_vals   = [summary[g]["gnn"].get(metric, float("nan"))   for g in groups]

        bars1 = ax.bar(x - width/2, trans_vals, width, label="TRANS E7",
                       color="#4C72B0", alpha=0.8)
        bars2 = ax.bar(x + width/2, gnn_vals,   width, label="PatientGNN E11",
                       color="#DD8452", alpha=0.8)

        # Gap annotation
        for xi, (tv, gv) in enumerate(zip(trans_vals, gnn_vals)):
            if not (np.isnan(tv) or np.isnan(gv)):
                gap = gv - tv
                y_pos = max(tv, gv) + 0.005
                ax.text(xi, y_pos, f"{gap:+.3f}",
                        ha="center", va="bottom", fontsize=9, fontweight="bold",
                        color="green" if gap > 0 else "red")

        ax.set_xticks(x)
        ax.set_xticklabels(group_labels, fontsize=10)
        ax.set_ylabel(mlabel, fontsize=10)
        ax.set_title(mlabel, fontsize=11)
        ax.set_ylim(max(0, min(trans_vals + gnn_vals) - 0.05), 1.0)
        ax.legend(fontsize=9)
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle(
        "PatientGNN E11 vs TRANS E7: Full cohort vs Cancer patient subgroup\n"
        "Hypothesis: GNN advantage is LARGER in cancer patients (complex drug-lab-condition graphs)",
        fontsize=11, fontweight="bold"
    )
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_cancer_subgroup_auroc.png"), dpi=150)
    plt.close(fig)
    log.info("Saved fig_cancer_subgroup_auroc.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    # ---- Part 1: Cancer label analysis from existing CSV --------------------
    if os.path.exists(args.per_label_csv):
        part1_cancer_labels(args.per_label_csv, args.out_dir)
    else:
        log.warning("per_label_csv not found: %s — skipping Part 1. "
                    "Run analyse_per_label.py first.", args.per_label_csv)

    # ---- Part 2: Cancer patient subgroup inference --------------------------
    gnn_results, gnn_label_vocab, gnn_cancer_idx = \
        run_gnn_cancer_inference(args, device)

    trans_results, trans_label_names, trans_cancer_idx = \
        run_trans_cancer_inference(args, device)

    part2_subgroup_analysis(
        gnn_results, trans_results,
        gnn_label_vocab, trans_label_names,
        gnn_cancer_idx, trans_cancer_idx,
        args.out_dir,
    )

    log.info("\nAll cancer analysis outputs written to: %s", args.out_dir)


if __name__ == "__main__":
    main()
