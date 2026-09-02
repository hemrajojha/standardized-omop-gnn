"""
analyse_early_prediction_separate_models.py
===========================================
Temporal early-warning analysis: PatientGNN E11 vs TRANS E7, side-by-side.

Question: How many visits in advance can each model predict eventual diagnoses?

Method
------
For each lead time k = 0, 1, 2, ..., max_lead (default 3):
  1. Build a truncated copy of each test patient's record, retaining only the
     first N-k visits (where N is the patient's total number of visits).
  2. Run each model on the truncated input.
  3. Evaluate against the ORIGINAL last-visit label — conditions confirmed at
     visit N-1 — which is held fixed across all lead times.

This answers:
  "Given medical records from k+1 visits ago, how well can each model predict
   the diagnoses that are eventually confirmed at the most recent visit?"

Outputs (written to --out_dir)
-------------------------------
  early_prediction_comparison.csv    per-label AUROC at each lead for both models
  fig_lead_vs_auroc_comparison.png   primary: macro-AUROC vs lead, both models
  fig_auroc_drop_comparison.png      AUROC degradation grouped bars
  fig_cohort_shrinkage.png           N patients retained at each lead
  fig_per_label_lead_scatter.png     scatter GNN vs TRANS per-label AUROC, k=1,2,3

HPC run command example
-----------------------
python analysis/analyse_early_prediction_separate_models.py \\
  --gnn_ckpt    /path/to/OMOP_GNN_PROJECT/logs/e11/checkpoints/best_model.pt \\
  --graphs_path /path/to/OMOP_GNN_PROJECT/data/processed/patient_graphs.pt \\
  --trans_ckpt  /path/to/TRANS/logs/best_TRANS_omop.ckpt \\
  --data_dir    /path/to/OMOP_GNN_PROJECT/data/export \\
  --trans_pkl   /path/to/TRANS/logs/omop_4.pkl \\
  --out_dir     analysis/early_prediction_separate \\
  --device      cuda:1 \\
  --batch_size  64 \\
  --max_lead    3
"""

import argparse
import csv
import json
import logging
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Colour palette (shared with analyse_label_categories_separate_models.py)
# ---------------------------------------------------------------------------
GNN_BLUE  = "#2196F3"
TRANS_RED = "#FF5722"
GAP_GREEN = "#4CAF50"
Q_COLORS  = {1: "#9C27B0", 2: "#2196F3", 3: "#4CAF50", 4: "#FF9800"}


# ===========================================================================
# Arguments
# ===========================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Early-prediction comparison: PatientGNN E11 vs TRANS E7"
    )
    # Required
    p.add_argument("--gnn_ckpt",    required=True,
                   help="Path to E11 PatientGNN best_model.pt checkpoint")
    p.add_argument("--graphs_path", required=True,
                   help="Path to patient_graphs.pt")
    p.add_argument("--trans_ckpt",  required=True,
                   help="Path to TRANS E7 best_TRANS_omop.ckpt")
    p.add_argument("--data_dir",    required=True,
                   help="OMOP CSV export directory (for OMOPDataset)")

    # Optional TRANS paths
    p.add_argument("--trans_pkl",  default=None,
                   help="Cached MMDataset pkl (omop_4.pkl). Load instead of rebuild if exists.")

    # Shared options
    p.add_argument("--out_dir",       default="analysis/early_prediction_separate")
    p.add_argument("--device",        default="cuda:1")
    p.add_argument("--batch_size",    type=int,   default=64)
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--val_frac",      type=float, default=0.1)
    p.add_argument("--test_frac",     type=float, default=0.15)
    p.add_argument("--max_lead",      type=int,   default=3)
    p.add_argument("--min_positives", type=int,   default=10)

    # PatientGNN hyperparams (must match E11 training)
    p.add_argument("--hidden_dim",             type=int,   default=128)
    p.add_argument("--kg_embed_dim",           type=int,   default=128)
    p.add_argument("--num_gnn_layers",         type=int,   default=2)
    p.add_argument("--num_transformer_layers", type=int,   default=2)
    p.add_argument("--nhead",                  type=int,   default=4)
    p.add_argument("--pe_dim",                 type=int,   default=4)
    p.add_argument("--alpha",                  type=float, default=0.8)
    p.add_argument("--dropout",                type=float, default=0.3)
    return p.parse_args()


# ===========================================================================
# SECTION A — PatientGNN
# ===========================================================================

def load_and_preprocess_gnn(args):
    """
    Load patient_graphs.pt, apply feature normalisation and leakage prevention
    identical to train.py, then return the test split.

    Returns
    -------
    test_graphs : list[HeteroData]
    label_vocab : list[str]          275 SNOMED concept ID strings
    vocab_sizes : dict               {'cond', 'proc', 'drug'} → int
    """
    from torch_geometric.data import HeteroData  # noqa — import once here

    log.info("=== PatientGNN — loading graphs ===")
    log.info("  Loading %s …", args.graphs_path)
    graphs = torch.load(args.graphs_path, weights_only=False)
    graphs = [g for g in graphs if hasattr(g, "y_diagnosis")]
    log.info("  %d graphs with y_diagnosis", len(graphs))

    graphs_dir = Path(args.graphs_path).parent

    with open(graphs_dir / "label_vocab.json") as f:
        label_vocab = json.load(f)
    log.info("  %d diagnosis labels", len(label_vocab))

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

    # Normalise visit features (same as train.py)
    all_x    = torch.cat([g["visit"].x for g in graphs], dim=0)
    mean4    = float(all_x[:, 4].mean())
    std4     = float(all_x[:, 4].std()) + 1e-6
    mean5    = float(all_x[:, 5].mean())
    std5     = float(all_x[:, 5].std()) + 1e-6
    for g in graphs:
        x = g["visit"].x.clone()
        x[:, 3] = torch.log1p(x[:, 3])
        x[:, 4] = (x[:, 4] - mean4) / std4
        x[:, 5] = (x[:, 5] - mean5) / std5
        g["visit"].x = x

    # Leakage prevention: remove last-visit condition edges
    for g in graphs:
        last_v = g["visit"].x.size(0) - 1
        ei = g["visit", "has_condition", "condition"].edge_index
        if ei.numel() > 0:
            g["visit", "has_condition", "condition"].edge_index = ei[:, ei[0] != last_v]

    # Deterministic test split (numpy seed=42, same as E11 training)
    rng    = np.random.default_rng(args.seed)
    idx    = rng.permutation(len(graphs))
    n_test = int(len(graphs) * args.test_frac)
    test_idx    = idx[:n_test]
    test_graphs = [graphs[i] for i in test_idx]
    log.info("  GNN test split: %d patients", len(test_graphs))

    return test_graphs, label_vocab, vocab_sizes


def make_truncated_graph(g, n_keep):
    """
    Build a truncated copy of patient graph g retaining only the first
    n_keep visit nodes.

    The new last visit (index n_keep-1) has its condition edges removed to
    prevent label leakage. Procedure and drug edges are retained for all
    n_keep visits.  y_diagnosis (ORIGINAL last-visit label) is unchanged.

    Parameters
    ----------
    g      : HeteroData  already preprocessed
    n_keep : int         number of visit nodes to retain (>= 2)
    """
    from torch_geometric.data import HeteroData

    assert n_keep >= 2, f"n_keep must be >= 2, got {n_keep}"

    new_g  = HeteroData()
    last_v = n_keep - 1

    # Visit features — first n_keep rows
    new_g["visit"].x = g["visit"].x[:n_keep]

    # Temporal edges
    if n_keep > 1:
        src = torch.arange(n_keep - 1, dtype=torch.long)
        new_g["visit", "temporal_next", "visit"].edge_index = torch.stack(
            [src, src + 1]
        )
    else:
        new_g["visit", "temporal_next", "visit"].edge_index = torch.zeros(
            2, 0, dtype=torch.long
        )

    # Concept edges with leakage prevention
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

        mask = orig_ei[0] < last_v if edge_rel == "has_condition" else orig_ei[0] < n_keep
        new_g[domain].x = orig_cx
        new_g["visit", edge_rel, domain].edge_index = orig_ei[:, mask]

    new_g.y_diagnosis = g.y_diagnosis
    if hasattr(g, "subject_id"):
        new_g.subject_id = g.subject_id
    return new_g


def build_gnn_model(args, vocab_sizes, num_labels, device):
    """Load PatientGNN from E11 checkpoint."""
    # Locate src/gnn relative to THIS script, then try project root
    this_dir = Path(__file__).parent
    for candidate in [
        this_dir.parent / "src" / "gnn",
        this_dir.parent.parent / "src" / "gnn",
        this_dir / "src" / "gnn",
    ]:
        if candidate.exists():
            gnn_src = candidate
            break
    else:
        gnn_src = this_dir.parent / "src" / "gnn"
        log.warning("src/gnn not found at expected location, using %s", gnn_src)

    if str(gnn_src) not in sys.path:
        sys.path.insert(0, str(gnn_src))

    from model import PatientGNN  # noqa

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
    log.info("  Loaded GNN checkpoint — epoch %d, val AUROC %.4f",
             ckpt.get("epoch", -1),
             ckpt.get("val_metrics", {}).get("auroc_macro", float("nan")))
    return model


@torch.no_grad()
def run_gnn_inference(graphs_list, model, device, batch_size, desc="GNN inference"):
    """
    Run PatientGNN inference on a list of HeteroData graphs.

    Returns
    -------
    y_true : np.ndarray  [N, num_labels]
    y_prob : np.ndarray  [N, num_labels]
    """
    from torch_geometric.data import DataLoader as GeoLoader

    loader = GeoLoader(graphs_list, batch_size=batch_size, shuffle=False, num_workers=0)
    y_true_all, y_prob_all = [], []

    for batch in tqdm(loader, desc=desc, leave=False):
        batch  = batch.to(device)
        out    = model(batch, task="diagnosis")
        logits = out["diagnosis"]
        labels = batch.y_diagnosis.view(logits.shape)
        y_true_all.append(labels.cpu().float().numpy())
        y_prob_all.append(torch.sigmoid(logits).cpu().float().numpy())

    y_true = np.concatenate(y_true_all, axis=0)
    y_prob = np.concatenate(y_prob_all, axis=0)
    y_prob = np.nan_to_num(y_prob, nan=0.5)
    return y_true, y_prob


def run_gnn_lead_analysis(test_graphs, model, device, args):
    """
    Evaluate PatientGNN at each lead time k = 0 .. max_lead.

    Returns
    -------
    dict  {k: {'auroc': np.ndarray [num_labels], 'n_valid': int,
               'y_true': np.ndarray, 'y_prob': np.ndarray}}
    """
    results = {}
    for k in range(args.max_lead + 1):
        log.info("── GNN Lead k=%d ──", k)
        if k == 0:
            valid_graphs = test_graphs
        else:
            valid_graphs = []
            skip = 0
            for g in test_graphs:
                n_visits = g["visit"].x.size(0)
                n_keep   = n_visits - k
                if n_keep < 2:
                    skip += 1
                    continue
                valid_graphs.append(make_truncated_graph(g, n_keep))
            log.info("  %d / %d valid (skipped %d with N <= %d)",
                     len(valid_graphs), len(test_graphs), skip, k + 1)

        if not valid_graphs:
            log.warning("  No valid graphs at GNN lead k=%d — skipping", k)
            continue

        y_true, y_prob = run_gnn_inference(
            valid_graphs, model, device, args.batch_size, desc=f"GNN k={k}"
        )
        auroc = per_label_auroc(y_true, y_prob, args.min_positives)
        log.info("  GNN Macro-AUROC @ k=%d: %.4f  (%d labels evaluated)",
                 k, np.nanmean(auroc), int(np.sum(~np.isnan(auroc))))
        results[k] = {"auroc": auroc, "n_valid": len(valid_graphs),
                      "y_true": y_true, "y_prob": y_prob}
    return results


# ===========================================================================
# SECTION B — TRANS E7
# ===========================================================================

def load_and_preprocess_trans(args):
    """
    Load TRANS E7 checkpoint and prepare the test split.

    Imports TRANS modules from the checkpoint parent-parent directory.

    Returns
    -------
    test_samples    : list[dict]   raw sample dicts from OMOPDataset
    Tokenizers      : object       from get_init_tokenizers
    label_tokenizer : Tokenizer
    label_vocab_trans: list[str]   ordered label list
    """
    log.info("=== TRANS — loading dataset ===")

    # ---- Seed ----------------------------------------------------------------
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # ---- Locate TRANS root ---------------------------------------------------
    trans_root = Path(args.trans_ckpt).parent.parent
    if not (trans_root / "utils.py").exists():
        # Fallback: look for local TRANS/ subdir
        candidates = [
            Path(__file__).parent.parent / "TRANS",
            Path(__file__).parent / "TRANS",
        ]
        for c in candidates:
            if c.exists() and (c / "utils.py").exists():
                trans_root = c
                break
    log.info("  TRANS root: %s", trans_root)
    if str(trans_root) not in sys.path:
        sys.path.insert(0, str(trans_root))

    from utils import get_init_tokenizers          # noqa
    from data.Task_omop import OMOPDataset         # noqa
    from data.Task import MMDataset, mm_collate_fn # noqa
    from pyhealth.tokenizer import Tokenizer       # noqa

    # ---- Build OMOPDataset --------------------------------------------------
    log.info("  Building OMOPDataset from %s …", args.data_dir)
    task_dataset = OMOPDataset(
        data_dir=args.data_dir,
        min_visits=2,
        dev=False,
        top_k_conditions=275,
    )
    Tokenizers = get_init_tokenizers(task_dataset)

    # ---- Label tokenizer (ordered vocab) ------------------------------------
    # Look for label_vocab_trans.json next to the checkpoint
    label_vocab_path = Path(args.trans_ckpt).parent / "label_vocab_trans.json"
    if label_vocab_path.exists():
        with open(label_vocab_path) as f:
            ordered_tokens = json.load(f)
        label_tokenizer = Tokenizer(tokens=ordered_tokens)
        log.info("  Loaded ordered label vocab from %s (%d tokens)",
                 label_vocab_path, len(ordered_tokens))
    else:
        log.warning(
            "  label_vocab_trans.json not found at %s — "
            "falling back to fresh tokenizer (TRANS AUROC may be invalid).",
            label_vocab_path,
        )
        ordered_tokens  = task_dataset.get_all_tokens("conditions")
        label_tokenizer = Tokenizer(tokens=ordered_tokens)

    # Extract ordered label list for alignment with GNN vocab later
    vocab_obj = label_tokenizer.vocabulary
    if hasattr(vocab_obj, "token2idx"):
        idx2tok = {v: k for k, v in vocab_obj.token2idx.items()}
    elif hasattr(vocab_obj, "idx2token"):
        idx2tok = dict(vocab_obj.idx2token)
    else:
        idx2tok = {}
    num_labels_trans = label_tokenizer.get_vocabulary_size()
    label_vocab_trans = [idx2tok.get(i, f"UNKNOWN_{i}") for i in range(num_labels_trans)]
    log.info("  TRANS label vocab size: %d", num_labels_trans)

    # ---- Build / load MMDataset (patient graphs) ----------------------------
    pkl_path = args.trans_pkl
    if pkl_path and os.path.exists(pkl_path):
        log.info("  Loading cached MMDataset from %s …", pkl_path)
        from joblib import load as jload
        mdataset = jload(pkl_path)
    else:
        log.info("  Building MMDataset from scratch (this is slow) …")
        graph_cache = str(trans_root / "cache" / "patient_graph_npz_omop")
        os.makedirs(graph_cache, exist_ok=True)
        mdataset = MMDataset(
            task_dataset, Tokenizers,
            dim=128,
            device=torch.device("cpu"),
            trans_dim=4,
            di=False,
            cache_dir=graph_cache,
        )
        if pkl_path:
            from joblib import dump as jdump
            os.makedirs(os.path.dirname(os.path.abspath(pkl_path)), exist_ok=True)
            jdump(mdataset, pkl_path)
            log.info("  Saved MMDataset to %s", pkl_path)

    # Patch graph cache path if it was baked as a relative path during training
    trans_cache = trans_root / "cache" / "patient_graph_npz_omop"
    if trans_cache.exists():
        def _patch_cache(ds):
            if hasattr(ds, "graph_ds") and hasattr(ds.graph_ds, "cache_dir"):
                ds.graph_ds.cache_dir = str(trans_cache)
            underlying = getattr(ds, "dataset", None)
            if underlying is not None:
                _patch_cache(underlying)
        _patch_cache(mdataset)
        log.info("  Patched graph cache → %s", trans_cache)

    # ---- Deterministic test split -------------------------------------------
    from torch.utils.data import random_split as _random_split
    total   = len(mdataset)
    n_train = int(total * 0.75)
    n_val   = int(total * 0.10)
    n_test  = total - n_train - n_val
    gen     = torch.Generator().manual_seed(args.seed)
    trainset, validset, testset = _random_split(
        mdataset, [n_train, n_val, n_test], generator=gen
    )
    log.info("  TRANS split — train=%d  val=%d  test=%d",
             len(trainset), len(validset), len(testset))

    # Patch cache on Subset datasets too
    for subset in (trainset, validset, testset):
        ds = getattr(subset, "dataset", None)
        if ds is not None and hasattr(ds, "graph_ds") and hasattr(ds.graph_ds, "cache_dir"):
            ds.graph_ds.cache_dir = str(trans_cache)

    # Extract raw sample dicts from test indices
    # Each element of OMOPDataset is a dict with keys like:
    #   patient_id, visit_id, conditions, procedures, drugs, cond_hist,
    #   procedures (per-visit), drugs (per-visit), adm_time, ...
    # We need the raw OMOPDataset samples (not MMDataset items which are graphs)
    # to truncate and rebuild graphs on the fly.
    raw_samples = task_dataset.samples  # list of patient dicts
    # Map MMDataset indices back to raw_samples indices
    # MMDataset wraps task_dataset: item i corresponds to raw_samples[i]
    test_indices = testset.indices  # from random_split Subset
    test_samples = [raw_samples[i] for i in test_indices]
    log.info("  TRANS test samples: %d", len(test_samples))

    return test_samples, Tokenizers, label_tokenizer, label_vocab_trans, mm_collate_fn


def truncate_trans_sample(sample, n_keep):
    """
    Truncate a TRANS raw sample dict to the first n_keep visits.

    Lists indexed by visit (cond_hist, procedures, drugs, adm_time) are sliced.
    The target label (conditions = last-visit diagnoses) is kept unchanged.

    Parameters
    ----------
    sample : dict   raw sample from OMOPDataset
    n_keep : int    number of visits to retain (>= 2)

    Returns
    -------
    dict   truncated sample (shallow copy with sliced lists)
    """
    truncated = dict(sample)
    for key in ("cond_hist", "procedures", "drugs", "adm_time"):
        if key in truncated and isinstance(truncated[key], list):
            truncated[key] = truncated[key][:n_keep]
    # 'conditions' (prediction target) is intentionally kept unchanged
    return truncated


def build_trans_model(args, Tokenizers, num_labels, device):
    """Load TRANS E7 from checkpoint."""
    from models.Model import TRANS, graph_meta  # noqa

    model = TRANS(
        Tokenizers, 128, num_labels,
        device, graph_meta=graph_meta, pe=4,
    ).to(device)
    model.load_state_dict(
        torch.load(args.trans_ckpt, map_location=device, weights_only=True)
    )
    model.eval()
    log.info("  Loaded TRANS checkpoint from %s", args.trans_ckpt)
    return model


@torch.no_grad()
def run_trans_inference_batch(
    samples, Tokenizers, label_tokenizer, model, device, batch_size, trans_root, desc="TRANS inf"
):
    """
    Run TRANS inference on a list of raw sample dicts.

    For each sample we build a per-patient PatientGraph on-the-fly, then
    batch and run the model.

    Returns
    -------
    y_true : np.ndarray  [N, num_labels]
    y_prob : np.ndarray  [N, num_labels]
    """
    from data.Task import mm_collate_fn, MMDataset  # noqa
    from data.GraphConstruction import PatientGraph  # noqa
    import torch.utils.data as tud

    graph_cache = str(trans_root / "cache" / "patient_graph_npz_omop")
    num_labels  = label_tokenizer.get_vocabulary_size()

    # Build a lightweight MMDataset-compatible wrapper that builds graphs on the fly
    # using the raw (possibly truncated) samples.
    # We build individual PatientGraph objects and fake a dataset wrapper.
    #
    # Because PatientGraph expects the full MMDataset interface, we instead build
    # the graph data by constructing a temporary MMDataset from the sample list.
    # This is the cleanest path that mirrors what training does.

    # Construct a minimal task-dataset-like object
    class _TmpTaskDataset:
        """Minimal OMOPDataset-compatible wrapper around a list of samples."""
        def __init__(self, sample_list):
            self.samples = sample_list

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            return self.samples[idx]

        def get_all_tokens(self, key):
            token_set = set()
            for s in self.samples:
                lst = s.get(key, [])
                if isinstance(lst, list):
                    for item in lst:
                        if isinstance(item, list):
                            token_set.update(item)
                        else:
                            token_set.add(item)
            return list(token_set)

    tmp_task = _TmpTaskDataset(samples)

    tmp_mmdataset = MMDataset(
        tmp_task, Tokenizers,
        dim=128,
        device=torch.device("cpu"),
        trans_dim=4,
        di=False,
        cache_dir=graph_cache,
    )

    loader = tud.DataLoader(
        tmp_mmdataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=mm_collate_fn,
        pin_memory=False,
    )

    y_true_all, y_prob_all = [], []
    for batch_data in tqdm(loader, desc=desc, leave=False):
        # Move tensors to device
        inputs = {}
        for k, v in batch_data.items():
            if k == "graph":
                inputs[k] = v.to(device) if hasattr(v, "to") else v
            elif isinstance(v, torch.Tensor):
                inputs[k] = v.to(device)
            else:
                inputs[k] = v

        logits = model(inputs)                             # [B, num_labels]
        probs  = torch.sigmoid(logits).cpu().float().numpy()

        # Build ground-truth labels from raw sample conditions
        # We need the batch's original samples — use the DataLoader index range
        # The mm_collate_fn returns 'conditions' (list of lists of tokens)
        raw_conds = batch_data.get("conditions", None)    # list[list[str]] or None
        if raw_conds is not None:
            labels_batch = label_tokenizer.batch_encode_2d(
                raw_conds, padding=False, truncation=False
            )
            # Convert to binary multi-hot [B, num_labels]
            y_np = np.zeros((len(raw_conds), num_labels), dtype=np.float32)
            for row_i, indices in enumerate(labels_batch):
                for idx in indices:
                    if 0 <= idx < num_labels:
                        y_np[row_i, idx] = 1.0
        else:
            # Fallback: pull labels from 'label' key if present
            raw_lbl = batch_data.get("label", None)
            if raw_lbl is not None and isinstance(raw_lbl, torch.Tensor):
                y_np = raw_lbl.cpu().float().numpy()
            else:
                y_np = np.zeros_like(probs)

        y_true_all.append(y_np)
        y_prob_all.append(probs)

    y_true = np.concatenate(y_true_all, axis=0)
    y_prob = np.concatenate(y_prob_all, axis=0)
    y_prob = np.nan_to_num(y_prob, nan=0.5)
    return y_true, y_prob


def run_trans_lead_analysis(test_samples, Tokenizers, label_tokenizer,
                            model, device, args, trans_root):
    """
    Evaluate TRANS at each lead time k = 0 .. max_lead.

    For each lead k:
      - For each patient, compute n_visits = len(sample['cond_hist']),
        n_keep = n_visits - k.  Skip if n_keep < 2.
      - Build truncated sample via truncate_trans_sample.
      - Run inference on the truncated batch.

    Returns
    -------
    dict  {k: {'auroc': np.ndarray [num_labels], 'n_valid': int,
               'y_true': np.ndarray, 'y_prob': np.ndarray}}
    """
    results = {}
    for k in range(args.max_lead + 1):
        log.info("── TRANS Lead k=%d ──", k)
        if k == 0:
            valid_samples = test_samples
        else:
            valid_samples = []
            skip = 0
            for s in test_samples:
                n_visits = len(s.get("cond_hist", []))
                n_keep   = n_visits - k
                if n_keep < 2:
                    skip += 1
                    continue
                valid_samples.append(truncate_trans_sample(s, n_keep))
            log.info("  %d / %d valid (skipped %d with N <= %d)",
                     len(valid_samples), len(test_samples), skip, k + 1)

        if not valid_samples:
            log.warning("  No valid TRANS samples at lead k=%d — skipping", k)
            continue

        y_true, y_prob = run_trans_inference_batch(
            valid_samples, Tokenizers, label_tokenizer,
            model, device, args.batch_size, trans_root,
            desc=f"TRANS k={k}",
        )
        auroc = per_label_auroc(y_true, y_prob, args.min_positives)
        log.info("  TRANS Macro-AUROC @ k=%d: %.4f  (%d labels evaluated)",
                 k, np.nanmean(auroc), int(np.sum(~np.isnan(auroc))))
        results[k] = {"auroc": auroc, "n_valid": len(valid_samples),
                      "y_true": y_true, "y_prob": y_prob}
    return results


# ===========================================================================
# Shared helpers
# ===========================================================================

def per_label_auroc(y_true, y_prob, min_positives=10):
    """Per-label AUROC array. Returns np.nan where < min_positives positives."""
    n_labels = y_true.shape[1]
    aurocs   = np.full(n_labels, np.nan)
    for i in range(n_labels):
        if y_true[:, i].sum() >= min_positives:
            try:
                aurocs[i] = roc_auc_score(y_true[:, i], y_prob[:, i])
            except Exception:
                pass
    return aurocs


def align_label_vocabs(label_vocab_gnn, label_vocab_trans):
    """
    Find common SNOMED codes between GNN and TRANS label vocabs.

    Returns
    -------
    common_codes : list[str]   codes present in both vocabs
    gnn_idx      : list[int]   positions in label_vocab_gnn
    trans_idx    : list[int]   positions in label_vocab_trans
    """
    gnn_set   = {c: i for i, c in enumerate(label_vocab_gnn)}
    trans_set = {c: i for i, c in enumerate(label_vocab_trans)}
    common    = sorted(set(gnn_set) & set(trans_set))
    gnn_idx   = [gnn_set[c]   for c in common]
    trans_idx = [trans_set[c] for c in common]
    log.info("  Label alignment: %d GNN, %d TRANS, %d common",
             len(label_vocab_gnn), len(label_vocab_trans), len(common))
    return common, gnn_idx, trans_idx


def get_quartile(freq, q25, q50, q75):
    """Return frequency quartile 1-4 for a single value."""
    if freq <= q25:
        return 1
    elif freq <= q50:
        return 2
    elif freq <= q75:
        return 3
    return 4


# ===========================================================================
# Output: CSV + 4 figures
# ===========================================================================

def save_outputs(gnn_results, trans_results, label_vocab_gnn, label_vocab_trans, args):
    """Write all CSVs and figures to args.out_dir."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(args.out_dir, exist_ok=True)

    gnn_leads   = sorted(gnn_results.keys())
    trans_leads = sorted(trans_results.keys())
    all_leads   = sorted(set(gnn_leads) | set(trans_leads))

    # Label alignment
    common_codes, gnn_idx, trans_idx = align_label_vocabs(
        label_vocab_gnn, label_vocab_trans
    )
    n_common = len(common_codes)

    # Compute macro-AUROC and retention stats
    def _macro(results, k):
        if k not in results:
            return float("nan")
        return float(np.nanmean(results[k]["auroc"]))

    def _n(results, k):
        return results[k]["n_valid"] if k in results else 0

    gnn_macro   = [_macro(gnn_results,   k) for k in all_leads]
    trans_macro  = [_macro(trans_results, k) for k in all_leads]
    gnn_n        = [_n(gnn_results,   k) for k in all_leads]
    trans_n      = [_n(trans_results, k) for k in all_leads]
    gnn_n0       = gnn_n[0]   if gnn_n   else 1
    trans_n0     = trans_n[0] if trans_n else 1
    gnn_pct      = [100.0 * n / gnn_n0   for n in gnn_n]
    trans_pct    = [100.0 * n / trans_n0 for n in trans_n]

    # Aligned per-label AUROC arrays keyed by lead
    def _aligned_gnn(k):
        if k not in gnn_results:
            return np.full(n_common, np.nan)
        return gnn_results[k]["auroc"][gnn_idx]

    def _aligned_trans(k):
        if k not in trans_results:
            return np.full(n_common, np.nan)
        return trans_results[k]["auroc"][trans_idx]

    # Label frequency (from k=0 y_true)
    if 0 in gnn_results:
        freq_gnn = gnn_results[0]["y_true"].mean(axis=0)[gnn_idx]
    else:
        freq_gnn = np.zeros(n_common)

    q25, q50, q75 = np.nanpercentile(freq_gnn, [25, 50, 75])
    quartiles = np.array([get_quartile(f, q25, q50, q75) for f in freq_gnn])

    # ── CSV ──────────────────────────────────────────────────────────────────
    csv_path = os.path.join(args.out_dir, "early_prediction_comparison.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        header = (
            ["snomed_code"]
            + [f"gnn_auroc_k{k}"   for k in all_leads]
            + [f"trans_auroc_k{k}" for k in all_leads]
            + ["freq", "quartile",
               "gnn_n_k0", "trans_n_k0"]
        )
        w.writerow(header)
        for i, code in enumerate(common_codes):
            row = (
                [code]
                + [f"{_aligned_gnn(k)[i]:.4f}"   if not np.isnan(_aligned_gnn(k)[i])   else "nan"
                   for k in all_leads]
                + [f"{_aligned_trans(k)[i]:.4f}"  if not np.isnan(_aligned_trans(k)[i]) else "nan"
                   for k in all_leads]
                + [f"{freq_gnn[i]:.6f}", str(int(quartiles[i])),
                   str(gnn_n0), str(trans_n0)]
            )
            w.writerow(row)
    log.info("Saved %s", csv_path)

    # ── Figure 1: fig_lead_vs_auroc_comparison.png ───────────────────────────
    log.info("Generating fig_lead_vs_auroc_comparison.png …")

    fig, ax1 = plt.subplots(figsize=(9, 6))
    ax1.plot(all_leads, gnn_macro, color=GNN_BLUE, marker="o", linewidth=2.5,
             markersize=9, linestyle="-",  label="PatientGNN E11 — Macro-AUROC")
    ax1.plot(all_leads, trans_macro, color=TRANS_RED, marker="s", linewidth=2.5,
             markersize=9, linestyle="--", label="TRANS E7 — Macro-AUROC")

    y_min = min(v for v in gnn_macro + trans_macro if not np.isnan(v))
    y_max = max(v for v in gnn_macro + trans_macro if not np.isnan(v))
    offset_dir = [1, -1, 1, -1]  # alternate above/below to avoid overlap
    for i, (k, gm, tm) in enumerate(zip(all_leads, gnn_macro, trans_macro)):
        d = offset_dir[i % 4]
        if not np.isnan(gm):
            ax1.annotate(f"{gm:.4f}", xy=(k, gm),
                         xytext=(0, 10 * d), textcoords="offset points",
                         ha="center", fontsize=8.5, color=GNN_BLUE, fontweight="bold")
        if not np.isnan(tm):
            ax1.annotate(f"{tm:.4f}", xy=(k, tm),
                         xytext=(0, -14 * d), textcoords="offset points",
                         ha="center", fontsize=8.5, color=TRANS_RED, fontweight="bold")

    ax1.set_xlabel("Lead time k  (number of recent visits withheld)", fontsize=11)
    ax1.set_ylabel("Macro-AUROC (mean over common labels)", fontsize=11, color="#333333")
    ax1.set_xticks(all_leads)
    ax1.set_ylim(max(0.50, y_min - 0.08), min(1.0, y_max + 0.08))
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    x  = np.asarray(all_leads, dtype=float)
    bw = 0.28
    ax2.bar(x - bw / 2, gnn_pct,   bw, alpha=0.25, color=GNN_BLUE,  label="GNN % retained")
    ax2.bar(x + bw / 2, trans_pct, bw, alpha=0.25, color=TRANS_RED, label="TRANS % retained")
    ax2.set_ylabel("% test patients retained", fontsize=10, color="grey")
    ax2.tick_params(axis="y", labelcolor="grey")
    ax2.set_ylim(0, 140)
    ax2.spines["right"].set_visible(True)

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="lower left", fontsize=9)
    ax1.set_title(
        "Early Warning Performance: PatientGNN E11 vs TRANS E7\n"
        "k=0: standard | k>0: k most-recent visits withheld",
        fontweight="bold",
    )
    plt.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "fig_lead_vs_auroc_comparison.png"), dpi=150)
    plt.close(fig)
    log.info("  Saved fig_lead_vs_auroc_comparison.png")

    # ── Figure 2: fig_auroc_drop_comparison.png ───────────────────────────────
    log.info("Generating fig_auroc_drop_comparison.png …")

    if 0 in gnn_results and 0 in trans_results:
        gnn_base   = gnn_macro[0]
        trans_base = trans_macro[0]
        lead_ks    = [k for k in all_leads if k > 0]
        gnn_drops  = [gnn_base   - _macro(gnn_results,   k) for k in lead_ks]
        trans_drops = [trans_base - _macro(trans_results, k) for k in lead_ks]

        x2 = np.arange(len(lead_ks))
        bw2 = 0.32
        fig, ax = plt.subplots(figsize=(8, 5))
        bars_g = ax.bar(x2 - bw2 / 2, gnn_drops,   bw2, color=GNN_BLUE,  alpha=0.85,
                        label="PatientGNN E11 AUROC drop")
        bars_t = ax.bar(x2 + bw2 / 2, trans_drops, bw2, color=TRANS_RED, alpha=0.85,
                        label="TRANS E7 AUROC drop")

        for bar, val in zip(list(bars_g) + list(bars_t),
                            gnn_drops + trans_drops):
            pct = 100.0 * val / (gnn_base if bar in list(bars_g) else trans_base + 1e-9)
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                    f"−{val:.3f}\n({pct:.1f}%)",
                    ha="center", va="bottom", fontsize=8.5)

        ax.set_xticks(x2)
        ax.set_xticklabels([f"k = {k}" for k in lead_ks])
        ax.set_xlabel("Lead time k")
        ax.set_ylabel("AUROC Drop from k=0 Baseline")
        ax.set_title("AUROC Degradation vs Lead Time\nPatientGNN E11 vs TRANS E7",
                     fontweight="bold")
        ax.legend()
        plt.tight_layout()
        fig.savefig(os.path.join(args.out_dir, "fig_auroc_drop_comparison.png"), dpi=150)
        plt.close(fig)
        log.info("  Saved fig_auroc_drop_comparison.png")
    else:
        log.warning("  Skipping fig_auroc_drop_comparison.png — k=0 missing for one model")

    # ── Figure 3: fig_cohort_shrinkage.png ────────────────────────────────────
    log.info("Generating fig_cohort_shrinkage.png …")

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(all_leads, gnn_n,   color=GNN_BLUE,  marker="o", linewidth=2.5,
             markersize=9, label=f"PatientGNN E11 (N₀={gnn_n0})")
    ax1.plot(all_leads, trans_n, color=TRANS_RED, marker="s", linewidth=2.5,
             markersize=9, linestyle="--", label=f"TRANS E7 (N₀={trans_n0})")

    for k, gn, tn in zip(all_leads, gnn_n, trans_n):
        ax1.text(k + 0.05, gn + 80, str(gn), fontsize=8.5, color=GNN_BLUE)
        ax1.text(k + 0.05, tn - 200, str(tn), fontsize=8.5, color=TRANS_RED)

    ax1.set_xlabel("Lead time k")
    ax1.set_ylabel("Number of patients retained", color="#333333")
    ax1.set_xticks(all_leads)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(all_leads, gnn_pct,   color=GNN_BLUE,  marker="o", linewidth=1.5,
             linestyle=":", alpha=0.6, markersize=6)
    ax2.plot(all_leads, trans_pct, color=TRANS_RED, marker="s", linewidth=1.5,
             linestyle=":", alpha=0.6, markersize=6)
    ax2.set_ylabel("% patients retained", color="grey")
    ax2.tick_params(axis="y", labelcolor="grey")
    ax2.set_ylim(0, 115)
    ax2.spines["right"].set_visible(True)

    ax1.legend(loc="upper right")
    ax1.set_title("Cohort Shrinkage vs Lead Time\n(patients with ≥ k+2 visits)",
                  fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "fig_cohort_shrinkage.png"), dpi=150)
    plt.close(fig)
    log.info("  Saved fig_cohort_shrinkage.png")

    # ── Figure 4: fig_per_label_lead_scatter.png ──────────────────────────────
    log.info("Generating fig_per_label_lead_scatter.png …")

    eval_leads = [k for k in [1, 2, 3] if k in gnn_results and k in trans_results]
    if eval_leads:
        ncols = len(eval_leads)
        fig, axes = plt.subplots(1, ncols, figsize=(5.5 * ncols, 5.5), sharey=False)
        if ncols == 1:
            axes = [axes]

        for ax, k in zip(axes, eval_leads):
            gnn_a   = _aligned_gnn(k)
            trans_a = _aligned_trans(k)
            valid   = ~(np.isnan(gnn_a) | np.isnan(trans_a))
            for q in [1, 2, 3, 4]:
                mask = valid & (quartiles == q)
                ax.scatter(
                    trans_a[mask], gnn_a[mask],
                    color=Q_COLORS[q], alpha=0.70, s=35, label=f"Q{q}",
                    zorder=3,
                )
            # Diagonal = equal performance
            lims = [0.50, 1.01]
            ax.plot(lims, lims, color="black", linewidth=1.2, linestyle="--",
                    alpha=0.7, zorder=2)
            # 0.80 reference lines
            ax.axhline(0.80, color="grey", linewidth=0.8, linestyle=":", alpha=0.6)
            ax.axvline(0.80, color="grey", linewidth=0.8, linestyle=":", alpha=0.6)

            ax.set_xlim(0.48, 1.02)
            ax.set_ylim(0.48, 1.02)
            ax.set_xlabel(f"TRANS AUROC @ k={k}", fontsize=10)
            ax.set_ylabel(f"PatientGNN AUROC @ k={k}", fontsize=10)
            ax.set_title(f"k = {k}  (N_labels={int(valid.sum())})", fontweight="bold")
            ax.legend(fontsize=8, loc="upper left", ncol=2)

            # Count GNN vs TRANS leading
            n_gnn_leads   = int(((gnn_a > trans_a) & valid).sum())
            n_trans_leads = int(((trans_a > gnn_a)  & valid).sum())
            ax.text(0.97, 0.03,
                    f"GNN better: {n_gnn_leads}\nTRANS better: {n_trans_leads}",
                    transform=ax.transAxes, fontsize=8, ha="right", va="bottom",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

        fig.suptitle(
            "Per-label AUROC: PatientGNN E11 vs TRANS E7 at Each Lead Time\n"
            "Colour = frequency quartile | diagonal = equal performance",
            fontsize=12, fontweight="bold",
        )
        plt.tight_layout(rect=[0, 0, 1, 0.93])
        fig.savefig(os.path.join(args.out_dir, "fig_per_label_lead_scatter.png"), dpi=150)
        plt.close(fig)
        log.info("  Saved fig_per_label_lead_scatter.png")
    else:
        log.warning("  Skipping fig_per_label_lead_scatter.png — insufficient lead times")

    # ── Console summary ───────────────────────────────────────────────────────
    log.info("\n%s", "=" * 72)
    log.info("EARLY PREDICTION COMPARISON SUMMARY")
    log.info("=" * 72)
    log.info("%-6s  %-14s  %-14s  %-12s  %-12s",
             "Lead k", "GNN Macro-AUROC", "TRANS Macro-AUROC",
             "GNN N", "TRANS N")
    for k, gm, tm, gn, tn in zip(all_leads, gnn_macro, trans_macro, gnn_n, trans_n):
        log.info("k=%-4d  %-14.4f  %-14.4f  %-12d  %-12d",
                 k, gm, tm, gn, tn)
    log.info("")
    if 0 in gnn_results and 0 in trans_results:
        gnn_b   = gnn_macro[0]
        trans_b = trans_macro[0]
        for k in [l for l in all_leads if l > 0]:
            gd = gnn_b   - _macro(gnn_results,   k)
            td = trans_b - _macro(trans_results, k)
            log.info("k=%d: GNN drop=%.4f (%.1f%%)  |  TRANS drop=%.4f (%.1f%%)",
                     k, gd, 100 * gd / (gnn_b + 1e-9),
                     td, 100 * td / (trans_b + 1e-9))
    log.info("=" * 72)


# ===========================================================================
# Main
# ===========================================================================

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    # ------------------------------------------------------------------
    # SECTION A: PatientGNN early prediction
    # ------------------------------------------------------------------
    log.info("\n%s\nSECTION A — PatientGNN E11\n%s", "=" * 60, "=" * 60)

    test_graphs, label_vocab_gnn, vocab_sizes = load_and_preprocess_gnn(args)
    num_labels_gnn = len(label_vocab_gnn)

    gnn_model = build_gnn_model(args, vocab_sizes, num_labels_gnn, device)
    gnn_results = run_gnn_lead_analysis(test_graphs, gnn_model, device, args)

    # Free GPU memory before loading TRANS
    del gnn_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # SECTION B: TRANS E7 early prediction
    # ------------------------------------------------------------------
    log.info("\n%s\nSECTION B — TRANS E7\n%s", "=" * 60, "=" * 60)

    # Locate TRANS root from checkpoint path
    trans_root = Path(args.trans_ckpt).parent.parent
    if not (trans_root / "utils.py").exists():
        for c in [Path(__file__).parent.parent / "TRANS",
                  Path(__file__).parent / "TRANS"]:
            if c.exists() and (c / "utils.py").exists():
                trans_root = c
                break

    (
        test_samples, Tokenizers, label_tokenizer,
        label_vocab_trans, mm_collate_fn_,
    ) = load_and_preprocess_trans(args)

    num_labels_trans = label_tokenizer.get_vocabulary_size()
    trans_model = build_trans_model(args, Tokenizers, num_labels_trans, device)

    trans_results = run_trans_lead_analysis(
        test_samples, Tokenizers, label_tokenizer,
        trans_model, device, args, trans_root,
    )

    del trans_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------
    log.info("\n%s\nSaving outputs\n%s", "=" * 60, "=" * 60)
    if not gnn_results:
        log.error("No GNN results produced — check patient graphs.")
        sys.exit(1)
    if not trans_results:
        log.error("No TRANS results produced — check TRANS checkpoint and data.")
        sys.exit(1)

    save_outputs(gnn_results, trans_results, label_vocab_gnn, label_vocab_trans, args)
    log.info("\nAll outputs written to: %s", args.out_dir)


if __name__ == "__main__":
    main()
