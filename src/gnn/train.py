"""
train.py
========
Training loop for PatientGNN.

Supports two tasks and two data sources:

  --task mort_read   --data_source omop      (original: OMOP graphs, binary)
  --task diagnosis   --data_source pyhealth  (comparable to TRANS: ICD, multi-label)
  --task diagnosis   --data_source pyhealth --no_kg  (ablation: no KG embeddings)

Usage (diagnosis, comparable to TRANS)
---------------------------------------
python train.py \\
    --task diagnosis --data_source pyhealth \\
    --mimic4_root /path/to/mimic-iv/hosp/ \\
    --out_dir /path/to/models/diagnosis \\
    --no_kg --device cuda:1

Usage (mortality + readmission, OMOP data)
-------------------------------------------
python train.py \\
    --task mort_read --data_source omop \\
    --graphs_path /data/.../patient_graphs.pt \\
    --vocab_path  /data/.../concept_vocab.json \\
    --kg_vocab    /data/.../kg_vocab.json \\
    --kg_matrix   /data/.../entity_embedding_matrix.npy \\
    --out_dir     /data/.../models \\
    --device cuda:1
"""

import argparse
import json
import logging
import sys
from math import log1p
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score
from torch_geometric.data import DataLoader, HeteroData
from tqdm import tqdm

from model import PatientGNN

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
    p = argparse.ArgumentParser(description="Train PatientGNN")

    # Task / data source
    p.add_argument("--task",        default="mort_read",
                   choices=["mort_read", "diagnosis"])
    p.add_argument("--data_source", default="omop",
                   choices=["omop", "pyhealth"])
    p.add_argument("--no_kg",       action="store_true",
                   help="Use random embeddings instead of KG embeddings (ablation)")

    # OMOP data paths (required for --data_source omop)
    p.add_argument("--graphs_path")
    p.add_argument("--vocab_path")
    p.add_argument("--kg_vocab")
    p.add_argument("--kg_matrix")

    # PyHealth data path (required for --data_source pyhealth)
    p.add_argument("--mimic4_root",
                   default=None)
    p.add_argument("--graph_cache", default=None,
                   help="Directory to cache built PyHealth graphs (avoids rebuild)")

    # Model
    p.add_argument("--hidden_dim",              type=int,   default=128)
    p.add_argument("--kg_embed_dim",            type=int,   default=128)
    p.add_argument("--num_gnn_layers",          type=int,   default=2)
    p.add_argument("--num_transformer_layers",  type=int,   default=2,
                   help="Transformer encoder layers in both graph and sequence branches")
    p.add_argument("--nhead",                   type=int,   default=4,
                   help="Attention heads (hidden_dim must be divisible by nhead)")
    p.add_argument("--pe_dim",                  type=int,   default=4,
                   help="Laplacian PE dim = MetaPath SE dim (each); visit_proj input += 2*pe_dim")
    p.add_argument("--alpha",                   type=float, default=0.8,
                   help="Graph branch weight in dual-branch fusion (0.8 = same as TRANS)")
    p.add_argument("--dropout",                 type=float, default=0.3)
    p.add_argument("--freeze_kg",      action="store_true", default=True)
    p.add_argument("--unfreeze_kg",    action="store_true")

    # Training
    p.add_argument("--epochs",      type=int,   default=50)
    p.add_argument("--batch_size",  type=int,   default=64)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--patience",    type=int,   default=10)
    p.add_argument("--num_workers", type=int,   default=4)

    # mort_read task weights
    p.add_argument("--w_mortality",   type=float, default=1.0)
    p.add_argument("--w_readmission", type=float, default=1.0)

    # Diagnosis pos_weight mode (controls precision/recall tradeoff)
    # E9  (default): --pw_mode cap50   → raw ratio clamped at 50  (high recall, lower AUPRC)
    # E10           : --pw_mode cap10   → raw ratio clamped at 10  (balanced)
    # E11           : --pw_mode sqrt5   → sqrt(ratio) clamped at 5 (TRANS-like precision)
    # E12           : --pw_mode none    → no weighting             (highest precision, risk of label collapse)
    p.add_argument("--pw_mode", default="cap50",
                   choices=["cap50", "cap10", "sqrt5", "none"],
                   help="pos_weight mode for diagnosis BCE loss")

    p.add_argument("--out_dir", required=True)
    p.add_argument("--device",  default="cuda")
    p.add_argument("--seed",    type=int, default=42)
    p.add_argument("--val_frac",  type=float, default=0.1)
    p.add_argument("--test_frac", type=float, default=0.15)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data: OMOP graphs (existing)
# ---------------------------------------------------------------------------

def split_graphs(graphs, val_frac, test_frac, seed):
    rng = np.random.default_rng(seed)
    n   = len(graphs)
    idx = rng.permutation(n)
    n_test = int(n * test_frac)
    n_val  = int(n * val_frac)
    test_idx  = idx[:n_test]
    val_idx   = idx[n_test:n_test + n_val]
    train_idx = idx[n_test + n_val:]
    train = [graphs[i] for i in train_idx]
    val   = [graphs[i] for i in val_idx]
    test  = [graphs[i] for i in test_idx]
    log.info("Split: train=%d  val=%d  test=%d", len(train), len(val), len(test))
    return train, val, test


def compute_pos_weight(graphs, label_attr):
    all_labels = torch.cat([getattr(g["visit"], label_attr) for g in graphs])
    n_pos = all_labels.sum().item()
    n_neg = len(all_labels) - n_pos
    pw = n_neg / max(n_pos, 1)
    log.info("  %s: pos=%d neg=%d pos_weight=%.2f", label_attr, int(n_pos), int(n_neg), pw)
    return torch.tensor([pw], dtype=torch.float)


# ---------------------------------------------------------------------------
# Data: PyHealth → PatientGNN instance-node graphs
# ---------------------------------------------------------------------------

def _convert_datetimes(datetime_strings):
    from datetime import datetime
    datetimes = [datetime.strptime(s, "%Y-%m-%d %H:%M") for s in datetime_strings]
    base = min(datetimes)
    deltas = [(dt - base).total_seconds() / 60.0 for dt in datetimes]  # minutes
    return deltas


def pyhealth_sample_to_graph(dp, cond_tok, proc_tok, drug_tok, label_tok, num_labels):
    """Convert one PyHealth patient sample to a PatientGNN HeteroData graph."""
    cond_hist  = dp["cond_hist"]    # list[list[str]]  per visit
    procedures = dp["procedures"]
    drugs      = dp["drugs"]
    adm_times  = dp["adm_time"]
    label_codes = dp["conditions"]  # next-visit diagnoses (labels)

    n_visits = len(cond_hist)

    # Visit features: [relative_time_norm, log1p(n_cond), log1p(n_proc), log1p(n_drug)]
    rel_times = _convert_datetimes(adm_times)
    max_t = max(rel_times) if max(rel_times) > 0.0 else 1.0
    visit_feats = [
        [
            rel_times[i] / max_t,
            log1p(len(cond_hist[i])),
            log1p(len(procedures[i])),
            log1p(len(drugs[i])),
        ]
        for i in range(n_visits)
    ]

    data = HeteroData()
    data["visit"].x = torch.tensor(visit_feats, dtype=torch.float32)

    # Temporal visit→visit edges
    if n_visits > 1:
        src = torch.arange(n_visits - 1, dtype=torch.long)
        dst = src + 1
        data["visit", "temporal_next", "visit"].edge_index = torch.stack([src, dst])
    else:
        data["visit", "temporal_next", "visit"].edge_index = torch.zeros(2, 0, dtype=torch.long)

    # Concept nodes and visit→concept edges for each domain
    for domain, hist, tok, edge_rel in [
        ("condition", cond_hist,  cond_tok, "has_condition"),
        ("procedure", procedures, proc_tok, "has_procedure"),
        ("drug",      drugs,      drug_tok, "has_drug"),
    ]:
        all_codes = list({c for visit_codes in hist for c in visit_codes})
        if not all_codes:
            data[domain].x = torch.zeros(1, 1, dtype=torch.float32)
            data["visit", edge_rel, domain].edge_index = torch.zeros(2, 0, dtype=torch.long)
            continue

        # Vocab index for each unique concept
        encoded = tok.batch_encode_2d([[c] for c in all_codes], padding=False)
        vocab_indices = [enc[0] if enc else 0 for enc in encoded]
        code_to_node  = {c: i for i, c in enumerate(all_codes)}

        data[domain].x = torch.tensor(vocab_indices, dtype=torch.float32).unsqueeze(1)

        v_src, c_dst = [], []
        for v_idx, visit_codes in enumerate(hist):
            for code in visit_codes:
                if code in code_to_node:
                    v_src.append(v_idx)
                    c_dst.append(code_to_node[code])

        if v_src:
            data["visit", edge_rel, domain].edge_index = torch.tensor(
                [v_src, c_dst], dtype=torch.long
            )
        else:
            data["visit", edge_rel, domain].edge_index = torch.zeros(2, 0, dtype=torch.long)

    # Multi-hot diagnosis label
    label_encoded = label_tok.batch_encode_2d([label_codes], padding=False)[0]
    y = torch.zeros(num_labels, dtype=torch.float32)
    for idx in label_encoded:
        if 0 <= idx < num_labels:
            y[idx] = 1.0
    data.y_diagnosis = y

    return data


def build_pyhealth_graphs(task_dataset, cond_tok, proc_tok, drug_tok, label_tok, num_labels):
    graphs, skipped = [], 0
    for dp in tqdm(task_dataset.samples, desc="Building graphs"):
        try:
            g = pyhealth_sample_to_graph(dp, cond_tok, proc_tok, drug_tok, label_tok, num_labels)
            graphs.append(g)
        except Exception:
            skipped += 1
    if skipped:
        log.warning("Skipped %d samples due to build errors", skipped)
    log.info("Built %d graphs", len(graphs))
    return graphs


def load_omop_diagnosis_data(args):
    """Load pre-built OMOP patient graphs for the diagnosis task."""
    graphs_path = Path(args.graphs_path)
    graphs_dir  = graphs_path.parent

    log.info("Loading OMOP patient graphs from %s …", graphs_path)
    graphs = torch.load(graphs_path, weights_only=False)
    log.info("  Loaded %d graphs", len(graphs))

    # Filter to graphs that have y_diagnosis (built with --top_k > 0)
    before = len(graphs)
    graphs = [g for g in graphs if hasattr(g, "y_diagnosis")]
    if len(graphs) < before:
        log.warning("  Dropped %d graphs without y_diagnosis (re-run build_patient_graphs.py --top_k 275)",
                    before - len(graphs))

    # Label vocabulary
    label_vocab_path = graphs_dir / "label_vocab.json"
    with open(label_vocab_path) as f:
        label_vocab = json.load(f)
    num_labels = len(label_vocab)
    log.info("  Diagnosis labels: %d", num_labels)

    # Vocab sizes for no-KG embedding tables
    vocab_path = Path(args.vocab_path) if args.vocab_path else graphs_dir / "concept_vocab.json"
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
    log.info("  Vocab sizes — cond: %d  proc: %d  drug: %d",
             vocab_sizes["cond"], vocab_sizes["proc"], vocab_sizes["drug"])

    # ---- Normalise visit features -----------------------------------------------
    # visit_x columns: [los_norm(0), age_norm(1), gender(2),
    #                   n_labs(3), mean_value(4), std_value(5), pct_abnormal(6)]
    # Columns 0-2 and 6 are already in [0,1] or log scale.
    # Columns 3-5 are raw counts / raw lab values — must be normalised to prevent
    # gradient explosion during training (mean_value mixes disparate lab scales).
    log.info("  Normalising visit features (cols 3-5: n_labs, mean_value, std_value) …")
    all_x = torch.cat([g["visit"].x for g in graphs], dim=0)   # [total_visits, 7]
    # col 3: n_labs → log scale (non-negative count); cols 4-5: z-score
    mean4, std4 = all_x[:, 4].mean().item(), all_x[:, 4].std().item() + 1e-6
    mean5, std5 = all_x[:, 5].mean().item(), all_x[:, 5].std().item() + 1e-6
    log.info("    mean_value  raw mean=%.2f std=%.2f", mean4, std4)
    log.info("    std_value   raw mean=%.2f std=%.2f", mean5, std5)
    for g in graphs:
        x = g["visit"].x.clone()
        x[:, 3] = torch.log1p(x[:, 3])
        x[:, 4] = (x[:, 4] - mean4) / std4
        x[:, 5] = (x[:, 5] - mean5) / std5
        g["visit"].x = x

    # ---- Remove last-visit condition edges to prevent label leakage -------------
    # y_diagnosis = last visit's top-K conditions. Those same conditions are nodes
    # in the graph connected to the last visit. A GNN can trivially route
    # last_visit → condition_nodes → back → output, reading the answer from input.
    # TRANS (E7) avoids this by construction: cond_hist excludes the last visit.
    # We match that by dropping has_condition edges from the last visit node.
    # The last visit's non-condition features (LOS, labs, age) remain in visit.x.
    log.info("  Removing has_condition edges from last visit node (prevent leakage) …")
    removed = 0
    for g in graphs:
        n_visits = g["visit"].x.size(0)
        last_v   = n_visits - 1
        ei = g["visit", "has_condition", "condition"].edge_index   # [2, E]
        if ei.numel() > 0:
            mask = ei[0] != last_v
            removed += int((~mask).sum().item())
            g["visit", "has_condition", "condition"].edge_index = ei[:, mask]
    log.info("    Removed %d last-visit→condition edges across %d graphs", removed, len(graphs))

    return graphs, num_labels, vocab_sizes


def load_pyhealth_data(args):
    from pyhealth.datasets import MIMIC4Dataset
    from pyhealth.tokenizer import Tokenizer
    from data.Task import diag_prediction_mimic4_fn   # reuse from TRANS

    log.info("Loading MIMIC4Dataset …")
    dataset = MIMIC4Dataset(
        root=args.mimic4_root,
        dev=False,
        tables=["diagnoses_icd", "procedures_icd", "prescriptions"],
        code_mapping={"NDC": ("ATC", {"target_kwargs": {"level": 3}})},
        refresh_cache=False,
    )
    task_dataset = dataset.set_task(task_fn=diag_prediction_mimic4_fn)

    keys = ["cond_hist", "procedures", "drugs"]
    tokenizers = {
        k: Tokenizer(tokens=task_dataset.get_all_tokens(k), special_tokens=["<pad>"])
        for k in keys
    }
    label_tok = Tokenizer(tokens=task_dataset.get_all_tokens("conditions"))
    num_labels = label_tok.get_vocabulary_size()

    log.info("Labels (diagnosis codes): %d", num_labels)
    log.info("Cond vocab: %d  Proc vocab: %d  Drug vocab: %d",
             tokenizers["cond_hist"].get_vocabulary_size(),
             tokenizers["procedures"].get_vocabulary_size(),
             tokenizers["drugs"].get_vocabulary_size())

    # Build or load cached graphs
    cache_path = Path(args.graph_cache) / "pyhealth_graphs.pt" if args.graph_cache else None
    if cache_path and cache_path.exists():
        log.info("Loading cached graphs from %s …", cache_path)
        graphs = torch.load(cache_path, weights_only=False)
    else:
        graphs = build_pyhealth_graphs(
            task_dataset,
            tokenizers["cond_hist"],
            tokenizers["procedures"],
            tokenizers["drugs"],
            label_tok,
            num_labels,
        )
        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(graphs, cache_path)
            log.info("Saved graphs to %s", cache_path)

    vocab_sizes = {
        "cond":  tokenizers["cond_hist"].get_vocabulary_size(),
        "proc":  tokenizers["procedures"].get_vocabulary_size(),
        "drug":  tokenizers["drugs"].get_vocabulary_size(),
    }
    return graphs, num_labels, vocab_sizes


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def code_level(y_true, y_prob, ks=(10, 20, 30)):
    total_pos = np.where(y_true == 1)[0].shape[0]
    recalls = []
    for k in ks:
        correct = 0
        for i in range(len(y_prob)):
            top_k = np.argsort(-y_prob[i])[:k]
            correct += y_true[i][top_k].sum()
        recalls.append(float(correct) / max(total_pos, 1))
    return recalls


def visit_level(y_true, y_prob, ks=(10, 20, 30)):
    precisions = []
    for k in ks:
        per_patient = []
        for i in range(len(y_true)):
            n_pos = y_true[i].sum()
            denom = min(k, n_pos)
            if denom == 0:
                per_patient.append(0.0)
                continue
            top_k = np.argsort(-y_prob[i])[:k]
            per_patient.append(float(y_true[i][top_k].sum()) / denom)
        precisions.append(float(np.mean(per_patient)))
    return precisions


def compute_diagnosis_metrics(y_true, y_prob):
    aurocs, auprcs = [], []
    for i in range(y_true.shape[1]):
        if y_true[:, i].sum() > 0:
            aurocs.append(roc_auc_score(y_true[:, i], y_prob[:, i]))
            auprcs.append(average_precision_score(y_true[:, i], y_prob[:, i]))
    return {
        "auroc_macro":    float(np.mean(aurocs)) if aurocs else float("nan"),
        "auprc_macro":    float(np.mean(auprcs)) if auprcs else float("nan"),
        "code_recall":    code_level(y_true, y_prob),
        "visit_precision": visit_level(y_true, y_prob),
    }


# ---------------------------------------------------------------------------
# Train / eval one epoch
# ---------------------------------------------------------------------------

def run_epoch(model, loader, optimizer, device, args, train: bool):
    model.train() if train else model.eval()

    total_loss = 0.0
    n_samples  = 0
    task = args.task

    # Accumulators for metrics
    y_true_all, y_prob_all = [], []   # diagnosis
    mort_logits_all, mort_labels_all  = [], []   # mort_read
    read_logits_all, read_labels_all  = [], []

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in loader:
            batch = batch.to(device)

            out = model(batch, task=task)

            if task == "diagnosis":
                logits = out["diagnosis"]            # [B, num_labels]
                labels = batch.y_diagnosis.view(logits.shape)  # [B*num_labels] → [B, num_labels]
                pw = getattr(args, "_diag_pos_weight", None)
                loss = F.binary_cross_entropy_with_logits(
                    logits, labels,
                    pos_weight=pw.to(device) if pw is not None else None,
                )

                if train and torch.isfinite(loss):
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

                if torch.isfinite(loss):
                    total_loss += loss.item() * labels.size(0)
                    n_samples  += labels.size(0)

                y_true_all.append(labels.detach().cpu().float().numpy())
                y_prob_all.append(torch.sigmoid(logits).detach().cpu().float().numpy())

            else:  # mort_read
                mort_labels = batch["visit"].y_mortality.float()
                read_labels = batch["visit"].y_readmission.float()

                loss_mort = F.binary_cross_entropy_with_logits(
                    out["mortality"], mort_labels,
                    pos_weight=args._mort_pos_weight.to(device),
                )
                loss_read = F.binary_cross_entropy_with_logits(
                    out["readmission"], read_labels,
                    pos_weight=args._read_pos_weight.to(device),
                )
                loss = args.w_mortality * loss_mort + args.w_readmission * loss_read

                if train and torch.isfinite(loss):
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

                if torch.isfinite(loss):
                    total_loss += loss.item() * mort_labels.size(0)
                    n_samples  += mort_labels.size(0)

                mort_logits_all.append(out["mortality"].detach().cpu())
                mort_labels_all.append(mort_labels.cpu())
                read_logits_all.append(out["readmission"].detach().cpu())
                read_labels_all.append(read_labels.cpu())

    avg_loss = total_loss / max(n_samples, 1)

    if task == "diagnosis":
        y_true = np.concatenate(y_true_all, axis=0)
        y_prob = np.concatenate(y_prob_all, axis=0)
        y_prob = np.nan_to_num(y_prob, nan=0.5)
        m = compute_diagnosis_metrics(y_true, y_prob)
        m["loss"] = avg_loss
        return m, (y_true, y_prob)

    else:
        def safe_auc(yt, yp):
            yt, yp = np.concatenate(yt), np.concatenate(yp)
            if yt.sum() == 0 or yt.sum() == len(yt):
                return float("nan")
            return roc_auc_score(yt, yp)

        def safe_ap(yt, yp):
            yt, yp = np.concatenate(yt), np.concatenate(yp)
            if yt.sum() == 0:
                return float("nan")
            return average_precision_score(yt, yp)

        m = {
            "loss":       avg_loss,
            "mort_auroc": safe_auc(mort_labels_all, [x.numpy() for x in mort_logits_all]),
            "mort_auprc": safe_ap(mort_labels_all,  [x.numpy() for x in mort_logits_all]),
            "read_auroc": safe_auc(read_labels_all, [x.numpy() for x in read_logits_all]),
            "read_auprc": safe_ap(read_labels_all,  [x.numpy() for x in read_logits_all]),
        }
        return m, None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    if args.unfreeze_kg:
        args.freeze_kg = False

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        log.warning("CUDA not available, using CPU")
        device = "cpu"

    out_dir = Path(args.out_dir)
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (out_dir / "results").mkdir(parents=True, exist_ok=True)

    # ---- Load data ----------------------------------------------------------
    num_labels  = 0
    vocab_sizes = {}

    if args.data_source == "omop" and args.task == "diagnosis":
        graphs, num_labels, vocab_sizes = load_omop_diagnosis_data(args)
        train_graphs, val_graphs, test_graphs = split_graphs(
            graphs, args.val_frac, args.test_frac, args.seed
        )
        # pos_weight per label: counters class imbalance in sparse multi-label BCE
        all_labels = torch.stack([g.y_diagnosis for g in train_graphs])  # [N, num_labels]
        n_pos = all_labels.sum(0)                                         # [num_labels]
        n_neg = len(train_graphs) - n_pos
        pw_mode = getattr(args, "pw_mode", "cap50")
        if pw_mode == "cap50":
            args._diag_pos_weight = (n_neg / n_pos.clamp(min=1)).clamp(max=50.0)
        elif pw_mode == "cap10":                                          # E10
            args._diag_pos_weight = (n_neg / n_pos.clamp(min=1)).clamp(max=10.0)
        elif pw_mode == "sqrt5":                                          # E11
            args._diag_pos_weight = (n_neg / n_pos.clamp(min=1)).sqrt().clamp(max=5.0)
        elif pw_mode == "none":                                           # E12
            args._diag_pos_weight = None
        else:
            raise ValueError(f"Unknown --pw_mode: {pw_mode}")

        if args._diag_pos_weight is not None:
            log.info("  Diagnosis pos_weight (mode=%s): min=%.1f  max=%.1f  mean=%.1f",
                     pw_mode,
                     args._diag_pos_weight.min().item(),
                     args._diag_pos_weight.max().item(),
                     args._diag_pos_weight.mean().item())
        else:
            log.info("  Diagnosis pos_weight (mode=%s): disabled", pw_mode)
        visit_feat_dim = 7

    elif args.data_source == "omop":  # mort_read
        log.info("Loading OMOP patient_graphs.pt …")
        graphs = torch.load(args.graphs_path, weights_only=False)
        log.info("  Loaded %d graphs", len(graphs))
        train_graphs, val_graphs, test_graphs = split_graphs(
            graphs, args.val_frac, args.test_frac, args.seed
        )
        log.info("Computing class weights …")
        args._mort_pos_weight = compute_pos_weight(train_graphs, "y_mortality")
        args._read_pos_weight = compute_pos_weight(train_graphs, "y_readmission")
        visit_feat_dim = 7

    else:  # pyhealth
        graphs, num_labels, vocab_sizes = load_pyhealth_data(args)
        # Same 75/10/15 split as TRANS
        train_graphs, val_graphs, test_graphs = split_graphs(
            graphs, args.val_frac, args.test_frac, args.seed
        )
        visit_feat_dim = 4  # [rel_time, log1p(n_cond), log1p(n_proc), log1p(n_drug)]

    train_loader = DataLoader(train_graphs, batch_size=args.batch_size,
                              shuffle=True,  num_workers=args.num_workers, pin_memory=True)
    val_loader   = DataLoader(val_graphs,   batch_size=args.batch_size * 2,
                              shuffle=False, num_workers=args.num_workers)
    test_loader  = DataLoader(test_graphs,  batch_size=args.batch_size * 2,
                              shuffle=False, num_workers=args.num_workers)

    # ---- Build model --------------------------------------------------------
    log.info("Building PatientGNN (task=%s, data=%s, no_kg=%s) …",
             args.task, args.data_source, args.no_kg)

    if args.no_kg or args.data_source == "pyhealth":
        model = PatientGNN.from_config(
            no_kg=True,
            cond_vocab_size=vocab_sizes.get("cond", 0),
            proc_vocab_size=vocab_sizes.get("proc", 0),
            drug_vocab_size=vocab_sizes.get("drug", 0),
            kg_embed_dim=args.kg_embed_dim,
            hidden_dim=args.hidden_dim,
            num_gnn_layers=args.num_gnn_layers,
            num_transformer_layers=args.num_transformer_layers,
            nhead=args.nhead,
            pe_dim=args.pe_dim,
            alpha=args.alpha,
            dropout=args.dropout,
            visit_feat_dim=visit_feat_dim,
            num_labels=num_labels,
        )
    else:
        model = PatientGNN.from_config(
            concept_vocab_path=args.vocab_path,
            kg_vocab_path=args.kg_vocab,
            kg_matrix_path=args.kg_matrix,
            hidden_dim=args.hidden_dim,
            num_gnn_layers=args.num_gnn_layers,
            num_transformer_layers=args.num_transformer_layers,
            nhead=args.nhead,
            pe_dim=args.pe_dim,
            alpha=args.alpha,
            dropout=args.dropout,
            freeze_kg=args.freeze_kg,
            visit_feat_dim=visit_feat_dim,
            num_labels=num_labels,
        )

    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("  Trainable parameters: %s", f"{n_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5
    )

    # ---- Training loop ------------------------------------------------------
    monitor_key   = "auroc_macro" if args.task == "diagnosis" else "mort_auroc"
    best_val_score = 0.0
    patience_counter = 0
    history = []
    ckpt_path = out_dir / "checkpoints" / "best_model.pt"

    for epoch in range(1, args.epochs + 1):
        train_m, _ = run_epoch(model, train_loader, optimizer, device, args, train=True)
        val_m, _   = run_epoch(model, val_loader,   optimizer, device, args, train=False)

        val_score = val_m.get(monitor_key, 0.0)
        if not np.isfinite(val_score):
            val_score = 0.0
        scheduler.step(val_score)

        if args.task == "diagnosis":
            log.info(
                "Epoch %03d | loss=%.4f | train AUROC=%.4f AUPRC=%.4f | "
                "val AUROC=%.4f AUPRC=%.4f",
                epoch,
                train_m["loss"],
                train_m["auroc_macro"], train_m["auprc_macro"],
                val_m["auroc_macro"],   val_m["auprc_macro"],
            )
        else:
            log.info(
                "Epoch %03d | loss=%.4f | mort AUROC=%.4f | read AUROC=%.4f | "
                "val mort=%.4f read=%.4f",
                epoch, train_m["loss"],
                train_m["mort_auroc"], train_m["read_auroc"],
                val_m["mort_auroc"],   val_m["read_auroc"],
            )

        history.append({"epoch": epoch, "train": train_m, "val": val_m})

        if val_score > best_val_score:
            best_val_score = val_score
            patience_counter = 0
            torch.save(
                {"epoch": epoch, "model_state": model.state_dict(),
                 "val_metrics": val_m, "args": vars(args)},
                ckpt_path,
            )
            log.info("  Saved checkpoint (val_%s=%.4f)", monitor_key, best_val_score)
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                log.info("Early stopping at epoch %d", epoch)
                break

    # ---- Test evaluation ----------------------------------------------------
    log.info("Loading best model for test evaluation …")
    ckpt = torch.load(ckpt_path, weights_only=False)
    model.load_state_dict(ckpt["model_state"])

    test_m, _ = run_epoch(model, test_loader, optimizer, device, args, train=False)

    if args.task == "diagnosis":
        log.info("\n=== Test Results ===")
        log.info("  AUROC (macro):     %.4f", test_m["auroc_macro"])
        log.info("  AUPRC (macro):     %.4f", test_m["auprc_macro"])
        log.info("  Code Recall@K:     %s",   test_m["code_recall"])
        log.info("  Visit Precision@K: %s",   test_m["visit_precision"])
    else:
        log.info("\n=== Test Results ===")
        log.info("  Mortality  AUROC=%.4f  AUPRC=%.4f",
                 test_m["mort_auroc"], test_m["mort_auprc"])
        log.info("  Readmission AUROC=%.4f  AUPRC=%.4f",
                 test_m["read_auroc"], test_m["read_auprc"])

    # Exclude internal tensor attributes (e.g. _diag_pos_weight) from JSON
    serializable_args = {k: v for k, v in vars(args).items()
                         if not isinstance(v, torch.Tensor)}
    results = {
        "task": args.task,
        "data_source": args.data_source,
        "no_kg": args.no_kg,
        "test": test_m,
        "best_val": ckpt["val_metrics"],
        "best_epoch": int(ckpt["epoch"]),
        "history": [{"epoch": r["epoch"],
                     "train_loss": r["train"]["loss"],
                     "val_score": r["val"].get(monitor_key)}
                    for r in history],
        "args": serializable_args,
    }
    results_path = out_dir / "results" / "metrics.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info("Results saved to %s", results_path)


if __name__ == "__main__":
    main()
