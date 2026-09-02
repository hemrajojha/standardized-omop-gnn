"""
model.py  (v2)
==============
PatientGNN — TRANS-aligned Heterogeneous Temporal GNN for patient graphs.

Architecture aligned to TRANS (IJCAI 2024):
  - HGTConv  (relation-specific attention)           replaces SAGEConv
  - Laplacian PE + MetaPath Random Walk SE            added to visit nodes
  - TransformerEncoder over visit sequence            replaces GRU
  - Dual-branch fusion (graph + sequence, alpha=0.8) replaces single pipeline

PatientGNN-specific contributions (unchanged vs TRANS):
  - Instance nodes — only concepts present in this patient's visits
  - KG pretrained RotatE embeddings from OMOP concept_relationship
  - OMOP SNOMED data support

Supports two tasks:
    mort_read  : per-visit mortality + 30-day readmission (binary, OMOP data)
    diagnosis  : next-visit diagnosis prediction (multi-label, PyHealth or OMOP)

Supports two concept embedding modes:
    KG mode    : pretrained RotatE embeddings from OMOP concept_relationship
    no-KG mode : learned nn.Embedding (random init) — for ablation / ICD data

Input graph schema
------------------
Node types:
    visit       [N_v, visit_feat_dim]
    condition   [N_c, 1]    vocab index (float)
    procedure   [N_p, 1]    vocab index (float)
    drug        [N_d, 1]    vocab index (float)

Edge types (stored as visit→concept, reversed for HGTConv):
    (visit, has_condition,  condition)
    (visit, has_procedure,  procedure)
    (visit, has_drug,       drug)
    (visit, temporal_next,  visit)
"""

import json
import logging
import math
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import HGTConv
from torch_geometric.utils import to_dense_batch

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Graph schema (must match build_patient_graphs.py and pyhealth_sample_to_graph)
# ---------------------------------------------------------------------------

_NODE_TYPES = ["visit", "condition", "procedure", "drug"]

_EDGE_TYPES = [
    ("condition", "r_has_condition", "visit"),
    ("procedure", "r_has_procedure", "visit"),
    ("drug",      "r_has_drug",      "visit"),
    ("visit",     "temporal_next",   "visit"),
]

_METADATA = (_NODE_TYPES, _EDGE_TYPES)


# ---------------------------------------------------------------------------
# KG embedding helpers (unchanged)
# ---------------------------------------------------------------------------

def _build_domain_embedding_table(
    domain: str,
    concept_vocab: dict,
    kg_vocab: dict,
    kg_matrix: np.ndarray,
    kg_embed_dim: int,
) -> tuple[np.ndarray, int, int]:
    domain_vocab = concept_vocab.get(domain, {})
    if not domain_vocab:
        return np.zeros((0, kg_embed_dim), dtype=np.float32), 0, 0

    vocab_size = max(int(v) for v in domain_vocab.values()) + 1
    rng = np.random.default_rng(42)
    table = rng.normal(0, 0.01, (vocab_size, kg_embed_dim)).astype(np.float32)

    n_matched = 0
    for cid_str, vidx in domain_vocab.items():
        kg_idx = kg_vocab.get(cid_str)
        if kg_idx is not None:
            table[int(vidx)] = kg_matrix[int(kg_idx)]
            n_matched += 1

    return table, n_matched, vocab_size


def load_kg_embeddings(
    concept_vocab_path: str | Path,
    kg_vocab_path: str | Path,
    kg_matrix_path: str | Path,
) -> dict[str, tuple[np.ndarray, int]]:
    with open(concept_vocab_path) as f:
        concept_vocab = json.load(f)
    with open(kg_vocab_path) as f:
        kg_vocab = json.load(f)

    kg_matrix = np.load(kg_matrix_path).astype(np.float32)
    kg_embed_dim = kg_matrix.shape[1]
    log.info("KG matrix shape: %s", kg_matrix.shape)

    results = {}
    for domain in ("condition", "procedure", "drug"):
        table, n_matched, vocab_size = _build_domain_embedding_table(
            domain, concept_vocab, kg_vocab, kg_matrix, kg_embed_dim
        )
        log.info(
            "  %s: vocab_size=%d, kg_matched=%d (%.1f%%)",
            domain, vocab_size, n_matched,
            100 * n_matched / max(vocab_size, 1),
        )
        results[domain] = (table, n_matched)

    return results


# ---------------------------------------------------------------------------
# Positional / structural encoding helpers
# ---------------------------------------------------------------------------

def _sinusoidal_lpe(positions: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Laplacian Positional Encoding for visit nodes.

    For a temporal chain graph (v0 → v1 → … → v_{T-1}), the eigenvectors
    of the graph Laplacian are discrete cosines — equivalent to sinusoidal
    positional encoding indexed by visit order.  This formulation is O(T)
    and avoids the scipy eigsh cost used in TRANS for large vocab graphs.

    Parameters
    ----------
    positions : [N] long tensor  — 0-based position of each visit in its patient
    dim       : PE output dimension (should be even)

    Returns
    -------
    [N, dim] float tensor
    """
    N = positions.size(0)
    pe = torch.zeros(N, dim, device=positions.device)
    pos = positions.float().unsqueeze(1)                              # [N, 1]
    half = dim // 2
    div = torch.exp(
        torch.arange(half, device=positions.device).float()
        * -(math.log(10000.0) / half)
    )                                                                  # [half]
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe                                                          # [N, dim]


def _metapath_rw_se(
    data: HeteroData,
    num_visits: int,
    se_dim: int,
    device: torch.device,
) -> torch.Tensor:
    """
    MetaPath Random Walk Structural Encoding for visit nodes.

    For each clinical domain (condition, procedure, drug), computes the
    normalised degree of each visit in the visit→concept→visit metapath,
    i.e. the diagonal of the normalised metapath adjacency A_vc @ A_cv.

    Features are constructed by cycling across domains and powers
    [cond^1, proc^1, drug^1, cond^2, proc^2, …] to fill se_dim slots,
    matching the spirit of TRANS's random-walk diagonal encoding.

    Parameters
    ----------
    data      : batched HeteroData (edge indices are globally offset)
    num_visits: total visits across all patients in the batch
    se_dim    : number of SE output features (should be divisible by 3)
    device    : target device

    Returns
    -------
    [num_visits, se_dim] float tensor
    """
    fwd_keys = [
        ("visit", "has_condition", "condition"),
        ("visit", "has_procedure", "procedure"),
        ("visit", "has_drug",      "drug"),
    ]

    # Degree of each visit in each domain metapath (normalised to [0,1])
    degs = []
    for fwd_key in fwd_keys:
        if fwd_key in data.edge_types and data[fwd_key].edge_index.size(1) > 0:
            ei = data[fwd_key].edge_index                   # [2, E]
            deg = torch.zeros(num_visits, device=device)
            deg.scatter_add_(
                0,
                ei[0].to(device),
                torch.ones(ei.size(1), device=device),
            )
            deg = deg / deg.max().clamp(min=1.0)
        else:
            deg = torch.zeros(num_visits, device=device)
        degs.append(deg)

    # Build se_dim features: cycle [cond^1, proc^1, drug^1, cond^2, proc^2, …]
    features = []
    for i in range(se_dim):
        domain_idx = i % 3
        power      = i // 3 + 1
        v = degs[domain_idx].pow(power)
        v = v / v.max().clamp(min=1e-8)
        features.append(v)

    return torch.stack(features, dim=1)      # [num_visits, se_dim]


# ---------------------------------------------------------------------------
# PatientGNN
# ---------------------------------------------------------------------------

class PatientGNN(nn.Module):
    """
    PatientGNN v2 — TRANS-aligned heterogeneous temporal GNN.

    Architecture is identical to TRANS on the five aligned dimensions:
      - HGTConv            (Hu et al., WWW 2020)
      - Laplacian PE       (Dwivedi & Bresson, 2021)
      - MetaPath RW SE     (TRANS, IJCAI 2024)
      - TransformerEncoder (Vaswani et al., 2017)
      - Dual-branch fusion (alpha * graph + (1-alpha) * seq)

    PatientGNN-specific (vs TRANS):
      - Instance-node graphs  (only concepts present in this patient)
      - KG pretrained RotatE embeddings for concept initialisation
      - OMOP SNOMED concept space

    Parameters
    ----------
    visit_feat_dim       : int   raw visit feature dimension (7 OMOP / 4 PyHealth)
    hidden_dim           : int   internal representation dimension
    num_gnn_layers       : int   number of HGTConv layers
    num_transformer_layers: int  number of Transformer encoder layers (both branches)
    nhead                : int   number of attention heads (hidden_dim % nhead == 0)
    dropout              : float dropout probability
    pe_dim               : int   Laplacian PE dim = MetaPath SE dim
                                 visit_proj input = visit_feat_dim + 2*pe_dim
    kg_embed_dim         : int   concept embedding dimension

    KG mode (no_kg=False, default):
        cond/proc/drug_embed_table : np.ndarray  pretrained embedding tables
        freeze_kg                  : bool        freeze KG weights

    No-KG mode (no_kg=True):
        cond/proc/drug_vocab_size  : int         vocabulary sizes for random init

    Task:
        num_labels : int  > 0  → diagnosis head (multi-label, last-visit pooling)
                          0    → mortality + readmission heads (per-visit)

    Fusion:
        alpha : float  weight of graph branch in dual-branch output (default 0.8)
    """

    def __init__(
        self,
        visit_feat_dim: int,
        hidden_dim: int,
        num_gnn_layers: int = 2,
        num_transformer_layers: int = 2,
        nhead: int = 4,
        dropout: float = 0.3,
        pe_dim: int = 4,
        kg_embed_dim: int = 128,
        # KG embedding tables (None in no-KG mode)
        cond_embed_table: Optional[np.ndarray] = None,
        proc_embed_table: Optional[np.ndarray] = None,
        drug_embed_table: Optional[np.ndarray] = None,
        freeze_kg: bool = True,
        # No-KG mode
        no_kg: bool = False,
        cond_vocab_size: int = 0,
        proc_vocab_size: int = 0,
        drug_vocab_size: int = 0,
        # Diagnosis task
        num_labels: int = 0,
        # Dual-branch fusion weight
        alpha: float = 0.8,
    ):
        super().__init__()

        assert hidden_dim % nhead == 0, (
            f"hidden_dim ({hidden_dim}) must be divisible by nhead ({nhead})"
        )

        self.hidden_dim = hidden_dim
        self.pe_dim     = pe_dim
        self.dropout    = dropout
        self.num_labels = num_labels
        self.alpha      = alpha

        # ---- Concept embeddings ------------------------------------------------
        self.cond_embed = self._make_embedding(
            cond_embed_table, cond_vocab_size, kg_embed_dim, freeze_kg, no_kg
        )
        self.proc_embed = self._make_embedding(
            proc_embed_table, proc_vocab_size, kg_embed_dim, freeze_kg, no_kg
        )
        self.drug_embed = self._make_embedding(
            drug_embed_table, drug_vocab_size, kg_embed_dim, freeze_kg, no_kg
        )

        # ---- Input projections -------------------------------------------------
        # Visit: raw features + Laplacian PE (pe_dim) + MetaPath SE (pe_dim)
        self.visit_proj   = nn.Sequential(
            nn.Linear(visit_feat_dim + 2 * pe_dim, hidden_dim), nn.ReLU()
        )
        self.concept_proj = nn.Linear(kg_embed_dim, hidden_dim)

        # ---- HGTConv layers (replaces SAGEConv) --------------------------------
        # HGTConv: relation-specific attention + type-specific projections
        # (Hu et al., "Heterogeneous Graph Transformer", WWW 2020)
        self.convs = nn.ModuleList([
            HGTConv(hidden_dim, hidden_dim, _METADATA, heads=nhead)
            for _ in range(num_gnn_layers)
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_gnn_layers)
        ])

        # ---- Graph branch: Transformer over GNN-enriched visit sequence --------
        # (replaces GRU; global self-attention over visits)
        graph_tf_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.graph_transformer = nn.TransformerEncoder(
            graph_tf_layer, num_layers=num_transformer_layers
        )

        # ---- Sequence branch: Transformer over raw visit features --------------
        # Parallel branch; receives no GNN output (pure sequence signal)
        self.seq_proj = nn.Sequential(
            nn.Linear(visit_feat_dim, hidden_dim), nn.ReLU()
        )
        seq_tf_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.seq_transformer = nn.TransformerEncoder(
            seq_tf_layer, num_layers=num_transformer_layers
        )

        # ---- Prediction heads --------------------------------------------------
        if num_labels > 0:
            self.graph_head = nn.Linear(hidden_dim, num_labels)
            self.seq_head   = nn.Linear(hidden_dim, num_labels)
        else:
            self.graph_mort_head = nn.Linear(hidden_dim, 1)
            self.graph_read_head = nn.Linear(hidden_dim, 1)
            self.seq_mort_head   = nn.Linear(hidden_dim, 1)
            self.seq_read_head   = nn.Linear(hidden_dim, 1)

        self._init_weights()

    # -----------------------------------------------------------------------

    @staticmethod
    def _make_embedding(
        table: Optional[np.ndarray],
        vocab_size: int,
        embed_dim: int,
        freeze: bool,
        no_kg: bool,
    ) -> nn.Embedding:
        if not no_kg and table is not None and table.shape[0] > 0:
            weight = torch.tensor(table, dtype=torch.float)
            return nn.Embedding.from_pretrained(weight, freeze=freeze)
        return nn.Embedding(max(vocab_size, 1), embed_dim)

    def _init_weights(self):
        if self.num_labels > 0:
            heads = [self.graph_head, self.seq_head, self.concept_proj]
        else:
            heads = [
                self.graph_mort_head, self.graph_read_head,
                self.seq_mort_head,   self.seq_read_head,
                self.concept_proj,
            ]
        for m in heads:
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    # -----------------------------------------------------------------------
    # PE helpers
    # -----------------------------------------------------------------------

    @staticmethod
    def _visit_positions(batch_vec: torch.Tensor) -> torch.Tensor:
        """
        Compute 0-based position of each visit within its patient.
        PyG Batch preserves node order (all visits of patient 0 first, etc.),
        so unique_consecutive gives exact per-patient counts.
        """
        _, counts = torch.unique_consecutive(batch_vec, return_counts=True)
        return torch.cat([
            torch.arange(c, device=batch_vec.device) for c in counts
        ])

    # -----------------------------------------------------------------------
    # Forward
    # -----------------------------------------------------------------------

    def forward(
        self, data: HeteroData, task: str = "mort_read"
    ) -> dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        data : HeteroData (single or batched)
        task : 'mort_read'  → per-visit mortality + readmission logits
               'diagnosis'  → per-patient last-visit diagnosis logits [B, num_labels]
        """
        device      = data["visit"].x.device
        raw_visit   = data["visit"].x                 # [N_v, visit_feat_dim]
        num_visits  = raw_visit.size(0)

        batch_vec = (
            data["visit"].batch
            if hasattr(data["visit"], "batch")
            else raw_visit.new_zeros(num_visits, dtype=torch.long)
        )

        # ---- 1. Positional / structural encodings --------------------------
        positions = self._visit_positions(batch_vec)           # [N_v]
        lpe = _sinusoidal_lpe(positions, self.pe_dim)          # [N_v, pe_dim]
        rws = _metapath_rw_se(                                 # [N_v, pe_dim]
            data, num_visits, self.pe_dim, device
        )

        # ---- 2. Build x_dict -----------------------------------------------
        visit_in = torch.cat([raw_visit, lpe, rws], dim=-1)   # [N_v, feat+2*pe]
        x_dict   = {"visit": self.visit_proj(visit_in)}

        for node_type, embed in [
            ("condition", self.cond_embed),
            ("procedure", self.proc_embed),
            ("drug",      self.drug_embed),
        ]:
            if node_type in data.node_types and data[node_type].x.size(0) > 0:
                idx = data[node_type].x[:, 0].long().clamp(
                    0, embed.weight.size(0) - 1
                )
                x_dict[node_type] = F.relu(self.concept_proj(embed(idx)))
            else:
                x_dict[node_type] = x_dict["visit"].new_zeros(
                    (0, self.hidden_dim)
                )

        # ---- 3. Build edge_index_dict for HGTConv --------------------------
        edge_index_dict = self._build_edge_index_dict(data)

        # ---- 4. HGTConv layers ---------------------------------------------
        for conv, norm in zip(self.convs, self.norms):
            out    = conv(x_dict, edge_index_dict)
            v_out  = out.get("visit", x_dict["visit"])
            v_out  = F.dropout(F.relu(v_out), p=self.dropout, training=self.training)
            x_dict["visit"] = norm(v_out + x_dict["visit"])   # residual

        visit_emb = x_dict["visit"]                            # [N_v, H]

        # ---- 5. Pack into dense [B, max_T, H] ------------------------------
        visit_padded, mask = to_dense_batch(visit_emb, batch_vec)
        key_padding_mask   = ~mask                             # True = padded pos

        # ---- 6. Graph branch: Transformer over GNN-enriched visits ---------
        graph_out = self.graph_transformer(
            visit_padded, src_key_padding_mask=key_padding_mask
        )   # [B, T, H]

        # ---- 7. Sequence branch: Transformer over raw visit features -------
        raw_padded, _ = to_dense_batch(
            self.seq_proj(raw_visit), batch_vec
        )   # [B, T, H]
        seq_out = self.seq_transformer(
            raw_padded, src_key_padding_mask=key_padding_mask
        )   # [B, T, H]

        # ---- 8. Dual-branch fusion + prediction heads ----------------------
        B      = graph_out.size(0)
        device = graph_out.device

        if task == "diagnosis":
            # Last non-padded visit per patient → [B, H]
            seq_lens = mask.sum(dim=1) - 1
            idx      = torch.arange(B, device=device)
            g_last   = graph_out[idx, seq_lens]     # [B, H]
            s_last   = seq_out[idx, seq_lens]       # [B, H]

            graph_logits = self.graph_head(g_last)  # [B, num_labels]
            seq_logits   = self.seq_head(s_last)    # [B, num_labels]
            logits = self.alpha * graph_logits + (1 - self.alpha) * seq_logits
            return {"diagnosis": logits}

        else:  # mort_read — per-visit predictions
            g_visits = graph_out[mask]              # [N_v, H]
            s_visits = seq_out[mask]                # [N_v, H]

            graph_mort = self.graph_mort_head(g_visits).squeeze(-1)
            graph_read = self.graph_read_head(g_visits).squeeze(-1)
            seq_mort   = self.seq_mort_head(s_visits).squeeze(-1)
            seq_read   = self.seq_read_head(s_visits).squeeze(-1)

            return {
                "mortality":   self.alpha * graph_mort + (1 - self.alpha) * seq_mort,
                "readmission": self.alpha * graph_read + (1 - self.alpha) * seq_read,
            }

    # -----------------------------------------------------------------------

    @staticmethod
    def _build_edge_index_dict(data: HeteroData) -> dict:
        edge_index_dict = {}

        reverse_map = {
            ("visit", "has_condition", "condition"): ("condition", "r_has_condition", "visit"),
            ("visit", "has_procedure", "procedure"): ("procedure", "r_has_procedure", "visit"),
            ("visit", "has_drug",      "drug"):      ("drug",      "r_has_drug",      "visit"),
        }
        for fwd_key, rev_key in reverse_map.items():
            if fwd_key in data.edge_types:
                ei = data[fwd_key].edge_index
                if ei.size(1) > 0:
                    edge_index_dict[rev_key] = ei.flip(0)

        temporal_key = ("visit", "temporal_next", "visit")
        if temporal_key in data.edge_types:
            ei = data[temporal_key].edge_index
            if ei.size(1) > 0:
                edge_index_dict[temporal_key] = ei

        return edge_index_dict

    # -----------------------------------------------------------------------
    # Factory
    # -----------------------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        # KG paths (required unless no_kg=True)
        concept_vocab_path: Optional[str | Path] = None,
        kg_vocab_path:      Optional[str | Path] = None,
        kg_matrix_path:     Optional[str | Path] = None,
        # Architecture
        hidden_dim:             int   = 128,
        num_gnn_layers:         int   = 2,
        num_transformer_layers: int   = 2,
        nhead:                  int   = 4,
        dropout:                float = 0.3,
        pe_dim:                 int   = 4,
        freeze_kg:              bool  = True,
        visit_feat_dim:         int   = 7,
        alpha:                  float = 0.8,
        # No-KG mode
        no_kg:          bool = False,
        cond_vocab_size: int = 0,
        proc_vocab_size: int = 0,
        drug_vocab_size: int = 0,
        # Task
        num_labels:  int = 0,
        kg_embed_dim: int = 128,
    ) -> "PatientGNN":
        """
        Instantiate PatientGNN.

        KG mode  : provide concept_vocab_path, kg_vocab_path, kg_matrix_path
        No-KG    : set no_kg=True and provide *_vocab_size values
        """
        if no_kg:
            return cls(
                visit_feat_dim=visit_feat_dim,
                hidden_dim=hidden_dim,
                num_gnn_layers=num_gnn_layers,
                num_transformer_layers=num_transformer_layers,
                nhead=nhead,
                dropout=dropout,
                pe_dim=pe_dim,
                kg_embed_dim=kg_embed_dim,
                no_kg=True,
                cond_vocab_size=cond_vocab_size,
                proc_vocab_size=proc_vocab_size,
                drug_vocab_size=drug_vocab_size,
                num_labels=num_labels,
                alpha=alpha,
            )

        # KG mode
        kg_tables    = load_kg_embeddings(concept_vocab_path, kg_vocab_path, kg_matrix_path)
        cond_table, _ = kg_tables["condition"]
        proc_table, _ = kg_tables["procedure"]
        drug_table, _ = kg_tables["drug"]
        actual_kg_dim = cond_table.shape[1] if cond_table.shape[0] > 0 else kg_embed_dim

        return cls(
            visit_feat_dim=visit_feat_dim,
            hidden_dim=hidden_dim,
            num_gnn_layers=num_gnn_layers,
            num_transformer_layers=num_transformer_layers,
            nhead=nhead,
            dropout=dropout,
            pe_dim=pe_dim,
            kg_embed_dim=actual_kg_dim,
            cond_embed_table=cond_table,
            proc_embed_table=proc_table,
            drug_embed_table=drug_table,
            freeze_kg=freeze_kg,
            no_kg=False,
            num_labels=num_labels,
            alpha=alpha,
        )
