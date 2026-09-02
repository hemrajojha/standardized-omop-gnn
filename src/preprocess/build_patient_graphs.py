"""
build_patient_graphs.py
=======================
Parse exported OMOP CSVs into per-patient PyG HeteroData graphs.

Input files (all in --data_dir):
    clinical_visits.csv
    clinical_conditions.csv
    clinical_procedures.csv
    clinical_drugs.csv
    clinical_measurements.csv          (optional – used as visit-level features)

    kg_concepts.csv                    (for concept_id → global node index lookup)

Output (written to --out_dir):
    patient_graphs.pt                  torch.save(list[HeteroData])
    concept_vocab.json                 {concept_id: int_index} for all domains
    stats.json                         dataset-level statistics

Graph schema per patient
------------------------
Node types:
    visit       [num_visits, F_visit]   F_visit = los_hours + age + gender + lab_stats(n_labs)
    condition   [num_unique_conds, 1]   feature = concept embedding index (filled later)
    procedure   [num_unique_procs, 1]
    drug        [num_unique_drugs, 1]

Edge types  (all undirected → stored as both directions):
    (visit, has_condition,   condition)   visit ←→ all conditions recorded that visit
    (visit, has_procedure,   procedure)
    (visit, has_drug,        drug)
    (visit, temporal_next,   visit)       visit[t] → visit[t+1]  (directed, temporal)

Node-level labels (on visit nodes):
    y_mortality     int  {0, 1}
    y_readmission   int  {0, 1}

Usage
-----
python build_patient_graphs.py \\
    --data_dir /path/to/omop_export \\
    --out_dir  /path/to/processed \\
    --min_visits 2 \\
    --max_patients 0          # 0 = all patients

HPC tip: run with ~32 GB RAM; measurements CSV is large but streamed per-patient.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# OMOP concept_id=0 means "no matching concept" — skip these
NULL_CONCEPT_ID = 0

# Gender concept IDs in standard OMOP vocabulary
GENDER_MALE   = 8507
GENDER_FEMALE = 8532

# Visit concept IDs kept in export (inpatient / ED+inpatient)
INPATIENT_VISIT_TYPES = {9201, 9203, 262}


# ---------------------------------------------------------------------------
# Vocabulary builder
# ---------------------------------------------------------------------------

class ConceptVocab:
    """Maps concept_id (int) → dense index (int) for a given domain."""

    def __init__(self):
        self._map: dict[int, int] = {}

    def __len__(self):
        return len(self._map)

    def get_or_add(self, concept_id: int) -> int:
        if concept_id not in self._map:
            self._map[concept_id] = len(self._map)
        return self._map[concept_id]

    def get(self, concept_id: int) -> int:
        """Return index; -1 if not seen during fit."""
        return self._map.get(concept_id, -1)

    def to_dict(self) -> dict[int, int]:
        return dict(self._map)

    @classmethod
    def from_dict(cls, d: dict) -> "ConceptVocab":
        v = cls()
        v._map = {int(k): int(v_) for k, v_ in d.items()}
        return v


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def load_visits(path: Path) -> pd.DataFrame:
    log.info("Loading clinical_visits.csv …")
    df = pd.read_csv(
        path,
        usecols=[
            "subject_id", "person_id", "hadm_id",
            "admittime", "dischtime",
            "los_hours",
            "year_of_birth", "gender_concept_id",
            "mortality_label", "readmission_30d_label",
            "visit_index",
        ],
        parse_dates=["admittime", "dischtime"],
    )
    df["los_hours"] = pd.to_numeric(df["los_hours"], errors="coerce").fillna(0.0)
    df["mortality_label"]    = df["mortality_label"].fillna(0).astype(int)
    df["readmission_30d_label"] = df["readmission_30d_label"].fillna(0).astype(int)
    df.sort_values(["subject_id", "visit_index"], inplace=True)
    log.info("  %d visit rows for %d patients", len(df), df["subject_id"].nunique())
    return df


def load_events(path: Path, concept_col: str, extra_cols: list[str] | None = None) -> pd.DataFrame:
    cols = ["subject_id", "hadm_id", concept_col] + (extra_cols or [])
    log.info("Loading %s …", path.name)
    df = pd.read_csv(path, usecols=cols)
    df[concept_col] = pd.to_numeric(df[concept_col], errors="coerce").fillna(0).astype(int)
    df = df[df[concept_col] != NULL_CONCEPT_ID]
    log.info("  %d rows", len(df))
    return df


def compute_label_vocab(conds_df: pd.DataFrame, top_k: int) -> list[int]:
    """
    Return an ordered list of condition concept IDs to use as diagnosis labels.
    Ordered by frequency (most common first); index in list = label index.
    top_k=0 means keep all.
    """
    from collections import Counter
    freq = Counter(conds_df["condition_concept_id"].tolist())
    if top_k > 0:
        return [c for c, _ in freq.most_common(top_k)]
    return list(freq.keys())


def load_measurements_aggregated(path: Path) -> dict[str, pd.DataFrame]:
    """
    Stream clinical_measurements.csv and compute per-(hadm_id) summary statistics.
    Returns a dict keyed by hadm_id (as str) → DataFrame with columns:
        n_labs, mean_value, std_value, pct_abnormal
    """
    log.info("Loading clinical_measurements.csv (streaming) …")
    chunks = pd.read_csv(
        path,
        usecols=["hadm_id", "value_as_number", "abnormal_flag"],
        chunksize=500_000,
    )
    rows = []
    for chunk in chunks:
        chunk["is_abnormal"] = (chunk["abnormal_flag"].isin(["LOW", "HIGH"])).astype(float)
        grp = chunk.groupby("hadm_id")["value_as_number"]
        agg = grp.agg(n_labs="count", mean_value="mean", std_value="std")
        abn = chunk.groupby("hadm_id")["is_abnormal"].mean().rename("pct_abnormal")
        rows.append(pd.concat([agg, abn], axis=1))
    if not rows:
        return {}
    result = pd.concat(rows).groupby(level=0).mean()
    result["n_labs"] = result["n_labs"].round().astype(int)
    result["std_value"] = result["std_value"].fillna(0.0)
    log.info("  Aggregated lab stats for %d visits", len(result))
    return result


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_graph(
    patient_visits: pd.DataFrame,
    cond_rows: pd.DataFrame,
    proc_rows: pd.DataFrame,
    drug_rows: pd.DataFrame,
    lab_agg: pd.DataFrame,
    cond_vocab: ConceptVocab,
    proc_vocab: ConceptVocab,
    drug_vocab: ConceptVocab,
    label_vocab: list[int] | None = None,
    last_hadm_id: int | None = None,
) -> HeteroData:
    """Build one HeteroData graph for a single patient."""

    data = HeteroData()

    # ---- visit node features -----------------------------------------------
    # Columns: los_hours, age (derived from year_of_birth), gender_flag,
    #          n_labs, mean_value, std_value, pct_abnormal
    visit_ids = patient_visits["hadm_id"].tolist()
    hadm_to_vidx = {hid: i for i, hid in enumerate(visit_ids)}
    n_visits = len(visit_ids)

    # Approximate age at admission
    admit_year = pd.to_datetime(patient_visits["admittime"]).dt.year
    yob = patient_visits["year_of_birth"].astype(float)
    age = (admit_year - yob).clip(0, 120).fillna(50).values

    gender_flag = (
        patient_visits["gender_concept_id"] == GENDER_FEMALE
    ).astype(float).values

    los = patient_visits["los_hours"].values.astype(float)

    # Lab summary features (per visit)
    if lab_agg is not None and len(lab_agg) > 0:
        def _lab_feat(hadm_id):
            if hadm_id in lab_agg.index:
                r = lab_agg.loc[hadm_id]
                return [float(r["n_labs"]), float(r["mean_value"]),
                        float(r["std_value"]), float(r["pct_abnormal"])]
            return [0.0, 0.0, 0.0, 0.0]
        lab_feats = np.array([_lab_feat(h) for h in visit_ids], dtype=np.float32)
    else:
        lab_feats = np.zeros((n_visits, 4), dtype=np.float32)

    # Normalise los (log scale)
    los_norm = np.log1p(np.clip(los, 0, None)).astype(np.float32).reshape(-1, 1)
    age_norm  = (age / 100.0).astype(np.float32).reshape(-1, 1)
    gender_f  = gender_flag.astype(np.float32).reshape(-1, 1)

    visit_x = np.concatenate([los_norm, age_norm, gender_f, lab_feats], axis=1)
    data["visit"].x = torch.tensor(visit_x, dtype=torch.float)

    # Labels
    data["visit"].y_mortality    = torch.tensor(
        patient_visits["mortality_label"].values, dtype=torch.long
    )
    data["visit"].y_readmission  = torch.tensor(
        patient_visits["readmission_30d_label"].values, dtype=torch.long
    )
    # Store hadm_id list for reference
    data["visit"].hadm_ids = torch.tensor(visit_ids, dtype=torch.long)

    # ---- temporal visit → visit edges --------------------------------------
    if n_visits > 1:
        src = list(range(n_visits - 1))
        dst = list(range(1, n_visits))
        data["visit", "temporal_next", "visit"].edge_index = torch.tensor(
            [src, dst], dtype=torch.long
        )
    else:
        data["visit", "temporal_next", "visit"].edge_index = torch.zeros(
            (2, 0), dtype=torch.long
        )

    # ---- helper: build visit ←→ concept edges ------------------------------
    def _concept_edges(event_df: pd.DataFrame, concept_col: str, vocab: ConceptVocab):
        """
        Returns:
            node_feat  [num_unique_concepts, 1]  (vocab index as float for embedding lookup)
            edge_src   [E]   visit indices
            edge_dst   [E]   concept (local) indices
        """
        if event_df.empty:
            return (
                torch.zeros((0, 1), dtype=torch.float),
                torch.zeros((2, 0), dtype=torch.long),
            )

        # Only keep rows whose hadm_id is in this patient's visit list
        ev = event_df[event_df["hadm_id"].isin(hadm_to_vidx)]
        if ev.empty:
            return (
                torch.zeros((0, 1), dtype=torch.float),
                torch.zeros((2, 0), dtype=torch.long),
            )

        # Deduplicate concept IDs → local concept node indices
        unique_concepts = ev[concept_col].unique()
        concept_to_local = {cid: i for i, cid in enumerate(unique_concepts)}

        # Build edge list  (visit_idx → local_concept_idx)
        src_list, dst_list = [], []
        for _, row in ev.iterrows():
            v_idx = hadm_to_vidx[row["hadm_id"]]
            c_idx = concept_to_local[row[concept_col]]
            src_list.append(v_idx)
            dst_list.append(c_idx)

        # Node feature: global vocab index (for embedding lookup in model)
        global_indices = [vocab.get_or_add(cid) for cid in unique_concepts]
        node_x = torch.tensor(global_indices, dtype=torch.float).unsqueeze(1)

        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
        return node_x, edge_index

    # ---- condition nodes + edges -------------------------------------------
    cond_x, cond_edge = _concept_edges(cond_rows, "condition_concept_id", cond_vocab)
    data["condition"].x = cond_x
    data["visit", "has_condition", "condition"].edge_index = cond_edge

    # ---- procedure nodes + edges -------------------------------------------
    proc_x, proc_edge = _concept_edges(proc_rows, "procedure_concept_id", proc_vocab)
    data["procedure"].x = proc_x
    data["visit", "has_procedure", "procedure"].edge_index = proc_edge

    # ---- drug nodes + edges ------------------------------------------------
    drug_x, drug_edge = _concept_edges(drug_rows, "drug_concept_id", drug_vocab)
    data["drug"].x = drug_x
    data["visit", "has_drug", "drug"].edge_index = drug_edge

    # ---- diagnosis label (multi-hot of last visit's top-K conditions) -------
    if label_vocab is not None and last_hadm_id is not None:
        last_conds = set(
            cond_rows[cond_rows["hadm_id"] == last_hadm_id]["condition_concept_id"].tolist()
        )
        code_to_idx = {c: i for i, c in enumerate(label_vocab)}
        y = torch.zeros(len(label_vocab), dtype=torch.float)
        for c in last_conds:
            if c in code_to_idx:
                y[code_to_idx[c]] = 1.0
        data.y_diagnosis = y

    return data


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Build patient HeteroData graphs from OMOP CSVs")
    p.add_argument("--data_dir",    required=True,  help="Directory containing exported OMOP CSVs")
    p.add_argument("--out_dir",     required=True,  help="Output directory for processed files")
    p.add_argument("--min_visits",  type=int, default=2,
                   help="Minimum visits per patient to include (default: 2)")
    p.add_argument("--max_patients", type=int, default=0,
                   help="Limit patients for debugging (0 = all)")
    p.add_argument("--no_measurements", action="store_true",
                   help="Skip loading measurements CSV (faster, less features)")
    p.add_argument("--top_k", type=int, default=275,
                   help="Top-K most frequent SNOMED condition codes for y_diagnosis labels "
                        "(matches E7 top-275 methodology; 0 = keep all)")
    return p.parse_args()


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load all CSVs --------------------------------------------------------
    visits = load_visits(data_dir / "clinical_visits.csv")
    conds  = load_events(data_dir / "clinical_conditions.csv", "condition_concept_id")
    procs  = load_events(data_dir / "clinical_procedures.csv", "procedure_concept_id")
    drugs  = load_events(data_dir / "clinical_drugs.csv",      "drug_concept_id")

    if args.no_measurements:
        lab_agg_global = pd.DataFrame()
    else:
        lab_agg_global = load_measurements_aggregated(data_dir / "clinical_measurements.csv")

    # ---- Compute label vocabulary (top-K condition codes) ---------------------
    if args.top_k != 0:
        log.info("Computing top-%d condition label vocabulary …", args.top_k)
        label_vocab = compute_label_vocab(conds, args.top_k)
        log.info("  Label vocab size: %d (from %d unique codes)",
                 len(label_vocab), conds["condition_concept_id"].nunique())
    else:
        label_vocab = None
        log.info("top_k=0: y_diagnosis labels disabled")

    # ---- Filter cohort --------------------------------------------------------
    visit_counts = visits.groupby("subject_id")["hadm_id"].nunique()
    valid_subjects = visit_counts[visit_counts >= args.min_visits].index
    visits = visits[visits["subject_id"].isin(valid_subjects)]
    log.info("Cohort after min_visits=%d filter: %d patients", args.min_visits, len(valid_subjects))

    subject_list = sorted(visits["subject_id"].unique())
    if args.max_patients > 0:
        subject_list = subject_list[: args.max_patients]
        log.info("Limiting to %d patients (debug mode)", args.max_patients)

    # Pre-index event dataframes by subject_id for fast lookup
    log.info("Indexing event tables by subject_id …")
    conds_by_subj = {sid: grp for sid, grp in conds.groupby("subject_id")}
    procs_by_subj = {sid: grp for sid, grp in procs.groupby("subject_id")}
    drugs_by_subj = {sid: grp for sid, grp in drugs.groupby("subject_id")}
    visits_by_subj = {sid: grp for sid, grp in visits.groupby("subject_id")}

    # ---- Shared vocabularies --------------------------------------------------
    cond_vocab = ConceptVocab()
    proc_vocab = ConceptVocab()
    drug_vocab = ConceptVocab()

    # ---- Build graphs ---------------------------------------------------------
    graphs: list[HeteroData] = []
    skipped = 0

    for sid in tqdm(subject_list, desc="Building graphs", unit="patient"):
        pv = visits_by_subj.get(sid)
        if pv is None or len(pv) < args.min_visits:
            skipped += 1
            continue

        pc = conds_by_subj.get(sid, pd.DataFrame(columns=["subject_id", "hadm_id", "condition_concept_id"]))
        pp = procs_by_subj.get(sid, pd.DataFrame(columns=["subject_id", "hadm_id", "procedure_concept_id"]))
        pd_ = drugs_by_subj.get(sid, pd.DataFrame(columns=["subject_id", "hadm_id", "drug_concept_id"]))

        last_hadm_id = int(pv.iloc[-1]["hadm_id"])

        try:
            g = build_graph(
                patient_visits=pv,
                cond_rows=pc,
                proc_rows=pp,
                drug_rows=pd_,
                lab_agg=lab_agg_global,
                cond_vocab=cond_vocab,
                proc_vocab=proc_vocab,
                drug_vocab=drug_vocab,
                label_vocab=label_vocab,
                last_hadm_id=last_hadm_id,
            )
            # Tag graph with subject_id for traceability
            g.subject_id = int(sid)
            graphs.append(g)
        except Exception as e:
            log.warning("Failed to build graph for subject %s: %s", sid, e)
            skipped += 1

    log.info("Built %d graphs, skipped %d", len(graphs), skipped)

    # ---- Save outputs ---------------------------------------------------------
    out_graphs = out_dir / "patient_graphs.pt"
    torch.save(graphs, out_graphs)
    log.info("Saved %s", out_graphs)

    # Save label vocabulary (condition concept IDs in label-index order)
    if label_vocab is not None:
        out_label_vocab = out_dir / "label_vocab.json"
        with open(out_label_vocab, "w") as f:
            json.dump([str(c) for c in label_vocab], f)
        log.info("Saved %s  (%d labels)", out_label_vocab, len(label_vocab))

    # Save vocabularies (needed by model to look up KG embedding indices later)
    vocab_data = {
        "condition": cond_vocab.to_dict(),
        "procedure": proc_vocab.to_dict(),
        "drug":      drug_vocab.to_dict(),
    }
    out_vocab = out_dir / "concept_vocab.json"
    with open(out_vocab, "w") as f:
        json.dump({k: {str(ck): cv for ck, cv in v.items()} for k, v in vocab_data.items()}, f)
    log.info("Saved %s", out_vocab)

    # ---- Dataset statistics ---------------------------------------------------
    n_visits_list   = [g["visit"].x.size(0) for g in graphs]
    n_conds_list    = [g["condition"].x.size(0) for g in graphs]
    n_procs_list    = [g["procedure"].x.size(0) for g in graphs]
    n_drugs_list    = [g["drug"].x.size(0) for g in graphs]
    mort_labels     = torch.cat([g["visit"].y_mortality for g in graphs])
    read_labels     = torch.cat([g["visit"].y_readmission for g in graphs])

    stats = {
        "n_patients":           len(graphs),
        "n_visits_total":       int(sum(n_visits_list)),
        "visit_feat_dim":       int(graphs[0]["visit"].x.size(1)) if graphs else 0,
        "avg_visits_per_patient": float(np.mean(n_visits_list)),
        "avg_conditions":       float(np.mean(n_conds_list)),
        "avg_procedures":       float(np.mean(n_procs_list)),
        "avg_drugs":            float(np.mean(n_drugs_list)),
        "vocab_size_condition": len(cond_vocab),
        "vocab_size_procedure": len(proc_vocab),
        "vocab_size_drug":      len(drug_vocab),
        "mortality_pos_rate":   float(mort_labels.float().mean()),
        "readmission_pos_rate": float(read_labels.float().mean()),
        "num_diagnosis_labels": len(label_vocab) if label_vocab is not None else 0,
    }
    out_stats = out_dir / "stats.json"
    with open(out_stats, "w") as f:
        json.dump(stats, f, indent=2)

    log.info("Dataset statistics:")
    for k, v in stats.items():
        log.info("  %-35s %s", k, v)
    log.info("Done.")


if __name__ == "__main__":
    main()
