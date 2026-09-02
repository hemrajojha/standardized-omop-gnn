# Standardized OMOP GNN — Heterogeneous Temporal GNN on OMOP CDM with Attribution Analysis

> Code repository for the paper:  
> **"Data Standardization Matters: Enhancing GNN-Based Patient Journey Modeling with OMOP CDM and Controlled Clinical Vocabularies"**  

---

## Overview

This repository provides a full end-to-end pipeline for clinical predictive modelling using a heterogeneous temporal graph neural network (GNN) trained on OMOP CDM patient graphs derived from MIMIC-IV. The pipeline spans four stages:

1. **ETL** — MIMIC-IV raw tables → OMOP CDM 5.4.1 (PostgreSQL)
2. **Graph Construction** — OMOP CDM exports → per-patient PyG heterogeneous graphs
3. **Model Training** — PatientGNN (E-TRANS): HGTConv + Transformer + RotatE KG embeddings
4. **Attribution Analysis** — Gradient-based attribution of top clinical concepts per patient, with a Streamlit dashboard for interactive exploration

**Key results:**
- AUROC **0.909** on next-visit diagnosis prediction (275 SNOMED CT labels, MIMIC-IV test set, 20,033 patients)
- Gradient attribution computed for 20,024 test patients across condition, procedure, and drug domains

---

## Repository Structure

```
standardized-omop-gnn/
├── OHDSI_MIMIC/                  # MIMIC-IV → OMOP CDM 5.4.1 ETL pipeline
│   ├── ddl/                      # Schema creation (raw, vocab, OMOP CDM tables)
│   ├── staging/                  # Raw CSV loading and intermediate staging tables
│   ├── etl/                      # CDM table population (21 scripts)
│   │                             #   person, visit, condition, procedure, drug,
│   │                             #   measurement, observation period, era tables
│   ├── export/                   # Export clinical events, KG triples, concept metadata
│   ├── unload/                   # Final unload to OMOP CDM schema
│   ├── custom/                   # Custom concept mappings
│   ├── run/run_pipeline.sh       # End-to-end pipeline runner
│   ├── docker-compose.yml        # PostgreSQL container setup
│   └── .env.example              # DB connection template
├── src/
│   ├── gnn/
│   │   ├── model.py              # PatientGNN: HGTConv + Transformer + dual-branch fusion
│   │   └── train.py              # Training loop (diagnosis + mort_read tasks)
│   ├── preprocess/
│   │   └── build_patient_graphs.py   # OMOP CDM exports → PyG HeteroData graphs
│   └── kg/
│       └── train_rotate.py       # RotatE KG embedding training (PyKEEN)
├── analysis/
│   ├── attribution_app.py        # Streamlit dashboard for attribution results
│   ├── batch_explain.py          # Batch gradient attribution (HPC, 20k+ patients)
│   ├── batch_query.py            # Query attribution JSONs by concept or label
│   ├── explain_patient.py        # Single-patient attribution + enrichment
│   ├── visualize_attribution.py  # Attribution bar plots per patient
│   ├── plot_attribution_examples.py  # Publication figures for attribution
│   ├── query_concepts.py         # OMOP concept lookup utilities
│   ├── analyse_rich_inference.py # Comprehensive inference analysis (calibration, PR, F1)
│   ├── analyse_early_prediction_separate_models.py  # Early prediction benchmarks
│   ├── analyse_early_prediction.py   # Early prediction curves
│   ├── analyse_per_label.py      # Per-label AUROC, TP/FP breakdown
│   ├── analyse_subgroup.py       # Demographic subgroup analysis
│   ├── analyse_cancer_subgroup.py    # Cancer diagnosis subgroup analysis
│   ├── analyse_complexity.py     # Graph complexity vs prediction quality
│   ├── plot_complexity_distribution.py  # Complexity distribution figures
│   └── plot_graph_complexity.py  # Graph complexity scatter plots
├── .env.example                  # Anthropic + environment variable template
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Requirements

- Python 3.10+
- PostgreSQL 14+ (or Docker)
- CUDA-capable GPU (recommended for training and batch attribution)
- MIMIC-IV access via [PhysioNet](https://physionet.org/content/mimiciv/) (credentialed)
- Athena vocabulary download: SNOMED CT, RxNorm, LOINC ([athena.ohdsi.org](https://athena.ohdsi.org/))

Install Python dependencies:

```bash
pip install -r requirements.txt
```

---

## Setup and Usage

### 1. OMOP CDM ETL Pipeline

> Patient-level data cannot be shared due to the MIMIC-IV Data Use Agreement.  
> See [Data Access](#data-access) for how to obtain MIMIC-IV.

The `OHDSI_MIMIC/` directory contains the full SQL pipeline mapping MIMIC-IV to OMOP CDM 5.4.1:

| Stage | Scripts | Description |
|-------|---------|-------------|
| DDL | `ddl/01–05` | Create raw, vocabulary, and OMOP CDM schemas |
| Staging | `staging/00–04` | Load raw CSVs and build intermediate staging tables |
| ETL | `etl/01–21` | Populate all CDM tables |
| Export | `export/01–03` | Export clinical events, KG triples, concept metadata |
| Unload | `unload/01` | Final unload to OMOP CDM schema |

**Start PostgreSQL:**
```bash
cd OHDSI_MIMIC
cp .env.example .env   # fill in DB credentials
docker-compose up -d
```

**Run the full pipeline:**
```bash
bash OHDSI_MIMIC/run/run_pipeline.sh
```

**Download Athena vocabularies** (SNOMED CT, RxNorm, LOINC) from [athena.ohdsi.org](https://athena.ohdsi.org/) and update the path in `OHDSI_MIMIC/staging/01_load_vocab.sql`.

---

### 2. Train RotatE KG Embeddings

Trains RotatE embeddings on OMOP concept triples exported from the CDM. Only concepts appearing in patient graphs are embedded to keep the table size manageable.

```bash
python src/kg/train_rotate.py \
    --triples_path /path/to/omop_export/kg_triples.csv \
    --vocab_path   /path/to/processed/concept_vocab.json \
    --out_dir      /path/to/processed/kg_embeddings \
    --embed_dim    128 \
    --num_epochs   200 \
    --batch_size   4096 \
    --device       cuda
```

**Output:** `entity_embeddings.pt`, `relation_embeddings.pt`, `kg_vocab.json`

---

### 3. Build Patient Graphs

Converts OMOP CDM exports into per-patient PyG `HeteroData` graphs.

```bash
python src/preprocess/build_patient_graphs.py \
    --data_dir /path/to/omop_export \
    --out_dir  /path/to/processed \
    --min_visits 2
```

**Graph schema per patient:**

| Node type | Features | Description |
|-----------|----------|-------------|
| `visit` | `[N_v, F]` | LOS, age, gender, lab stats |
| `condition` | `[N_c, 1]` | Concept embedding index |
| `procedure` | `[N_p, 1]` | Concept embedding index |
| `drug` | `[N_d, 1]` | Concept embedding index |

**Edge types:** `visit→condition`, `visit→procedure`, `visit→drug`, `visit→visit` (temporal)

**Output:** `patient_graphs.pt`, `concept_vocab.json`, `stats.json`

---

### 4. Train the GNN

> **GNN implementation** based on [TRANS](https://github.com/The-Real-JerryChen/TRANS) — official implementation for "Predictive Modeling with Temporal Graphical Representation on Electronic Health Records" (IJCAI 2024). The PatientGNN architecture replicates and extends TRANS with OMOP CDM support and RotatE KG embeddings.

PatientGNN (E-TRANS) supports two tasks:

**Diagnosis prediction (multi-label, 275 SNOMED CT labels):**
```bash
python src/gnn/train.py \
    --task diagnosis \
    --data_source omop \
    --graphs_path /path/to/processed/patient_graphs.pt \
    --vocab_path  /path/to/processed/concept_vocab.json \
    --kg_vocab    /path/to/processed/kg_embeddings/kg_vocab.json \
    --kg_matrix   /path/to/processed/kg_embeddings/entity_embedding_matrix.npy \
    --out_dir     /path/to/models \
    --device      cuda
```

**Mortality + 30-day readmission (binary):**
```bash
python src/gnn/train.py \
    --task mort_read \
    --data_source omop \
    --graphs_path /path/to/processed/patient_graphs.pt \
    --out_dir     /path/to/models \
    --device      cuda
```

**Model architecture:**
- HGTConv (relation-specific attention) over heterogeneous patient graph
- Laplacian PE + MetaPath Random Walk SE on visit nodes
- TransformerEncoder over temporal visit sequence
- Dual-branch fusion (graph + sequence, α = 0.8)
- Pretrained RotatE embeddings for concept nodes (OMOP concept_relationship)

---

### 5. Gradient Attribution

**Single patient:**
```bash
python analysis/explain_patient.py \
    --patient_idx  42 \
    --graphs_path  /path/to/processed/patient_graphs.pt \
    --checkpoint   /path/to/models/best_model.pt \
    --vocab_path   /path/to/processed/concept_vocab.json \
    --label_vocab  /path/to/processed/label_vocab.json \
    --out_dir      analysis/explain
```

**Batch attribution (full test set, HPC):**
```bash
python analysis/batch_explain.py \
    --graphs_path /path/to/processed/patient_graphs.pt \
    --checkpoint  /path/to/models/best_model.pt \
    --vocab_path  /path/to/processed/concept_vocab.json \
    --label_vocab /path/to/processed/label_vocab.json \
    --out_dir     analysis/explain/batch \
    --top_k       10 \
    --device      cuda
```

Outputs one enriched JSON per patient with top-k attributed concepts per domain (condition, procedure, drug).

---

### 6. Attribution Dashboard (Streamlit)

Interactive browser for exploring attribution results across 20,000+ patients:

```bash
streamlit run analysis/attribution_app.py -- --data_dir analysis/explain/batch
```

Features:
- Searchable patient selector with label and probability metadata
- Per-domain attribution bar charts (condition, procedure, drug)
- SNOMED concept names and attribution scores

---

### 7. Analysis Scripts

| Script | Description |
|--------|-------------|
| `analyse_rich_inference.py` | Calibration, PR curves, F1 by demographic group |
| `analyse_early_prediction_separate_models.py` | Early prediction benchmark vs TRANS baseline |
| `analyse_early_prediction.py` | Early prediction curves by visit index |
| `analyse_per_label.py` | Per-label AUROC, TP/FP breakdown, TRANS comparison |
| `analyse_subgroup.py` | Subgroup AUROC by age, gender, ethnicity |
| `analyse_cancer_subgroup.py` | Cancer diagnosis subgroup deep-dive |
| `analyse_complexity.py` | Graph complexity metrics vs prediction quality |
| `plot_attribution_examples.py` | Publication-quality attribution figures |
| `plot_complexity_distribution.py` | Graph complexity distribution plots |
| `plot_graph_complexity.py` | Graph complexity scatter plots |
| `batch_query.py` | Query attribution results by concept or label |
| `query_concepts.py` | OMOP concept lookup and mapping utilities |
| `visualize_attribution.py` | Per-patient attribution visualisation |

---

## Model Performance

| Task | Metric | Score |
|------|--------|-------|
| Diagnosis (275 labels) | Macro AUROC | 0.909 |
| Diagnosis (275 labels) | Micro AUROC | — |
| Mort + Readmission | AUROC | — |

Evaluated on MIMIC-IV test set (20,033 patients).

---

## Data Access

Patient-level data (MIMIC-IV) cannot be shared under the PhysioNet Data Use Agreement. To reproduce:

1. Complete the required training at [CITI Program](https://about.citiprogram.org/)
2. Apply for credentialed access at [physionet.org/content/mimiciv](https://physionet.org/content/mimiciv/)
3. Follow the ETL instructions in [Setup](#1-omop-cdm-etl-pipeline) above

---


## License

Code: MIT License  
Data: Subject to [PhysioNet Credentialed Health Data License](https://physionet.org/content/mimiciv/view-license/) and MIMIC-IV Data Use Agreement
