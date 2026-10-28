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
- Macro AUROC **0.909** on diagnosis prediction (275 SNOMED CT labels) on the MIMIC-IV test set (15,024 patients), best configuration E11 (see [Train the GNN](#3-train-the-gnn))
- Gradient attribution per patient across condition, procedure and drug domains

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
│   ├── batch_explain.py          # Batch gradient attribution over the test set (HPC)
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
├── .env.example                  # Database settings for the analysis scripts
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Requirements

- Python 3.11 (tested with 3.11.15; scikit-learn 1.8 needs 3.11+)
- PostgreSQL 14+ (or Docker)
- CUDA-capable GPU (recommended for training and batch attribution)
- MIMIC-IV **v3.1** access via [PhysioNet](https://physionet.org/content/mimiciv/) (credentialed; 364,627 patients, 546,028 admissions)
- Athena vocabularies ([athena.ohdsi.org](https://athena.ohdsi.org/)), release **v5.0 27-FEB-26** used for the paper. Select at least: SNOMED, ICD9CM, ICD10CM, ICD9Proc, ICD10PCS, RxNorm, RxNorm Extension, NDC, ATC, LOINC, UCUM (plus the default domain vocabularies). CPT4 is optional and needs a UMLS licence.

Install Python dependencies (tested with PyTorch 2.5.1, CUDA 12.4, PyTorch Geometric 2.7.0, PyKEEN 1.11.1). Install PyTorch for your platform first, then the rest:

```bash
# GPU (CUDA 12.4)
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
# or CPU only
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu

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

**Download Athena vocabularies** (see [Requirements](#requirements) for the list) from [athena.ohdsi.org](https://athena.ohdsi.org/).

**Paths to edit before running.** File paths in the SQL are placeholders:

| File | What to change |
|------|----------------|
| `staging/00_load_raw_csv.sql` | `'/path/to/mimiciv/...'` in each `\COPY` line → your MIMIC-IV `hosp/` and `icu/` folders (`\COPY` reads files from the machine running `psql`) |
| `staging/01_load_vocab.sql` | the vocabulary CSV paths → your Athena download folder |
| `export/01–03` | `'C:/omop_export/...'` in each `COPY ... TO` → your export folder. Server-side `COPY` writes on the database server (inside the Docker container) and needs superuser rights; use `\copy` in `psql` to write locally instead |

---

### 2. Build Patient Graphs

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

**Output:** `patient_graphs.pt`, `concept_vocab.json`, `label_vocab.json`, `stats.json`

---

### 2b. Train RotatE KG Embeddings (optional, KG variant only)

Needed only for the KG-embedding variant (E8); the best configuration (E11) does not use it. Requires `concept_vocab.json` from step 2. Trains RotatE embeddings on OMOP concept triples exported from the CDM. Only concepts appearing in patient graphs are embedded to keep the table size manageable.

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

### 3. Train the GNN

> **GNN implementation** based on [TRANS](https://github.com/The-Real-JerryChen/TRANS) — official implementation for "Predictive Modeling with Temporal Graphical Representation on Electronic Health Records" (IJCAI 2024). The PatientGNN architecture replicates and extends TRANS with OMOP CDM support and RotatE KG embeddings.

PatientGNN (E-TRANS) supports two tasks:

**Diagnosis prediction (multi-label, 275 SNOMED CT labels), best configuration E11 (macro AUROC 0.909):**
```bash
python src/gnn/train.py     --task diagnosis     --data_source omop     --graphs_path /path/to/processed/patient_graphs.pt     --vocab_path  /path/to/processed/concept_vocab.json     --no_kg     --pw_mode     sqrt5     --out_dir     /path/to/models     --device      cuda
```

Other configurations from the paper (OMOP data; all other settings default, `--seed 42`, 75/10/15 split):

| Exp | KG embeddings | `--pw_mode` | Flags |
|-----|---------------|-------------|-------|
| E8  | Yes (RotatE)  | `cap50` | `--kg_vocab ... --kg_matrix ...` |
| E9  | No            | `cap50` | `--no_kg` |
| E10 | No            | `cap10` | `--no_kg --pw_mode cap10` |
| E11 | No            | `sqrt5` | `--no_kg --pw_mode sqrt5` |

With KG embeddings (E8):
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
- Optional pretrained RotatE embeddings for concept nodes (OMOP concept_relationship); not used in E11

---

### 4. Gradient Attribution

The attribution scripts build the model without KG embeddings, so use a checkpoint trained with `--no_kg` (E9–E11). `batch_explain.py` uses the same preprocessing and seeded test split as `train.py`.

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

**Batch attribution (full test set):**
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

Outputs one JSON per patient with the top-k attributed concepts per domain (condition, procedure, drug). `analysis/query_concepts.py --attribution <file>` adds concept names and SNOMED ancestors from the OMOP vocabulary (writes `<file>_enriched.json` to the current folder; set `DB_HOST`, `DB_NAME`, `DB_USER`, `DB_PASSWORD`).

---

### 5. Attribution Dashboard (Streamlit)

Interactive browser for exploring attribution results:

```bash
streamlit run analysis/attribution_app.py -- --data_dir analysis/explain/batch
```

Features:
- Searchable patient selector with label and probability metadata
- Per-domain attribution bar charts (condition, procedure, drug)
- SNOMED concept names and attribution scores

---

### 6. Analysis Scripts

The TRANS comparison scripts (`analyse_per_label.py`, `analyse_cancer_subgroup.py`, `analyse_early_prediction_separate_models.py`) and `train.py --data_source pyhealth` also need the TRANS benchmark code, adapted to OMOP, on the Python path. It is not included in this repository.

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
| Diagnosis (275 labels) | Macro AUPRC | 0.336 |

Configuration E11. Evaluated on the MIMIC-IV test set (15,024 patients).

---

## Data Access

Patient-level data (MIMIC-IV) cannot be shared under the PhysioNet Data Use Agreement. To reproduce:

1. Complete the required training at [CITI Program](https://about.citiprogram.org/)
2. Apply for credentialed access at [physionet.org/content/mimiciv](https://physionet.org/content/mimiciv/)
3. Follow the ETL instructions in [Setup](#1-omop-cdm-etl-pipeline) above

---


## License

Data: Subject to [PhysioNet Credentialed Health Data License](https://physionet.org/content/mimiciv/view-license/) and MIMIC-IV Data Use Agreement
