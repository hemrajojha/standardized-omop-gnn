#!/usr/bin/env bash
# =======================================================================
# run_pipeline.sh
# Full OHDSI MIMIC-IV → OMOP CDM pipeline orchestration.
#
# Usage:
#   cd OHDSI_MIMIC
#   cp ../.env.example .env          # adjust paths and credentials
#   source .env
#   bash run/run_pipeline.sh
#
# Or pass DATABASE_URL directly:
#   DATABASE_URL=postgresql://user:pass@host:5432/db bash run/run_pipeline.sh
# =======================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

# Load .env if present
if [[ -f "$ROOT_DIR/.env" ]]; then
    # shellcheck disable=SC1091
    set -a; source "$ROOT_DIR/.env"; set +a
fi

: "${DATABASE_URL:?DATABASE_URL must be set (e.g. postgresql://user:pass@host:5432/db)}"

# psql wrapper – prints step, times it
run_sql() {
    local label="$1"
    local file="$2"
    echo ""
    echo "────────────────────────────────────────────────────────"
    echo "  STEP: $label"
    echo "  FILE: $file"
    echo "────────────────────────────────────────────────────────"
    time psql "$DATABASE_URL" -v ON_ERROR_STOP=1 -f "$file"
}

echo "========================================================"
echo "  OHDSI MIMIC-IV → OMOP CDM 5.4.1 Pipeline"
echo "  Started: $(date)"
echo "========================================================"

# -----------------------------------------------------------------------
# DDL
# -----------------------------------------------------------------------
run_sql "Create schemas + helper function"   "$ROOT_DIR/ddl/01_create_schemas.sql"
run_sql "Raw MIMIC-IV table DDL"             "$ROOT_DIR/ddl/02_raw_mimic_tables.sql"
run_sql "Vocabulary table DDL"               "$ROOT_DIR/ddl/03_vocab_tables.sql"
run_sql "OMOP CDM 5.4.1 table DDL"          "$ROOT_DIR/ddl/04_omop_cdm_tables.sql"

# -----------------------------------------------------------------------
# Load raw data
# -----------------------------------------------------------------------
run_sql "Load MIMIC-IV CSVs"                 "$ROOT_DIR/staging/00_load_raw_csv.sql"
run_sql "Load OHDSI Athena vocabulary"       "$ROOT_DIR/staging/01_load_vocab.sql"
run_sql "Custom concept mappings"            "$ROOT_DIR/custom/custom_mapping.sql"

# -----------------------------------------------------------------------
# Source snapshots (staging src_* tables)
# -----------------------------------------------------------------------
run_sql "Snapshot: core (patients/admissions/transfers)" "$ROOT_DIR/staging/02_st_core.sql"
run_sql "Snapshot: hosp tables"              "$ROOT_DIR/staging/03_st_hosp.sql"
run_sql "Snapshot: ICU tables"               "$ROOT_DIR/staging/04_st_icu.sql"

# -----------------------------------------------------------------------
# ETL (run in numbered order)
# -----------------------------------------------------------------------
for etl_file in "$ROOT_DIR"/etl/*.sql; do
    label=$(basename "$etl_file" .sql)
    run_sql "ETL: $label" "$etl_file"
done

# -----------------------------------------------------------------------
# Unload to final OMOP CDM schema
# -----------------------------------------------------------------------
run_sql "Unload to omopcdm schema"           "$ROOT_DIR/unload/01_unload_to_omopcdm.sql"

echo ""
echo "========================================================"
echo "  Pipeline complete: $(date)"
echo "  OMOP CDM data is in schema: omopcdm"
echo "  Connect ATLAS WebAPI to:"
echo "    cdm_schema   = omopcdm"
echo "    vocab_schema = omopcdm"
echo "    results_schema = results"
echo "========================================================"
