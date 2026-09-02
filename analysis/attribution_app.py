"""
attribution_app.py
==================
Streamlit dashboard for browsing patient-level GNN attribution results.
Optimised for large cohorts (20,000+ patients):
  - Startup loads only lightweight metadata (no concepts dict)
  - Full JSON loaded only when a patient is selected
  - Searchable selectbox replaces radio list

Run:
    streamlit run analysis/attribution_app.py -- --data_dir analysis/explain/batch
"""

import argparse
import json
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import streamlit as st

# ── CLI args ──────────────────────────────────────────────────────────────────
_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument("--data_dir", default=None)
_cli_args, _ = _parser.parse_known_args()

# ── Colours ───────────────────────────────────────────────────────────────────
DOMAIN_COLORS = {
    "condition": "#e07b39",
    "procedure": "#3a7abf",
    "drug":      "#4aab6d",
}
LABEL_COLOR = "#9b59b6"


# ── Data loading (two-stage) ──────────────────────────────────────────────────

@st.cache_data(show_spinner="Scanning patient files...")
def load_metadata(data_dir: str) -> dict:
    """
    Fast startup: read only top-level scalar fields from each enriched JSON.
    Returns {patient_idx (int): {label_name, label_snomed, probability,
                                  n_conditions, n_procedures, n_drugs, path}}
    """
    p     = Path(data_dir)
    files = sorted(p.glob("*_enriched.json"))
    meta  = {}
    for f in files:
        try:
            with open(f) as fh:
                d = json.load(fh)
            pid = int(d["patient_idx"])
            meta[pid] = {
                "label_name":   d.get("label_name") or d.get("label_snomed", "?"),
                "label_snomed": d.get("label_snomed", "?"),
                "probability":  d.get("probability", 0),
                "n_conditions": d.get("n_conditions", "?"),
                "n_procedures": d.get("n_procedures", "?"),
                "n_drugs":      d.get("n_drugs", "?"),
                "path":         str(f),
            }
        except Exception:
            pass
    return meta


@st.cache_data(show_spinner="Loading patient data...")
def load_patient(path: str) -> dict:
    """Load full enriched JSON for one patient (cached per path)."""
    with open(path) as f:
        return json.load(f)


def find_data_dir() -> str:
    if _cli_args.data_dir:
        return _cli_args.data_dir
    for c in [Path("analysis/explain/batch"), Path("analysis/explain"), Path(".")]:
        if c.exists() and list(c.glob("*_enriched.json")):
            return str(c)
    return "analysis/explain/batch"


# ── DB helpers (optional — only needed for sibling lookup) ────────────────────

def _open_db(host, dbname, user, password, port):
    """Open a psycopg2 connection; return None on failure (psycopg2 not installed
    or connection refused). Errors are surfaced to the caller."""
    import psycopg2
    return psycopg2.connect(
        host=host, dbname=dbname, user=user, password=password, port=port
    )


@st.cache_data(show_spinner="Fetching sibling concepts...", ttl=3600)
def fetch_siblings(
    concept_ids: tuple,        # hashable for cache key
    db_host: str, db_name: str, db_user: str, db_pass: str, db_port: int,
    limit_per: int = 3,
) -> dict:
    """
    For each concept_id return up to limit_per sibling concepts —
    i.e. other direct children of the same level-1 SNOMED/RxNorm parent.

    Returns {concept_id (int): [{"concept_id": int, "name": str}]}
    Excludes any concept that is itself in concept_ids.
    """
    try:
        conn = _open_db(db_host, db_name, db_user, db_pass, db_port)
    except Exception as exc:
        st.warning(f"DB connection failed — siblings unavailable: {exc}")
        return {}

    already = set(concept_ids)
    result  = {}
    cur     = conn.cursor()
    for cid in concept_ids:
        try:
            cur.execute("""
                SELECT DISTINCT c.concept_id, c.concept_name
                FROM omopcdm.concept_ancestor ca_up
                JOIN omopcdm.concept_ancestor ca_down
                    ON  ca_down.ancestor_concept_id = ca_up.ancestor_concept_id
                    AND ca_down.min_levels_of_separation = 1
                JOIN omopcdm.concept c
                    ON  c.concept_id = ca_down.descendant_concept_id
                WHERE ca_up.descendant_concept_id = %s
                  AND ca_up.min_levels_of_separation  = 1
                  AND ca_down.descendant_concept_id  != %s
                LIMIT %s
            """, (cid, cid, limit_per + 5))
            rows = cur.fetchall()
            sibs = [{"concept_id": int(r[0]), "name": r[1]}
                    for r in rows if int(r[0]) not in already][:limit_per]
            if sibs:
                result[cid] = sibs
        except Exception:
            pass
    cur.close()
    conn.close()
    return result


# ── Plot helpers ──────────────────────────────────────────────────────────────

def short(name, n=45):
    return name if len(name) <= n else name[:n-3] + "..."

def wrap(text, w=18):
    return "\n".join(textwrap.wrap(text, w))


def fig_bars(data: dict, top_k: int = 8) -> plt.Figure:
    concepts = data.get("concepts", {})
    domains  = [d for d in ("condition", "procedure", "drug")
                if data["attributions"].get(d)]
    if not domains:
        fig, ax = plt.subplots(figsize=(5, 2))
        ax.text(0.5, 0.5, "No attributions", ha="center", va="center", fontsize=13)
        ax.axis("off")
        return fig

    fig, axes = plt.subplots(1, len(domains),
                             figsize=(5.5 * len(domains), 4.5),
                             constrained_layout=True)
    if len(domains) == 1:
        axes = [axes]

    for ax, domain in zip(axes, domains):
        entries = data["attributions"][domain][:top_k]
        color   = DOMAIN_COLORS[domain]
        names, scores = [], []
        for e in entries:
            cid  = e["concept_id"]
            info = concepts.get(str(cid), {})
            names.append(short(info.get("name", f"cid={cid}"), 40))
            scores.append(e["score"])

        y    = np.arange(len(names))
        bars = ax.barh(y, scores, color=color, alpha=0.78,
                       edgecolor=color, linewidth=0.8, height=0.65)
        for bar, sc in zip(bars, scores):
            ax.text(bar.get_width() + max(scores)*0.01,
                    bar.get_y() + bar.get_height()/2,
                    f"{sc:.4f}", va="center", ha="left", fontsize=11)
        ax.set_yticks(y)
        ax.set_yticklabels(names, fontsize=11)
        ax.invert_yaxis()
        ax.set_xlabel("Attribution score", fontsize=12)
        ax.set_title(domain.upper(), fontsize=13, fontweight="bold", color=color)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlim(0, max(scores)*1.30)
        ax.tick_params(axis="x", labelsize=10)
    return fig


def abbrev(name: str, max_words: int = 3, max_chars: int = 22) -> str:
    words = name.split()
    s = " ".join(words[:max_words])
    return s if len(s) <= max_chars else s[:max_chars - 1] + "."


def fig_network(data: dict, top_k: int = 3) -> plt.Figure:
    """Star network with manual polar layout — labels on outer ring, no overlap."""
    concepts = data.get("concepts", {})
    fig, ax  = plt.subplots(figsize=(8, 8))

    all_entries = []
    for domain in ("condition", "procedure", "drug"):
        for e in data["attributions"].get(domain, [])[:top_k]:
            all_entries.append((domain, e))

    if not all_entries:
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes, fontsize=13)
        ax.axis("off")
        return fig

    scores  = [e["score"] for _, e in all_entries]
    s_min, s_max = min(scores), max(scores)
    def norm_size(s):
        return 500 + (s - s_min) / max(s_max - s_min, 1e-9) * 1500

    n_outer = len(all_entries)
    R_node  = 1.0
    R_label = 1.55
    cx, cy  = 0.0, 0.0
    angles  = [2 * np.pi * i / n_outer - np.pi / 2 for i in range(n_outer)]

    # Draw edges
    for i, (domain, e) in enumerate(all_entries):
        x0 = cx + R_node * np.cos(angles[i])
        y0 = cy + R_node * np.sin(angles[i])
        ax.annotate("", xy=(cx, cy), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", color="#bbbbbb",
                                   lw=0.9, alpha=0.5))

    # Draw concept nodes
    for i, (domain, e) in enumerate(all_entries):
        x = cx + R_node * np.cos(angles[i])
        y = cy + R_node * np.sin(angles[i])
        ax.scatter(x, y, s=norm_size(e["score"]),
                   c=DOMAIN_COLORS[domain], alpha=0.88,
                   zorder=3, linewidths=0.5, edgecolors="white")

    # Draw centre node
    ax.scatter(cx, cy, s=2400, c=LABEL_COLOR, alpha=0.88,
               zorder=3, linewidths=0.5, edgecolors="white")
    ax.text(cx, cy,
            f"Predicted\n{abbrev(data.get('label_name','?'), 3, 20)}\np={data['probability']:.2f}",
            ha="center", va="center", fontsize=9,
            color="#111111", fontweight="normal",
            multialignment="center", zorder=4)

    # Draw concept labels on outer ring — full names, wrapped at 22 chars
    for i, (domain, e) in enumerate(all_entries):
        cid  = e["concept_id"]
        info = concepts.get(str(cid), {})
        name = info.get("name", f"cid={cid}")
        name = "\n".join(textwrap.wrap(name, 22))
        a    = angles[i]
        lx   = cx + R_label * np.cos(a)
        ly   = cy + R_label * np.sin(a)
        ha   = "left"   if np.cos(a) >  0.15 else \
               "right"  if np.cos(a) < -0.15 else "center"
        va   = "bottom" if np.sin(a) >  0.15 else \
               "top"    if np.sin(a) < -0.15 else "center"
        ax.text(lx, ly, name, ha=ha, va=va,
                fontsize=7, color="#111111", fontweight="normal",
                multialignment=ha, linespacing=1.3,
                bbox=dict(boxstyle="round,pad=0.2", fc="white",
                          ec="none", alpha=0.85),
                zorder=5)

    margin = R_label + 1.1
    ax.set_xlim(cx - margin, cx + margin)
    ax.set_ylim(cy - margin, cy + margin)
    ax.set_aspect("equal")
    ax.axis("off")

    patches = [mpatches.Patch(color=DOMAIN_COLORS[d], label=d.capitalize())
               for d in ("condition", "procedure", "drug")]
    patches.append(mpatches.Patch(color=LABEL_COLOR, label="Predicted label"))
    ax.legend(handles=patches, loc="lower right", fontsize=8,
              framealpha=0.9, edgecolor="#cccccc")
    plt.tight_layout()
    return fig


def fig_journey(data: dict, max_visits: int = 0) -> plt.Figure:
    """
    Patient journey map: swim-lane timeline across visits.
    Conditions / Procedures / Drugs as horizontal lanes.

    Prior visits: observed concepts (filled circles). Attributed ones are
    coloured + sized by score; unattributed are small grey dots.

    Current visit: condition edges are removed during preprocessing to prevent
    label leakage, so conditions from the attribution scores are shown instead
    as "predicted" (diamond markers with black border). Procedures and drugs
    at the current visit are observed normally.

    max_visits=0 means show all; otherwise show first + last (max_visits-1).
    """
    visits = data.get("visits", [])
    if not visits:
        fig, ax = plt.subplots(figsize=(7, 3))
        ax.text(0.5, 0.5, "No visit timeline data.\nRe-run batch_query.py with --overwrite.",
                ha="center", va="center", fontsize=12, color="#888888",
                transform=ax.transAxes)
        ax.axis("off")
        return fig

    # Subset visits
    if max_visits and len(visits) > max_visits:
        shown = [visits[0]] + visits[-(max_visits - 1):]
        gap_marker = True
    else:
        shown = visits
        gap_marker = False

    domain_key_map = {"condition": "conditions",
                      "procedure": "procedures",
                      "drug":      "drugs"}

    # Top-K attributed concepts per domain — same as attribution bars tab
    # {domain_key: {concept_id (int): score}}
    concepts_lookup = data.get("concepts", {})
    top_k_scores  = {}  # for lookup in prior visits
    predicted     = {}  # sorted list for current visit diamonds
    for attr_domain, domain_key in domain_key_map.items():
        entries = data.get("attributions", {}).get(attr_domain, [])
        top_k_scores[domain_key] = {int(e["concept_id"]): e["score"] for e in entries}
        predicted[domain_key] = sorted(
            [{"concept_id": int(e["concept_id"]),
              "name": concepts_lookup.get(str(e["concept_id"]), {}).get(
                  "name", str(e["concept_id"])),
              "score": e["score"]}
             for e in entries],
            key=lambda x: x["score"], reverse=True
        )

    max_score = max(
        (s for ds in top_k_scores.values() for s in ds.values()),
        default=1.0
    )

    LANES   = {"conditions": 2.0, "procedures": 1.0, "drugs": 0.0}
    LCOLORS = {"conditions": DOMAIN_COLORS["condition"],
               "procedures": DOMAIN_COLORS["procedure"],
               "drugs":      DOMAIN_COLORS["drug"]}
    n_v     = len(shown)
    fig_w   = max(10, n_v * 1.6 + 2)
    fig, ax = plt.subplots(figsize=(fig_w, 5))

    # Lane baselines
    for ly in LANES.values():
        ax.axhline(ly, color="#e0e0e0", lw=1.2, zorder=0)

    # Gap marker between first and rest
    if gap_marker:
        ax.axvline(0.5, color="#cccccc", lw=1.5, ls=":", zorder=1)
        ax.text(0.5, 2.55, "···", ha="center", va="bottom",
                fontsize=14, color="#aaaaaa")

    x_positions = list(range(n_v))
    x_labels    = []

    for xi, visit in enumerate(shown):
        date       = visit.get("visit_date", f"V{xi+1}")[:10]
        vtype      = visit.get("visit_type", "")
        is_current = visit.get("is_current", False)
        x_labels.append(f"{date}\n{vtype[:18]}" if vtype else date)

        # Highlight current visit column
        if is_current:
            ax.axvspan(xi - 0.42, xi + 0.42, color="#9b59b6", alpha=0.07, zorder=0)
            ax.axvline(xi, color="#9b59b6", lw=1.2, ls="--", alpha=0.5, zorder=1)
            ax.text(xi, 2.62, "Current visit\n(see attribution bars)", ha="center", va="bottom",
                    fontsize=8, color="#9b59b6", fontweight="bold")

        for domain_key, ly in LANES.items():
            color = LCOLORS[domain_key]

            # ── Current visit: leave blank (condition edges removed for label
            # leakage prevention; visit-concept mapping fix requires re-run)
            if is_current:
                continue

            # ── Prior visits: show top-K concepts that occurred in this visit ──
            # Same top-K as attribution bars — coloured if in top-K, grey otherwise
            domain_topk       = top_k_scores.get(domain_key, {})
            concepts_in_visit = visit.get(domain_key, [])

            topk_here  = [(c, domain_topk[c["concept_id"]])
                          for c in concepts_in_visit
                          if c["concept_id"] in domain_topk]
            other_here = [c for c in concepts_in_visit
                          if c["concept_id"] not in domain_topk]

            # Non-top-K: small grey dots
            for i, c in enumerate(other_here):
                ax.scatter(xi, ly + 0.04 * i, s=14, c="#cccccc",
                           alpha=0.5, zorder=2, linewidths=0, marker="o")

            # Top-K: coloured + sized by attribution score, labelled
            for rank, (c, score) in enumerate(
                    sorted(topk_here, key=lambda x: x[1], reverse=True)):
                norm  = score / max_score
                size  = 80 + norm * 380
                alpha = 0.45 + norm * 0.50
                ax.scatter(xi, ly, s=size, c=color, alpha=alpha,
                           zorder=4, linewidths=0.6, edgecolors="white",
                           marker="o")
                if rank < 2:
                    name = c.get("name", str(c["concept_id"]))
                    name = name[:22] + "." if len(name) > 22 else name
                    ax.text(xi, ly + 0.22 + rank * 0.22, name,
                            ha="center", va="bottom", fontsize=7.5,
                            color="#333333", rotation=30,
                            bbox=dict(boxstyle="round,pad=0.15", fc="white",
                                      ec="none", alpha=0.75))

    # Axes
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, fontsize=9, ha="center")
    ax.set_yticks([0.0, 1.0, 2.0])
    ax.set_yticklabels(["Drugs", "Procedures", "Conditions"], fontsize=11)
    ax.set_xlim(-0.7, n_v - 0.3)
    ax.set_ylim(-0.55, 3.1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)

    # Legend
    legend_handles = [
        mpatches.Patch(color=DOMAIN_COLORS["condition"], label="Condition (in top-K attribution)"),
        mpatches.Patch(color=DOMAIN_COLORS["procedure"], label="Procedure (in top-K attribution)"),
        mpatches.Patch(color=DOMAIN_COLORS["drug"],      label="Drug (in top-K attribution)"),
        mpatches.Patch(color="#cccccc",                  label="Observed, not in top-K"),
        mpatches.Patch(color="#9b59b6", alpha=0.3,       label="Current visit (blank — see attribution bars tab)"),
    ]
    ax.legend(handles=legend_handles, loc="upper center",
              bbox_to_anchor=(0.5, -0.18), fontsize=8,
              framealpha=0.9, edgecolor="#cccccc", ncol=3)

    plt.tight_layout(rect=[0, 0.12, 1, 1])
    return fig


HIERARCHY_TITLES = {
    "condition": "SNOMED conditions",
    "procedure": "SNOMED procedures",
    "drug":      "RxNorm drugs",
}


def _draw_domain_hierarchy(ax, entries, concepts, domain, siblings: dict = None):
    """
    Draw one ancestor ladder for a single domain onto ax.

    siblings: {concept_id (int): [{"concept_id": int, "name": str}]}
              When provided, sibling concepts (same SNOMED/RxNorm parent) are
              shown below each concept column as clinician suggestions.
    """
    color      = DOMAIN_COLORS[domain]
    has_sibs   = bool(siblings)
    ax.axis("off")
    ax.set_title(HIERARCHY_TITLES[domain],
                 fontsize=11, fontweight="bold", color=color, pad=4)

    if not entries:
        ax.text(0.5, 0.5, f"No {domain} data", ha="center", va="center",
                transform=ax.transAxes, fontsize=11, color="#888")
        return

    n_cols   = len(entries)
    y_bottom = -1.85 if has_sibs else -0.45
    ax.set_xlim(-0.55, n_cols - 0.45)
    ax.set_ylim(y_bottom, 2.75)
    ax.invert_yaxis()   # y=0 at top → concept at top, ancestors below

    for col, e in enumerate(entries):
        cid  = e["concept_id"]
        info = concepts.get(str(cid), {})
        ancs = sorted(info.get("ancestors", []), key=lambda a: a["levels_up"])

        # ── Concept box (row 0) ───────────────────────────────────────────────
        ax.text(col, 0, wrap(info.get("name", f"cid={cid}"), 16),
                ha="center", va="center", fontsize=10, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.38",
                          fc=color + "cc", ec=color, lw=1.2))
        ax.text(col, 0.42, f"score: {e['score']:.4f}",
                ha="center", va="center", fontsize=9, color="#555555")

        # ── Ancestor rows (1 = parent, 2 = grandparent) ───────────────────────
        plotted = 0
        for anc in ancs:
            lv = anc["levels_up"]
            if lv > 2 or plotted >= 2:
                continue
            shade = 0.55 + lv * 0.12
            fc    = f"#{int(shade*200):02x}{int(shade*210):02x}{int(shade*230):02x}"
            ax.text(col, lv, wrap(anc["name"], 16),
                    ha="center", va="center", fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.3", fc=fc, ec="#aaaaaa", lw=0.8))
            ax.plot([col, col], [plotted + 0.3, lv - 0.3],
                    color="#bbbbbb", lw=0.8, ls="--", zorder=0)
            plotted += 1

        # ── Sibling section (below concept box) ───────────────────────────────
        if has_sibs:
            sibs = (siblings or {}).get(cid, [])
            # Dotted connector from concept box to sibling section
            ax.plot([col, col], [-0.38, -0.62],
                    color="#bbbbbb", lw=0.7, ls=":", zorder=0)
            if sibs:
                sib_lines = "\n".join(
                    f"• {s['name'][:22] + '…' if len(s['name']) > 22 else s['name']}"
                    for s in sibs
                )
                ax.text(col, -0.68, sib_lines,
                        ha="center", va="top", fontsize=8.5,
                        color="#444444", linespacing=1.45,
                        bbox=dict(boxstyle="round,pad=0.3",
                                  fc="#f4f4f8", ec="#bbbbcc", lw=0.7))
            else:
                ax.text(col, -0.72, "(no siblings found)",
                        ha="center", va="top", fontsize=8,
                        color="#aaaaaa", style="italic")

    # ── Row labels on left margin ─────────────────────────────────────────────
    row_labels = [("Concept", 0), ("+1 Parent", 1), ("+2 Grandparent", 2)]
    if has_sibs:
        row_labels.append(("Similar\n(siblings)", -0.72))
    for lbl, row in row_labels:
        ax.text(-0.5, row, lbl, ha="right", va="center",
                fontsize=9.5, color="#666666", style="italic")


def fig_ancestors(data: dict, n: int = 4, siblings: dict = None) -> plt.Figure:
    """
    Three-domain hierarchy figure: SNOMED conditions, SNOMED procedures, RxNorm drugs.
    siblings: {concept_id (int): [{"concept_id": int, "name": str}]}
    """
    concepts = data.get("concepts", {})
    domains  = ("condition", "procedure", "drug")

    # Determine figure width from widest domain
    max_cols = max(
        len(data["attributions"].get(d, [])[:n]) for d in domains
    )
    fig_w  = max(6, max_cols * 3.5)
    fig_h  = 14 if siblings else 11   # extra height for sibling rows

    fig, axes = plt.subplots(3, 1, figsize=(fig_w, fig_h),
                             constrained_layout=True)

    for ax, domain in zip(axes, domains):
        entries = data["attributions"].get(domain, [])[:n]
        _draw_domain_hierarchy(ax, entries, concepts, domain, siblings=siblings)

    return fig


# ── App ───────────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="Patient Attribution Explorer",
        layout="wide",
    )

    # ── Global font size CSS ──────────────────────────────────────────────────
    st.markdown("""
    <style>
        /* Sidebar labels and text */
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] .stMarkdown p,
        [data-testid="stSidebar"] .stSelectbox label,
        [data-testid="stSidebar"] .stSlider label,
        [data-testid="stSidebar"] .stTextInput label {
            font-size: 15px !important;
        }
        [data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] {
            font-size: 14px !important;
        }
        /* Sidebar title */
        [data-testid="stSidebar"] h1 {
            font-size: 20px !important;
        }
        /* Tab labels */
        .stTabs [data-baseweb="tab"] {
            font-size: 15px !important;
            padding: 8px 18px !important;
        }
        /* Metric labels and values */
        [data-testid="stMetricLabel"] {
            font-size: 14px !important;
        }
        [data-testid="stMetricValue"] {
            font-size: 18px !important;
        }
        /* Main content text */
        .stMarkdown p, .stCaption {
            font-size: 14px !important;
        }
        /* Expander label */
        .streamlit-expanderHeader {
            font-size: 15px !important;
        }
        /* Slider labels */
        .stSlider label {
            font-size: 14px !important;
        }
    </style>
    """, unsafe_allow_html=True)

    # ── Sidebar ───────────────────────────────────────────────────────────────
    st.sidebar.title("Patient Attribution Explorer")

    data_dir = st.sidebar.text_input("Data directory", value=find_data_dir())
    meta     = load_metadata(data_dir)

    if not meta:
        st.error(f"No `*_enriched.json` files found in **{data_dir}**.\n\n"
                 "Run `batch_query.py` first.")
        st.stop()

    st.sidebar.success(f"{len(meta):,} patients loaded")
    st.sidebar.markdown("---")

    # ── Optional DB connection (for sibling concept lookup) ───────────────────
    with st.sidebar.expander("Database (sibling lookup)", expanded=False):
        db_host = st.text_input("Host",     value="localhost",    key="db_host")
        db_name = st.text_input("Database", value="mimiciv_omop", key="db_name")
        db_user = st.text_input("User",     value="postgres",     key="db_user")
        db_pass = st.text_input("Password", value="postgres",
                                type="password", key="db_pass")
        db_port = st.number_input("Port", value=5432, step=1, key="db_port")
        db_enabled = st.toggle("Enable sibling lookup", value=False, key="db_enabled")

    st.sidebar.markdown("---")

    min_prob = st.sidebar.slider("Min probability", 0.0, 1.0, 0.0, 0.05)

    all_labels   = sorted({m["label_name"] for m in meta.values()})
    label_filter = st.sidebar.selectbox("Filter by predicted label",
                                        ["(all)"] + all_labels)

    filtered = {
        pid: m for pid, m in meta.items()
        if m["probability"] >= min_prob
        and (label_filter == "(all)" or m["label_name"] == label_filter)
    }

    if not filtered:
        st.warning("No patients match the current filters.")
        st.stop()

    st.sidebar.markdown(f"**{len(filtered):,} patients match filters**")
    st.sidebar.markdown("---")

    sorted_pids = sorted(filtered.keys(),
                         key=lambda p: filtered[p]["probability"],
                         reverse=True)

    def option_label(pid):
        m = filtered[pid]
        return f"P{pid}  |  {m['label_name'][:40]}  |  p={m['probability']:.3f}"

    selected_label = st.sidebar.selectbox(
        "Select patient  (type to search)",
        options=sorted_pids,
        format_func=option_label,
    )

    # ── Load full data ────────────────────────────────────────────────────────
    m    = filtered[selected_label]
    data = load_patient(m["path"])

    label_name   = data.get("label_name") or data.get("label_snomed", "?")
    label_snomed = data.get("label_snomed", "?")
    prob         = data.get("probability", 0)

    # ── Main panel ────────────────────────────────────────────────────────────
    st.title(f"Patient {selected_label}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Predicted condition", label_name)
    c2.metric("SNOMED", label_snomed)
    c3.metric("Probability", f"{prob:.4f}")
    c4.metric("Graph nodes",
              f"{m['n_conditions']}c / {m['n_procedures']}p / {m['n_drugs']}d")

    st.markdown("---")
    top_k = st.slider("Top-K concepts per domain", 3, 15, 8)

    tab_bars, tab_net, tab_anc, tab_journey, tab_raw = st.tabs([
        "Attribution bars", "Concept graph", "Concept hierarchy",
        "Patient journey", "Raw JSON",
    ])

    with tab_bars:
        st.pyplot(fig_bars(data, top_k=top_k), use_container_width=False)
        st.caption("Bar length = gradient of predicted logit w.r.t. concept "
                   "embedding (absolute sum). Higher = more influential.")

    with tab_net:
        k_net = st.slider("Concepts per domain (network)", 3, 10, 5)
        st.pyplot(fig_network(data, top_k=k_net), use_container_width=False)
        st.caption("Node size scales with attribution score. Colour = domain.")

    with tab_anc:
        n_anc = st.slider("Top concepts per domain", 2, 6, 4)

        # ── Sibling lookup (requires DB connection toggled on in sidebar) ──────
        siblings = None
        if db_enabled:
            # Collect all attributed concept IDs shown in the hierarchy
            all_cids = tuple(
                int(e["concept_id"])
                for domain in ("condition", "procedure", "drug")
                for e in data["attributions"].get(domain, [])[:n_anc]
            )
            if all_cids:
                n_sibs = st.slider("Siblings per concept", 1, 5, 3,
                                   key="n_sibs_anc")
                with st.spinner("Fetching sibling concepts from DB..."):
                    siblings = fetch_siblings(
                        all_cids,
                        db_host, db_name, db_user, db_pass, int(db_port),
                        limit_per=n_sibs,
                    )
                total_sibs = sum(len(v) for v in siblings.values())
                st.caption(f"Found {total_sibs} sibling concepts across "
                           f"{len(siblings)} attributed concepts.")
        else:
            st.info("Enable **Database (sibling lookup)** in the sidebar to show "
                    "clinically similar concepts beneath each entry.",
                    icon="💡")

        st.pyplot(fig_ancestors(data, n=n_anc, siblings=siblings),
                  use_container_width=True)
        caption = (
            "Parent/grandparent hierarchy for top concepts in each domain. "
            "Conditions and procedures follow the SNOMED CT hierarchy; "
            "drugs follow the RxNorm ingredient → dose group → form hierarchy."
        )
        if siblings is not None:
            caption += (" Siblings (Similar) are concepts sharing the same "
                        "immediate parent — alternative diagnoses, procedures, "
                        "or medications in the same clinical class.")
        st.caption(caption)

    with tab_journey:
        n_visits_total = len(data.get("visits", []))
        if n_visits_total == 0:
            st.warning("No visit timeline data. Re-run `batch_query.py --overwrite` "
                       "to add visit data to enriched JSONs.")
        else:
            st.markdown(f"**{n_visits_total} visits** in patient record")
            show_all = st.toggle("Show all visits", value=n_visits_total <= 8)
            max_v    = 0 if show_all else st.slider(
                "Visits to show (first + last N)", 4, min(20, n_visits_total),
                min(8, n_visits_total))
            st.pyplot(fig_journey(data, max_visits=0 if show_all else max_v),
                      use_container_width=True)
            st.caption(
                "Each column = one visit (sorted by date). "
                "Coloured bubbles = concepts with attribution scores (size ~ score). "
                "Grey dots = concepts present but not in top attribution. "
                "Purple band = current (last) visit used for prediction."
            )

    with tab_raw:
        st.json(data)

    # ── Summary table ─────────────────────────────────────────────────────────
    st.markdown("---")
    with st.expander(f"All {len(filtered):,} filtered patients"):
        import pandas as pd
        rows = [{
            "Patient":    pid,
            "Label":      m["label_name"][:50],
            "Prob":       round(m["probability"], 4),
            "Conditions": m["n_conditions"],
            "Procedures": m["n_procedures"],
            "Drugs":      m["n_drugs"],
        } for pid, m in sorted(filtered.items(),
                               key=lambda x: x[1]["probability"],
                               reverse=True)]
        st.dataframe(pd.DataFrame(rows), use_container_width=True,
                     hide_index=True)


if __name__ == "__main__":
    main()