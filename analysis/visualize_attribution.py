"""
visualize_attribution.py
========================
Produces a three-panel publication-quality figure from an enriched
attribution JSON (output of query_concepts.py).

Panel A  --  Horizontal attribution bars, grouped by domain
             (Conditions / Procedures / Drugs).
Panel B  --  Patient subgraph: nodes sized & coloured by attribution score,
             laid out as a star around the predicted label.
Panel C  --  SNOMED ancestor ladder for the top-3 conditions
             (concept -> parent -> grandparent).

Usage:
    py -3 analysis/visualize_attribution.py \
        --enriched attribution_p42_enriched.json \
        --out      analysis/explain/attribution_p42_viz.png

Requires:  matplotlib, networkx  (pip install networkx)
"""

import argparse
import json
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np

# ── Colour palette ─────────────────────────────────────────────────────────────
DOMAIN_COLORS = {
    "condition": "#e07b39",   # orange
    "procedure": "#3a7abf",   # blue
    "drug":      "#4aab6d",   # green
}
LABEL_COLOR   = "#9b59b6"    # purple for predicted label node

# ── Helpers ────────────────────────────────────────────────────────────────────

def short_name(name: str, max_len: int = 40) -> str:
    """Truncate long concept names for axis labels."""
    return name if len(name) <= max_len else name[:max_len - 3] + "..."


def wrap(text: str, width: int = 22) -> str:
    """Wrap text for node labels."""
    return "\n".join(textwrap.wrap(text, width))


# ── Panel A: Attribution bar charts ───────────────────────────────────────────

def plot_bars(ax, entries, concepts, domain, color, top_k=8):
    """Horizontal bar chart for one domain — names wrapped to avoid overlap."""
    entries = entries[:top_k]
    names  = []
    scores = []
    for e in entries:
        cid  = e["concept_id"]
        info = concepts.get(str(cid), {})
        # Wrap at 28 chars so long names never push off the axis
        names.append(wrap(info.get("name", f"concept {cid}"), 22))
        scores.append(e["score"])

    y    = np.arange(len(names))
    bars = ax.barh(y, scores, color=color, alpha=0.78, edgecolor=color,
                   linewidth=0.8, height=0.55)

    for bar, sc in zip(bars, scores):
        ax.text(bar.get_width() + max(scores) * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{sc:.4f}", va="center", ha="left", fontsize=10, color="#333333")

    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=10, linespacing=1.3)
    ax.invert_yaxis()
    ax.set_xlabel("Attribution score", fontsize=11)
    ax.set_title(domain.upper(), fontsize=12, fontweight="bold",
                 color=color, pad=5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", labelsize=10)
    ax.set_xlim(0, max(scores) * 1.35)


# ── Panel B: Patient subgraph network ─────────────────────────────────────────

def abbrev(name: str, wrap_width: int = 18) -> str:
    """Wrap concept name for network labels — full text, wrapped at wrap_width chars."""
    return "\n".join(textwrap.wrap(name, wrap_width))


def plot_network(ax, data, concepts, top_k_per_domain=3):
    """Star network with manual polar layout.

    Nodes are placed on a circle at evenly-spaced angles.
    Labels sit on a wider outer ring, aligned radially so they
    never overlap regardless of node count.
    """
    # ── Collect entries (max 3 per domain → 9 outer nodes) ───────────────
    all_entries = []
    for domain in ("condition", "procedure", "drug"):
        for e in data["attributions"].get(domain, [])[:top_k_per_domain]:
            all_entries.append((domain, e))

    n_outer = len(all_entries)
    if n_outer == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes)
        ax.axis("off")
        return

    # ── Score normalisation ───────────────────────────────────────────────
    scores  = [e["score"] for _, e in all_entries]
    s_min, s_max = min(scores), max(scores)
    def norm_size(s):
        return 600 + (s - s_min) / max(s_max - s_min, 1e-9) * 1800

    # ── Manual polar positions ────────────────────────────────────────────
    R_node  = 1.0   # radius of node ring
    R_label = 1.55  # radius of label ring (further out)
    cx, cy  = 0.0, 0.0

    angles = [2 * np.pi * i / n_outer - np.pi / 2   # start at top
              for i in range(n_outer)]

    node_pos   = {}   # node_id -> (x, y) on node ring
    label_pos  = {}   # node_id -> (x, y) on label ring

    all_node_colors = [LABEL_COLOR]
    all_node_sizes  = [2800]
    all_node_ids    = ["label_center"]

    for i, (domain, e) in enumerate(all_entries):
        cid     = e["concept_id"]
        node_id = f"{domain}_{cid}"
        a       = angles[i]
        node_pos[node_id]  = (cx + R_node  * np.cos(a), cy + R_node  * np.sin(a))
        label_pos[node_id] = (cx + R_label * np.cos(a), cy + R_label * np.sin(a))
        all_node_ids.append(node_id)
        all_node_colors.append(DOMAIN_COLORS[domain])
        all_node_sizes.append(norm_size(e["score"]))

    # Centre node position
    node_pos["label_center"] = (cx, cy)

    # ── Draw edges (spokes from concept nodes to centre) ──────────────────
    for node_id in all_node_ids[1:]:
        x0, y0 = node_pos[node_id]
        ax.annotate("", xy=(cx, cy), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", color="#bbbbbb",
                                   lw=0.9, alpha=0.5))

    # ── Draw nodes ────────────────────────────────────────────────────────
    for idx, node_id in enumerate(all_node_ids):
        x, y = node_pos[node_id]
        ax.scatter(x, y,
                   s=all_node_sizes[idx],
                   c=all_node_colors[idx],
                   alpha=0.88, zorder=3,
                   linewidths=0.5, edgecolors="white")

    # ── Centre label (inside hub) ─────────────────────────────────────────
    label_name = abbrev(data.get("label_name", "?"), 16)
    ax.text(cx, cy,
            f"Predicted\n{label_name}\np={data['probability']:.2f}",
            ha="center", va="center", fontsize=9,
            color="#111111", fontweight="normal",
            multialignment="center", zorder=4)

    # ── Concept labels on outer ring ──────────────────────────────────────
    for i, (domain, e) in enumerate(all_entries):
        cid     = e["concept_id"]
        node_id = f"{domain}_{cid}"
        info    = concepts.get(str(cid), {})
        name    = abbrev(info.get("name", f"cid={cid}"), 16)

        lx, ly = label_pos[node_id]
        a       = angles[i]

        ha = "left"  if np.cos(a) >  0.15 else \
             "right" if np.cos(a) < -0.15 else "center"
        va = "bottom" if np.sin(a) >  0.15 else \
             "top"    if np.sin(a) < -0.15 else "center"

        ax.text(lx, ly, name,
                ha=ha, va=va,
                fontsize=9.5, color="#111111", fontweight="normal",
                multialignment=ha, linespacing=1.3,
                bbox=dict(boxstyle="round,pad=0.25", fc="white",
                          ec="none", alpha=0.85),
                zorder=5)

    # ── Axes ─────────────────────────────────────────────────────────────
    margin = R_label + 0.85
    ax.set_xlim(cx - margin, cx + margin)
    ax.set_ylim(cy - margin, cy + margin)
    ax.set_aspect("equal")
    ax.set_title("Patient subgraph  (node size ~ attribution score)",
                 fontsize=12, fontweight="bold", pad=6)
    ax.axis("off")

    patches = [mpatches.Patch(color=DOMAIN_COLORS[d], label=d.capitalize())
               for d in ("condition", "procedure", "drug")]
    patches.append(mpatches.Patch(color=LABEL_COLOR, label="Predicted label"))
    ax.legend(handles=patches, loc="lower left", fontsize=10,
              framealpha=0.9, edgecolor="#cccccc")


# ── Panel C: Ancestor hierarchy ladder (all three domains) ────────────────────

# Vocabulary hierarchy titles per domain
HIERARCHY_TITLES = {
    "condition": "SNOMED conditions",
    "procedure": "SNOMED procedures",
    "drug":      "RxNorm drugs",
}


def plot_domain_hierarchy(ax, attributions, concepts, domain, n_concepts=3):
    """
    One horizontal ladder for a single domain.
    Each column is one top concept; rows show concept -> parent -> grandparent.
    """
    color   = DOMAIN_COLORS[domain]
    entries = attributions.get(domain, [])[:n_concepts]

    if not entries:
        ax.text(0.5, 0.5, f"No {domain} data", ha="center", va="center",
                transform=ax.transAxes, fontsize=9, color="#888")
        ax.axis("off")
        return

    n_cols = len(entries)

    ax.set_xlim(-0.7, n_cols - 0.3)
    ax.set_ylim(-0.55, 2.6)
    ax.invert_yaxis()
    ax.axis("off")
    ax.set_title(HIERARCHY_TITLES[domain],
                 fontsize=10, fontweight="bold", color=color, pad=3)

    for col, e in enumerate(entries):
        cid       = e["concept_id"]
        score     = e["score"]
        info      = concepts.get(str(cid), {})
        name      = info.get("name", f"cid={cid}")
        ancestors = sorted(info.get("ancestors", []), key=lambda a: a["levels_up"])

        # Level 0 — the concept itself
        ax.text(col, 0, wrap(name, 13), ha="center", va="center", fontsize=8,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.32",
                          fc=color + "cc", ec=color, linewidth=1.0))
        ax.text(col, 0.42, f"{score:.4f}", ha="center", va="center",
                fontsize=7.5, color="#555555")

        # Ancestor levels 1 and 2
        plotted = 0
        for anc in ancestors:
            lv = anc["levels_up"]
            if lv > 2 or plotted >= 2:
                continue
            row   = lv
            shade = 0.55 + lv * 0.10
            fc    = f"#{int(shade*200):02x}{int(shade*210):02x}{int(shade*230):02x}"
            ax.text(col, row, wrap(anc["name"], 13), ha="center", va="center",
                    fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.25", fc=fc, ec="#aaaaaa",
                              linewidth=0.7))
            ax.plot([col, col], [plotted + 0.3, row - 0.3],
                    color="#bbbbbb", lw=0.8, ls="--", zorder=0)
            plotted += 1

    # Row labels on left margin
    for row, label in enumerate(["Concept", "Parent", "Grandparent"]):
        ax.text(-0.65, row, label, ha="right", va="center", fontsize=7.5,
                color="#666666", style="italic")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--enriched", required=True,
                   help="Enriched attribution JSON from query_concepts.py")
    p.add_argument("--out", default=None,
                   help="Output PNG path (default: <enriched_stem>_viz.png)")
    p.add_argument("--top_k", type=int, default=8,
                   help="Max concepts per domain in bar chart")
    p.add_argument("--dpi", type=int, default=200)
    args = p.parse_args()

    with open(args.enriched) as f:
        data = json.load(f)

    concepts = data["concepts"]   # str(concept_id) -> info dict

    out_path = Path(args.out) if args.out else \
               Path(args.enriched).parent / (Path(args.enriched).stem + "_viz.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Layout ─────────────────────────────────────────────────────────────────
    # Row 0 (small): three attribution bar charts
    # Row 1 (large): network (left half) + SNOMED hierarchy (right half)
    fig = plt.figure(figsize=(20, 20))
    fig.suptitle(
        f"Patient {data['patient_idx']}  |  Predicted: {data.get('label_name','?')} "
        f"(SNOMED {data['label_snomed']})  |  p = {data['probability']:.4f}",
        fontsize=13, fontweight="bold", y=0.995,
    )

    gs = gridspec.GridSpec(2, 2, figure=fig,
                           hspace=0.38, wspace=0.35,
                           height_ratios=[0.7, 1.3])

    # Top row: three bar charts share the two columns via nested gridspec
    gs_top = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[0, :],
                                              wspace=0.45)
    ax_cond = fig.add_subplot(gs_top[0])
    ax_proc = fig.add_subplot(gs_top[1])
    ax_drug = fig.add_subplot(gs_top[2])

    # Bottom row: network left, 3 stacked hierarchy panels right
    ax_net = fig.add_subplot(gs[1, 0])

    gs_hier = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs[1, 1],
                                               hspace=0.55)
    ax_hier_cond = fig.add_subplot(gs_hier[0])
    ax_hier_proc = fig.add_subplot(gs_hier[1])
    ax_hier_drug = fig.add_subplot(gs_hier[2])

    for domain, ax, color in [
        ("condition", ax_cond, DOMAIN_COLORS["condition"]),
        ("procedure", ax_proc, DOMAIN_COLORS["procedure"]),
        ("drug",      ax_drug, DOMAIN_COLORS["drug"]),
    ]:
        entries = data["attributions"].get(domain, [])
        if entries:
            plot_bars(ax, entries, concepts, domain, color, top_k=args.top_k)
        else:
            ax.text(0.5, 0.5, f"No {domain} data", ha="center", va="center",
                    transform=ax.transAxes)
            ax.axis("off")

    plot_network(ax_net, data, concepts, top_k_per_domain=3)
    plot_domain_hierarchy(ax_hier_cond, data["attributions"], concepts, "condition", n_concepts=3)
    plot_domain_hierarchy(ax_hier_proc, data["attributions"], concepts, "procedure", n_concepts=3)
    plot_domain_hierarchy(ax_hier_drug, data["attributions"], concepts, "drug",      n_concepts=3)

    plt.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()