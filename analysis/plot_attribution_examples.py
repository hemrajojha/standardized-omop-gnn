"""
plot_attribution_examples.py
============================
Generates publication-quality attribution figures for two selected patients:
  - p8271  : Atherosclerosis of coronary artery (common disease)
  - p19485 : Crohn's disease (uncommon disease)

Layout (2 rows × 3 cols):
  Row 1 — Attribution bar charts (condition / procedure / drug)
  Row 2 left (cols 0-1) — Concept star graph  (near-square space)
  Row 2 right (col 2)   — Concept hierarchies stacked (SNOMED / SNOMED / RxNorm)

Run:
    py -3 analysis/plot_attribution_examples.py
"""

import json
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

plt.rcParams.update({
    "font.size":         15,
    "axes.labelsize":    15,
    "xtick.labelsize":   13,
    "ytick.labelsize":   13,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

DOMAIN_COLORS = {
    "condition": "#e07b39",
    "procedure": "#3a7abf",
    "drug":      "#4aab6d",
}
LABEL_COLOR = "#9b59b6"

DATA_DIR = Path("analysis/explain/batch")
OUT_DIR  = Path("analysis/explain")
PATIENTS = {
    8271:  "common",    # Atherosclerosis of coronary artery
    19485: "uncommon",  # Crohn's disease
}

TOP_K_BARS = 5   # bars shown per domain
TOP_K_NET  = 3   # nodes per domain in star graph
TOP_K_ANC  = TOP_K_BARS   # hierarchy matches bars


# ── Helpers ───────────────────────────────────────────────────────────────────

def short(name, n=38):
    return name if len(name) <= n else name[:n - 3] + "..."

def wrap_label(name, width=16):
    return "\n".join(textwrap.wrap(name, width))

def abbrev(name, max_words=3, max_chars=18):
    words = name.split()
    s = " ".join(words[:max_words])
    return s if len(s) <= max_chars else s[:max_chars - 1] + "."

def hier_label(name: str, n_cols: int) -> str:
    """
    Wrap concept names to at most 2 lines so boxes grow vertically
    rather than horizontally.  Line width shrinks with more columns.
    """
    width = {1: 16, 2: 13, 3: 11, 4: 9}.get(n_cols, 8)
    lines = textwrap.wrap(name, width)[:2]
    if not lines:
        return name
    # If original had more content, mark truncation on last line
    joined = "\n".join(lines)
    if len(name) > len(joined.replace("\n", " ")):
        lines[-1] = lines[-1].rstrip() + "…"
    return "\n".join(lines)


# ── Row 1: Attribution bars ───────────────────────────────────────────────────

def draw_bars(ax, data, domain, top_k):
    concepts = data.get("concepts", {})
    entries  = data["attributions"].get(domain, [])[:top_k]
    color    = DOMAIN_COLORS[domain]

    names, scores = [], []
    for e in entries:
        cid  = e["concept_id"]
        info = concepts.get(str(cid), {})
        # Wrap to 2 lines max — tighter width for narrower vertical panel
        raw = info.get("name", f"cid={cid}")
        wrapped = "\n".join(textwrap.wrap(raw, 24))
        names.append(wrapped)
        scores.append(e["score"])

    if not scores:
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes, fontsize=11, color="#888")
        ax.axis("off")
        return

    y    = np.arange(len(names))
    bars = ax.barh(y, scores, color=color, alpha=0.75,
                   edgecolor=color, linewidth=0.6, height=0.55)
    for bar, sc in zip(bars, scores):
        ax.text(bar.get_width() + max(scores) * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{sc:.4f}", va="center", ha="left", fontsize=12)

    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=12, linespacing=1.2)
    ax.invert_yaxis()
    ax.set_xlabel("Attribution score", fontsize=13)
    ax.set_title(domain.upper(), fontsize=15, fontweight="bold", color=color, pad=4)
    ax.set_xlim(0, max(scores) * 1.45)
    ax.tick_params(axis="x", labelsize=11)
    ax.tick_params(axis="y", pad=2)


# ── Concept star graph ────────────────────────────────────────────────────────

def draw_network(ax, data, top_k):
    concepts = data.get("concepts", {})

    all_entries = []
    for domain in ("condition", "procedure", "drug"):
        for e in data["attributions"].get(domain, [])[:top_k]:
            all_entries.append((domain, e))

    if not all_entries:
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes)
        ax.axis("off")
        return

    scores       = [e["score"] for _, e in all_entries]
    s_min, s_max = min(scores), max(scores)

    def norm_size(s):
        return 500 + (s - s_min) / max(s_max - s_min, 1e-9) * 1600

    n_outer = len(all_entries)
    R_node  = 1.0
    R_label = 1.55
    cx, cy  = 0.0, 0.0
    angles  = [2 * np.pi * i / n_outer - np.pi / 2 for i in range(n_outer)]

    for i in range(n_outer):
        x0 = cx + R_node * np.cos(angles[i])
        y0 = cy + R_node * np.sin(angles[i])
        ax.annotate("", xy=(cx, cy), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", color="#cccccc",
                                   lw=0.8, alpha=0.6))

    for i, (domain, e) in enumerate(all_entries):
        x = cx + R_node * np.cos(angles[i])
        y = cy + R_node * np.sin(angles[i])
        ax.scatter(x, y, s=norm_size(e["score"]),
                   c=DOMAIN_COLORS[domain], alpha=0.88,
                   zorder=3, linewidths=0.5, edgecolors="white")

    # Centre node — larger, readable label
    ax.scatter(cx, cy, s=2400, c=LABEL_COLOR, alpha=0.88,
               zorder=3, linewidths=0.5, edgecolors="white")
    ax.text(cx, cy,
            f"Predicted\n{abbrev(data.get('label_name','?'), 3, 18)}\n"
            f"p={data['probability']:.2f}",
            ha="center", va="center", fontsize=14,
            color="#111111", multialignment="center", zorder=4)

    for i, (domain, e) in enumerate(all_entries):
        cid  = e["concept_id"]
        info = concepts.get(str(cid), {})
        name = info.get("name", f"cid={cid}")
        name = "\n".join(textwrap.wrap(name, 18))
        a    = angles[i]
        lx   = cx + R_label * np.cos(a)
        ly   = cy + R_label * np.sin(a)
        ha   = "left"   if np.cos(a) >  0.15 else \
               "right"  if np.cos(a) < -0.15 else "center"
        va   = "bottom" if np.sin(a) >  0.15 else \
               "top"    if np.sin(a) < -0.15 else "center"
        ax.text(lx, ly, name, ha=ha, va=va,
                fontsize=13, color="#111111",
                multialignment=ha, linespacing=1.3,
                bbox=dict(boxstyle="round,pad=0.22", fc="white",
                          ec="none", alpha=0.88),
                zorder=5)

    margin = R_label + 1.0
    ax.set_xlim(cx - margin, cx + margin)
    ax.set_ylim(cy - margin, cy + margin)
    ax.set_aspect("equal")
    ax.axis("off")

    patches = [mpatches.Patch(color=DOMAIN_COLORS[d], label=d.capitalize(), alpha=0.85)
               for d in ("condition", "procedure", "drug")]
    patches.append(mpatches.Patch(color=LABEL_COLOR, label="Predicted label", alpha=0.85))
    ax.legend(handles=patches, loc="lower right", fontsize=13,
              framealpha=0.9, edgecolor="#cccccc", handlelength=1.2)


# ── Row 3: Concept hierarchies (condition / procedure / drug) ─────────────────

HIERARCHY_TITLES = {
    "condition": "SNOMED conditions",
    "procedure": "SNOMED procedures",
    "drug":      "RxNorm drugs",
}


def draw_domain_hierarchy(ax, data, domain, n_concepts, show_level_labels=False):
    """One horizontal ladder for a single domain."""
    concepts = data.get("concepts", {})
    color    = DOMAIN_COLORS[domain]
    entries  = data["attributions"].get(domain, [])[:n_concepts]

    ROW_GAP = 1.6   # vertical gap between levels (accommodates 2-line boxes)

    ax.axis("off")
    ax.set_title(HIERARCHY_TITLES[domain],
                 fontsize=14, fontweight="bold", color=color, pad=4)

    if not entries:
        ax.text(0.5, 0.5, f"No {domain} data", ha="center", va="center",
                fontsize=10, color="#888")
        return

    n_cols = len(entries)

    # Column spacing: tighter for many concepts, wider for few
    COL_STEP = max(1.7, 4.0 / max(n_cols - 1, 1))
    COL_STEP = min(COL_STEP, 3.5)

    # Left margin: wider when showing level labels to give clear gap from boxes
    margin = 2.2 if show_level_labels else 0.5
    x_max  = (n_cols - 1) * COL_STEP
    ax.set_xlim(-margin, x_max + 0.5)
    ax.set_ylim(-0.55, 2 * ROW_GAP + 0.55)

    for col, e in enumerate(entries):
        x    = col * COL_STEP
        cid  = e["concept_id"]
        info = concepts.get(str(cid), {})
        ancs = sorted(info.get("ancestors", []), key=lambda a: a["levels_up"])

        # Concept box — light fill, domain colour only on border
        ax.text(x, 0, hier_label(info.get("name", f"cid={cid}"), n_cols),
                ha="center", va="center", fontsize=12, fontweight="bold",
                linespacing=1.3,
                bbox=dict(boxstyle="round,pad=0.30",
                          fc=color + "30", ec=color, lw=1.2))

        plotted = 0
        for anc in ancs:
            lv = anc["levels_up"]
            if lv > 2 or plotted >= 2:
                continue
            row = lv * ROW_GAP
            # Very light grey-blue tint for ancestor boxes
            fc  = "#eef0f5" if lv == 1 else "#f5f5f8"
            ax.text(x, row, hier_label(anc["name"], n_cols),
                    ha="center", va="center", fontsize=12,
                    linespacing=1.3,
                    bbox=dict(boxstyle="round,pad=0.28",
                              fc=fc, ec="#cccccc", lw=0.8))
            ax.plot([x, x], [plotted * ROW_GAP + 0.50, row - 0.50],
                    color="#cccccc", lw=0.8, ls="--", zorder=0)
            plotted += 1

    # Level labels on left — only drawn for the first (condition) panel
    if show_level_labels:
        for lv, lbl in [(0, "Concept"), (1, "Parent"), (2, "Grandparent")]:
            ax.text(-margin + 0.1, lv * ROW_GAP, lbl, ha="left", va="center",
                    fontsize=11, color="#666", style="italic")


# ── Compose full figure ───────────────────────────────────────────────────────

def make_figure(data, out_path):
    # Layout:
    #   Top (~62%): bar charts (left col, 3 stacked) | star graph (right col)
    #   Bottom (~38%): 3 concept hierarchy panels side by side
    fig = plt.figure(figsize=(22, 16))
    gs  = fig.add_gridspec(2, 1, height_ratios=[1.6, 1], hspace=0.30)

    # Top row — bars (left ~42%) | network (right ~58%)
    gs_top = gs[0].subgridspec(1, 2, width_ratios=[1, 1.4], wspace=0.30)

    # Left: 3 attribution bar charts stacked vertically
    gs_bars = gs_top[0].subgridspec(3, 1, hspace=0.48)
    for i, domain in enumerate(("condition", "procedure", "drug")):
        ax = fig.add_subplot(gs_bars[i])
        draw_bars(ax, data, domain, TOP_K_BARS)

    # Right: concept star graph
    ax_net = fig.add_subplot(gs_top[1])
    draw_network(ax_net, data, TOP_K_NET)

    # Bottom row — 3 hierarchy panels side by side (condition | procedure | drug)
    # Wider gap between panels (intra-domain spacing); level labels only on first
    gs_hier = gs[1].subgridspec(1, 3, wspace=0.55)
    for i, domain in enumerate(("condition", "procedure", "drug")):
        ax_h = fig.add_subplot(gs_hier[i])
        draw_domain_hierarchy(ax_h, data, domain, TOP_K_ANC,
                              show_level_labels=(i == 0))

    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for pid, category in PATIENTS.items():
        json_path = DATA_DIR / f"attribution_p{pid}_enriched.json"
        if not json_path.exists():
            print(f"[WARN] {json_path} not found, skipping.")
            continue

        with open(json_path) as f:
            data = json.load(f)

        label = data.get("label_name", data.get("label_snomed", "unknown"))
        print(f"\nPatient {pid} ({category}): {label}  p={data.get('probability',0):.3f}")
        for domain in ("condition", "procedure", "drug"):
            print(f"  {domain}: {len(data['attributions'].get(domain,[]))} attributed")

        out_path = OUT_DIR / f"fig_attribution_p{pid}_{category}.png"
        make_figure(data, out_path)