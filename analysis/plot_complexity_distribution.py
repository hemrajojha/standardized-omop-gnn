"""
plot_complexity_distribution.py
================================
Histogram of patient graph sizes (concept node counts) across all
15,024 test patients on a log x-axis, with quartile boundaries and
TRANS global graph size as a vertical reference line.

Run:
    py -3 analysis/plot_complexity_distribution.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker

plt.rcParams.update({
    "font.size": 13,
    "axes.labelsize": 13,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 11,
})

# ── Data ──────────────────────────────────────────────────────────────────────
df = pd.read_csv("analysis/complexity/complexity_per_patient.csv")
complexity = df["complexity"].values.astype(float)
complexity = complexity[complexity > 0]   # log scale requires > 0

TRANS_GLOBAL = 15435   # verified from omop_4.pkl

q25 = np.percentile(complexity, 25)   # ~30
q50 = np.percentile(complexity, 50)   # ~52
q75 = np.percentile(complexity, 75)   # ~88

# ── Colours ───────────────────────────────────────────────────────────────────
C_SIMPLE  = "#4aab6d"
C_COMPLEX = "#e07b39"
C_TRANS   = "#9b59b6"

# ── Figure ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 6.5))

# Log-spaced bins from 1 to just beyond TRANS
log_bins = np.logspace(np.log10(1), np.log10(TRANS_GLOBAL * 1.5), 60)

simple_vals  = complexity[complexity <= q50]
complex_vals = complexity[complexity  > q50]

ax.hist(simple_vals,  bins=log_bins, color=C_SIMPLE,  alpha=0.75)
ax.hist(complex_vals, bins=log_bins, color=C_COMPLEX, alpha=0.75)

ax.set_xscale("log")

# ── Quartile / TRANS vertical lines ───────────────────────────────────────────
ymax = ax.get_ylim()[1]

# Quartile boundary lines — no text labels
for q, ls in [(q25, "--"), (q50, "-"), (q75, "--")]:
    ax.axvline(q, color="#555555", lw=1.2, ls=ls, zorder=3)

# TRANS line — full height; label box sits mid-line, below where legend ends
ax.set_ylim(0, ymax * 1.45)
trans_line_top = ymax * 1.35
ax.annotate("", xy=(TRANS_GLOBAL, trans_line_top), xytext=(TRANS_GLOBAL, 0),
            arrowprops=dict(arrowstyle="-", color=C_TRANS, lw=1.8))
ax.text(TRANS_GLOBAL, ymax * 0.55,
        f"TRANS global graph\n{TRANS_GLOBAL:,} nodes\n(all n={len(complexity):,} patients)",
        va="center", ha="center", fontsize=9.5, color=C_TRANS,
        fontweight="bold", linespacing=1.4,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=C_TRANS, lw=0.8, alpha=0.9))

# ── Axes ──────────────────────────────────────────────────────────────────────
ax.set_xlabel("Patient graph size (number of concept nodes, log scale)", fontsize=13)
ax.set_ylabel("Number of patients", fontsize=13)
ax.set_xlim(1, TRANS_GLOBAL * 3)
ax.xaxis.set_major_formatter(ticker.FuncFormatter(
    lambda v, _: f"{int(v):,}" if v >= 1000 else f"{int(v)}"
))
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# ── Legend ────────────────────────────────────────────────────────────────────
patches = [
    mpatches.Patch(color=C_SIMPLE,  alpha=0.75,
                   label=f"Simple patients (≤{int(q50)} nodes, n={len(simple_vals):,})"),
    mpatches.Patch(color=C_COMPLEX, alpha=0.75,
                   label=f"Complex patients (>{int(q50)} nodes, n={len(complex_vals):,})"),
    mpatches.Patch(color=C_TRANS,   alpha=0.9,
                   label=f"TRANS global graph ({TRANS_GLOBAL:,} nodes)"),
]
ax.legend(handles=patches, loc="upper right", fontsize=10.5,
          framealpha=0.9, edgecolor="#cccccc",
          bbox_to_anchor=(0.98, 0.98), borderaxespad=0)

plt.tight_layout()

out = "analysis/complexity/fig_complexity_distribution.png"
plt.savefig(out, dpi=200, bbox_inches="tight")
print(f"Saved -> {out}")