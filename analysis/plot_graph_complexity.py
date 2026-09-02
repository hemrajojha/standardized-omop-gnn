"""
plot_graph_complexity.py
========================
Single-panel figure comparing patient-local graph sizes (V_patient)
against the TRANS global graph (V_global).

Panel A  --  Node count: Simple patients / Complex patients / TRANS global
             Y-axis log scale.

Data source: analysis/complexity/complexity_results.csv
TRANS global node count: 15,435 (verified from omop_4.pkl:
  vocab_co=276 conditions, vocab_pr=12,559 procedures, vocab_dh=2,600 drugs)

Run:
    py -3 analysis/plot_graph_complexity.py
"""

import numpy as np
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
# From complexity_results.csv (weighted means for Simple = Q1+Q2, Complex = Q3+Q4)
# Q1: n=3758, mean=19.8
# Q2: n=3818, mean=41.0
# Q3: n=3710, mean=68.3
# Q4: n=3738, mean=141.2

n_q1, m_q1 = 3758, 19.8
n_q2, m_q2 = 3818, 41.0
n_q3, m_q3 = 3710, 68.3
n_q4, m_q4 = 3738, 141.2

# Weighted means for two groups
n_simple  = n_q1 + n_q2
n_complex = n_q3 + n_q4

mean_nodes_simple  = (n_q1*m_q1 + n_q2*m_q2) / n_simple   # ~30.5
mean_nodes_complex = (n_q3*m_q3 + n_q4*m_q4) / n_complex   # ~104.9

TRANS_GLOBAL = 15435   # verified from omop_4.pkl: 276 co + 12559 pr + 2600 dh

# ── Colours ───────────────────────────────────────────────────────────────────
C_SIMPLE  = "#4aab6d"   # green  — simple patients
C_COMPLEX = "#e07b39"   # orange — complex patients
C_TRANS   = "#9b59b6"   # purple — TRANS global

# ── Figure ────────────────────────────────────────────────────────────────────
fig, ax1 = plt.subplots(figsize=(7, 6))

# ─────────────────────────────────────────────────────────────────────────────
# Panel — Node count comparison (log scale)
# ─────────────────────────────────────────────────────────────────────────────
groups  = ["Simple\npatients\n(Q1+Q2)", "Complex\npatients\n(Q3+Q4)", "TRANS\nglobal\ngraph"]
values  = [mean_nodes_simple, mean_nodes_complex, TRANS_GLOBAL]
colors  = [C_SIMPLE, C_COMPLEX, C_TRANS]
x       = np.arange(len(groups))
width   = 0.5

bars = ax1.bar(x, values, width=width, color=colors, alpha=0.82,
               edgecolor="white", linewidth=0.8)

# Value labels on top of each bar
for bar, val in zip(bars, values):
    label = f"{val:,.0f}" if val >= 1000 else f"{val:.1f}"
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.12,
             label, ha="center", va="bottom", fontsize=11.5, fontweight="bold")

ax1.set_yscale("log")
ax1.set_xticks(x)
ax1.set_xticklabels(groups, fontsize=12)
ax1.set_ylabel("Number of concept nodes  (log scale)", fontsize=13)
ax1.set_ylim(5, TRANS_GLOBAL * 4)
ax1.yaxis.set_major_formatter(ticker.FuncFormatter(
    lambda v, _: f"{int(v):,}" if v >= 1000 else f"{v:.0f}"
))
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)


# ── Legend ────────────────────────────────────────────────────────────────────
patches = [
    mpatches.Patch(color=C_SIMPLE,  alpha=0.82, label=f"Simple patients (Q1+Q2, n={n_simple:,})"),
    mpatches.Patch(color=C_COMPLEX, alpha=0.82, label=f"Complex patients (Q3+Q4, n={n_complex:,})"),
    mpatches.Patch(color=C_TRANS,   alpha=0.82, label="TRANS global graph"),
]
ax1.legend(handles=patches, loc="upper left", fontsize=10.5,
           framealpha=0.9, edgecolor="#cccccc")

plt.tight_layout()

out = "analysis/complexity/fig_graph_complexity.png"
plt.savefig(out, dpi=200, bbox_inches="tight")
print(f"Saved -> {out}")