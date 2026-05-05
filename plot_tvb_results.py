#!/usr/bin/env python3
"""
Plot TVB Motor Cortex Phase 1 Simulation Results
================================================

What this simulation models
-----------------------------
The Virtual Brain (TVB) is a whole-brain simulator that uses patient-specific
structural connectivity (from Diffusion Tensor Imaging, DTI) to simulate
how neural populations interact across the entire cortex.

This Phase 1 model focuses on **motor stroke**:
  - 68 cortical regions (Desikan-Killiany atlas)
  - Neural mass model: Reduced Wong-Wang (excitatory synaptic gating dynamics)
  - Stroke: left primary motor cortex (M1) — excitability reduced to 30%
    and structural connectivity to/from M1_L reduced (simulating white matter damage)
  - Output: firing rates per region → corticospinal tract (CST) drive for each leg
  - Downstream use: these rates drive a tinyCPG spinal cord locomotion model

Key question answered: How does a unilateral M1 stroke change the cortical
drive signal that the spinal cord receives for each leg?
"""

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch, Rectangle, FancyBboxPatch
from matplotlib.colors import Normalize
import matplotlib.cm as cm
from pathlib import Path
import glob

# ── load latest result file ──────────────────────────────────────────────────
files = sorted(glob.glob("tvb_output/phase1/*.h5"))
if not files:
    raise FileNotFoundError("No H5 files found in tvb_output/phase1/")
h5path = files[-1]
print(f"Plotting: {h5path}")

with h5py.File(h5path, "r") as f:
    attrs          = dict(f.attrs)
    time_s         = f["time_s"][()]
    firing_hz      = f["firing_rates_hz"][()]       # (68, 100000)
    cst_left_hz    = f["corticospinal_drive_left_hz"][()]
    cst_right_hz   = f["corticospinal_drive_right_hz"][()]

n_regions, n_steps = firing_hz.shape
duration_s = time_s[-1]

# ── region metadata ───────────────────────────────────────────────────────────
MOTOR = {
    "M1_L":  4,  "PMC_L":  3,  "SMA_L": 31, "S1_L": 22,
    "M1_R": 39,  "PMC_R": 38,  "SMA_R": 66, "S1_R": 57,
}
MOTOR_PAIRS = [("M1_L", "M1_R"), ("PMC_L", "PMC_R"),
               ("SMA_L", "SMA_R"), ("S1_L", "S1_R")]
MOTOR_FULL  = {"M1": "Primary Motor", "PMC": "Premotor",
               "SMA": "Supplementary Motor", "S1": "Primary Sensory"}

# Downsample for plotting (100 k steps → 2000 pts)
ds  = max(1, n_steps // 2000)
t   = time_s[::ds]
fr  = firing_hz[:, ::ds]
cst_l = cst_left_hz[::ds]
cst_r = cst_right_hz[::ds]

# Mean firing rate per region (over whole simulation)
mean_fr = firing_hz.mean(axis=1)

# ── figure layout ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 14))
fig.patch.set_facecolor("#0e1117")

gs = gridspec.GridSpec(
    3, 3,
    figure=fig,
    left=0.06, right=0.97,
    top=0.91, bottom=0.07,
    hspace=0.52, wspace=0.38,
)

AX_TITLE  = fig.add_subplot(gs[0, :])   # full-width explanatory header row
AX_CST    = fig.add_subplot(gs[1, :2])  # CST drive time series (wide)
AX_ASYM   = fig.add_subplot(gs[1, 2])   # asymmetry bar
AX_HEAT   = fig.add_subplot(gs[2, :2])  # firing-rate heatmap (wide)
AX_MOTOR  = fig.add_subplot(gs[2, 2])   # motor region comparison

DARK  = "#0e1117"
LIGHT = "#e0e0e0"
STROKE_C = "#ff4b4b"   # red  – damaged left hemisphere
INTACT_C = "#4bc8ff"   # blue – intact right hemisphere
ACCENT   = "#f9a825"

for ax in [AX_TITLE, AX_CST, AX_ASYM, AX_HEAT, AX_MOTOR]:
    ax.set_facecolor("#1a1e2e")
    for spine in ax.spines.values():
        spine.set_edgecolor("#3a3f5c")
    ax.tick_params(colors=LIGHT, labelsize=8)
    ax.xaxis.label.set_color(LIGHT)
    ax.yaxis.label.set_color(LIGHT)

# ── Panel 0: explanatory diagram ─────────────────────────────────────────────
AX_TITLE.set_xlim(0, 1)
AX_TITLE.set_ylim(0, 1)
AX_TITLE.axis("off")

# Title
AX_TITLE.text(
    0.5, 0.92,
    "TVB Motor Cortex Phase 1 — Stroke Simulation Results",
    ha="center", va="top", fontsize=15, fontweight="bold", color=LIGHT,
    transform=AX_TITLE.transAxes,
)

# Pipeline boxes
boxes = [
    (0.04, "DTI\nConnectome\n(68 regions)", "#2e3b6e"),
    (0.22, "Wong-Wang\nNeural Mass\nModel", "#2e3b6e"),
    (0.40, f"Stroke:\nLeft M1\n(-70% excit.)", "#6e2e2e"),
    (0.58, "Cortical\nFiring Rates\n[Hz]", "#2e3b6e"),
    (0.76, "CST Drive\nLeft / Right\n[Hz]", "#1a4a2e"),
    (0.93, "→ tinyCPG\nSpinal CPG\nInput", "#2a4a1e"),
]
for x0, label, col in boxes:
    rect = FancyBboxPatch(
        (x0 - 0.08, 0.08), 0.15, 0.72,
        boxstyle="round,pad=0.02", linewidth=1.2,
        edgecolor="#5a6090", facecolor=col,
        transform=AX_TITLE.transAxes, clip_on=False,
    )
    AX_TITLE.add_patch(rect)
    AX_TITLE.text(
        x0, 0.44, label, ha="center", va="center",
        fontsize=7.5, color=LIGHT, transform=AX_TITLE.transAxes,
        fontweight="bold", linespacing=1.4,
    )

# Arrows between boxes
for x0, x1 in [(0.12, 0.22), (0.30, 0.40), (0.48, 0.58), (0.66, 0.76), (0.85, 0.93)]:
    AX_TITLE.annotate(
        "", xy=(x1 - 0.07, 0.44), xytext=(x0 + 0.01, 0.44),
        xycoords="axes fraction", textcoords="axes fraction",
        arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1.5),
    )

# Patient / stroke summary
info = (
    f"Patient: {attrs.get('patient_id','?')}  |  "
    f"Simulation: {attrs.get('simulation_duration_ms',0)/1000:.0f} s  |  "
    f"Stroke: LEFT M1  |  Excitability: {attrs.get('stroke_excitability_factor',0)*100:.0f}% of normal  |  "
    f"Regions: 68 (Desikan-Killiany atlas)"
)
AX_TITLE.text(
    0.5, 0.01, info, ha="center", va="bottom", fontsize=8,
    color="#a0a8c0", transform=AX_TITLE.transAxes,
)

# ── Panel 1: CST drive time series ───────────────────────────────────────────
AX_CST.plot(t, cst_l, color=STROKE_C, lw=1.4, label="Left CST  (stroke side)")
AX_CST.plot(t, cst_r, color=INTACT_C, lw=1.4, label="Right CST (intact side)", alpha=0.9)
AX_CST.fill_between(t, cst_l, cst_r, where=(cst_r > cst_l),
                     color=INTACT_C, alpha=0.08)

AX_CST.set_xlabel("Time [s]", fontsize=9)
AX_CST.set_ylabel("Corticospinal Tract Drive [Hz]", fontsize=9)
AX_CST.set_title(
    "Corticospinal Tract Drive — Left vs Right",
    fontsize=10, color=LIGHT, pad=6,
)
AX_CST.legend(fontsize=8, facecolor="#1a1e2e", edgecolor="#3a3f5c",
               labelcolor=LIGHT, loc="upper right")
AX_CST.set_xlim(t[0], t[-1])

# Annotate the asymmetry
mean_l = cst_left_hz.mean()
mean_r = cst_right_hz.mean()
asym   = (mean_r - mean_l) / mean_r * 100
AX_CST.axhline(mean_l, color=STROKE_C, lw=0.8, ls="--", alpha=0.5)
AX_CST.axhline(mean_r, color=INTACT_C, lw=0.8, ls="--", alpha=0.5)
AX_CST.text(
    t[-1] * 0.02, mean_l + 0.02, f"μ = {mean_l:.2f} Hz",
    color=STROKE_C, fontsize=7.5,
)
AX_CST.text(
    t[-1] * 0.02, mean_r + 0.02, f"μ = {mean_r:.2f} Hz",
    color=INTACT_C, fontsize=7.5,
)

note = (
    "The corticospinal tract (CST) carries motor commands\n"
    "from cortex → spinal cord. LEFT CST is reduced because\n"
    "the stroke damaged left M1's excitability by 70%."
)
AX_CST.text(
    0.98, 0.10, note, transform=AX_CST.transAxes,
    ha="right", va="bottom", fontsize=7, color="#a0a8c0",
    bbox=dict(fc="#1a1e2e", ec="#3a3f5c", pad=3, lw=0.8),
)

# ── Panel 2: Asymmetry bar ────────────────────────────────────────────────────
labels_asym = ["Left\n(stroke)", "Right\n(intact)"]
vals_asym   = [mean_l, mean_r]
colors_asym = [STROKE_C, INTACT_C]
bars = AX_ASYM.bar(labels_asym, vals_asym, color=colors_asym, width=0.5,
                    edgecolor="#3a3f5c", linewidth=0.8)
for bar, v in zip(bars, vals_asym):
    AX_ASYM.text(
        bar.get_x() + bar.get_width() / 2, v + 0.01,
        f"{v:.3f} Hz", ha="center", va="bottom", fontsize=9,
        color=LIGHT, fontweight="bold",
    )

AX_ASYM.set_ylim(0, max(vals_asym) * 1.35)
AX_ASYM.set_ylabel("Mean CST Drive [Hz]", fontsize=9)
AX_ASYM.set_title("Hemispheric Asymmetry", fontsize=10, color=LIGHT, pad=6)
AX_ASYM.text(
    0.5, 0.82,
    f"Asymmetry:\n{asym:.1f}% deficit\n(stroke side)",
    transform=AX_ASYM.transAxes, ha="center", va="center",
    fontsize=8.5, color=ACCENT, fontweight="bold",
    bbox=dict(fc="#2a2a1e", ec=ACCENT, pad=4, lw=1),
)

# ── Panel 3: Heatmap — all 68 regions over time ──────────────────────────────
# Subsample to ~500 timepoints for display
ds2  = max(1, fr.shape[1] // 500)
fr2  = fr[:, ::ds2]
t2   = t[::ds2]

im = AX_HEAT.imshow(
    fr2, aspect="auto", origin="lower",
    extent=[t2[0], t2[-1], 0, n_regions],
    cmap="inferno", vmin=0, vmax=mean_fr.max() * 1.1,
    interpolation="nearest",
)
cbar = fig.colorbar(im, ax=AX_HEAT, pad=0.01, fraction=0.03)
cbar.set_label("Firing Rate [Hz]", color=LIGHT, fontsize=8)
cbar.ax.tick_params(colors=LIGHT, labelsize=7)

# Mark motor regions
for name, idx in MOTOR.items():
    side_col = STROKE_C if name.endswith("_L") else INTACT_C
    AX_HEAT.axhline(idx, color=side_col, lw=0.6, alpha=0.7, ls="--")
    AX_HEAT.text(
        t2[-1] * 1.01, idx, name, color=side_col,
        fontsize=5.5, va="center", clip_on=False,
    )

AX_HEAT.set_xlabel("Time [s]", fontsize=9)
AX_HEAT.set_ylabel("Cortical Region Index", fontsize=9)
AX_HEAT.set_title(
    "Firing Rate Heatmap — All 68 Cortical Regions Over Time",
    fontsize=10, color=LIGHT, pad=6,
)
AX_HEAT.set_xlim(t2[0], t2[-1])
AX_HEAT.set_ylim(0, n_regions)

note2 = (
    "Each row = one of 68 Desikan-Killiany regions.\n"
    "Brighter = higher firing rate. Dashed lines mark\n"
    "motor regions (red=left/stroke, blue=right/intact)."
)
AX_HEAT.text(
    0.02, 0.97, note2, transform=AX_HEAT.transAxes,
    ha="left", va="top", fontsize=6.5, color="#a0a8c0",
    bbox=dict(fc="#1a1e2e", ec="#3a3f5c", pad=2, lw=0.6),
)

# ── Panel 4: Motor region mean firing rates ───────────────────────────────────
pair_labels = [l for l, r in MOTOR_PAIRS]
left_means  = [mean_fr[MOTOR[l]] for l, r in MOTOR_PAIRS]
right_means = [mean_fr[MOTOR[r]] for l, r in MOTOR_PAIRS]

x = np.arange(len(MOTOR_PAIRS))
w = 0.35
b1 = AX_MOTOR.bar(x - w/2, left_means,  w, color=STROKE_C, label="Left (stroke)",
                   edgecolor="#3a3f5c", lw=0.8)
b2 = AX_MOTOR.bar(x + w/2, right_means, w, color=INTACT_C, label="Right (intact)",
                   edgecolor="#3a3f5c", lw=0.8, alpha=0.9)

for bar, v in list(zip(b1, left_means)) + list(zip(b2, right_means)):
    AX_MOTOR.text(
        bar.get_x() + bar.get_width() / 2, v + 0.02,
        f"{v:.2f}", ha="center", va="bottom", fontsize=6.5, color=LIGHT,
    )

full_names = [MOTOR_FULL[l.replace("_L", "")] for l, r in MOTOR_PAIRS]
AX_MOTOR.set_xticks(x)
AX_MOTOR.set_xticklabels(full_names, fontsize=7, rotation=15, ha="right")
AX_MOTOR.set_ylabel("Mean Firing Rate [Hz]", fontsize=9)
AX_MOTOR.set_title("Motor Region Firing Rates\n(Left vs Right)", fontsize=10,
                    color=LIGHT, pad=6)
AX_MOTOR.legend(fontsize=7, facecolor="#1a1e2e", edgecolor="#3a3f5c",
                 labelcolor=LIGHT, loc="upper left")
AX_MOTOR.set_ylim(0, max(max(left_means), max(right_means)) * 1.45)

# Highlight asymmetric M1 pair
for i, (lv, rv) in enumerate(zip(left_means, right_means)):
    if rv > 0:
        asym_i = (rv - lv) / rv * 100
        AX_MOTOR.text(
            x[i], max(lv, rv) * 1.15,
            f"Δ{asym_i:.0f}%",
            ha="center", fontsize=6, color=ACCENT, fontweight="bold",
        )

# ── super-title and save ──────────────────────────────────────────────────────
fig.suptitle(
    "TVB Motor Cortex Stroke Simulation — Phase 1 Results",
    fontsize=14, fontweight="bold", color=LIGHT, y=0.975,
)

outpath = Path("tvb_output/phase1/phase1_results.png")
outpath.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(outpath, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"Saved: {outpath}")
plt.close()
