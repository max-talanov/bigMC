#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TVB Motor Cortex — Healthy vs Stroke Paired Simulation
=======================================================

Runs two simulations on the same synthetic patient connectome:
  1. HEALTHY  — intact connectivity and uniform background current (I_o)
  2. STROKE   — left M1: connectivity cut by 60%, I_o reduced to 30% of normal

The excitability reduction is now applied per-region by setting a spatially
varying I_o array before the simulator is configured, correctly encoding the
neural silence caused by infarction.

Outputs
-------
  tvb_output/comparison/
    healthy_PATIENT_001_<ts>.h5
    stroke_PATIENT_001_<ts>.h5
    healthy_vs_stroke_comparison.png
"""

import sys
import copy
import logging
import h5py
import numpy as np
from pathlib import Path
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm
from scipy.stats import pearsonr

# ── import simulation infrastructure from Phase 1 ────────────────────────────
from tvb_motor_cortex_phase1 import (
    PatientData, SimulationConfig, StrokeParameters,
    TVBMotorCortexModel, MotorRegionIndices,
)

# ── logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.WARNING)   # suppress TVB noise
log = logging.getLogger("HealthyVsStroke")
log.setLevel(logging.INFO)
_h = logging.StreamHandler()
_h.setFormatter(logging.Formatter("[%(name)s] %(message)s"))
log.addHandler(_h)

OUTPUT_DIR = Path("./tvb_output/comparison")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MOTOR = MotorRegionIndices()
MOTOR_PAIRS = [
    ("M1",  MOTOR.M1_left,  MOTOR.M1_right),
    ("PMC", MOTOR.PMC_left, MOTOR.PMC_right),
    ("SMA", MOTOR.SMA_left, MOTOR.SMA_right),
    ("S1",  MOTOR.S1_left,  MOTOR.S1_right),
]


# ── subclass: adds per-region I_o (excitability) reduction for stroke ─────────
class TVBMotorCortexModelWithExcitability(TVBMotorCortexModel):
    """
    Extends the base model so that stroke regions have a spatially varying
    background current (I_o). In the Reduced Wong-Wang model, I_o sets the
    resting-state drive to each neural population. Reducing it in a specific
    region captures the hypo-excitability caused by ischaemic damage without
    fully disconnecting the region.
    """

    def setup_neuron_model(self):
        model = super().setup_neuron_model()

        if self.stroke_params is None or not self.stroke_params.is_lesioned:
            return model   # healthy: uniform I_o (already set by super)

        # Build spatially varying I_o: (n_regions,)
        n = self.patient.n_regions
        I_o_uniform = float(model.I_o[0])
        I_o_spatial = np.full(n, I_o_uniform)

        # Identify stroke target region
        ri = MotorRegionIndices()
        if self.stroke_params.lesion_primary_region == "M1":
            target_idx = (ri.M1_left if self.stroke_params.lesion_side == "left"
                          else ri.M1_right)
        elif self.stroke_params.lesion_primary_region == "S1":
            target_idx = (ri.S1_left if self.stroke_params.lesion_side == "left"
                          else ri.S1_right)
        else:
            return model   # region not handled → keep uniform

        I_o_spatial[target_idx] *= self.stroke_params.excitability_factor
        model.I_o = I_o_spatial          # TVB accepts (n_regions,) array
        log.info(f"Excitability reduction applied: region {target_idx} "
                 f"I_o → {I_o_spatial[target_idx]:.4f} nA "
                 f"(factor {self.stroke_params.excitability_factor})")
        return model


# ── simulation runner ─────────────────────────────────────────────────────────
def run_condition(condition: str, patient_id: str = "PATIENT_001") -> dict:
    """
    Run one simulation condition ("healthy" or "stroke").
    Returns a dict with results + metadata.
    """
    assert condition in ("healthy", "stroke")
    log.info(f"─── Running {condition.upper()} condition ───")

    config = SimulationConfig(
        simulation_duration_ms=10_000.0,
        dt_ms=0.1,
        output_dir=str(OUTPUT_DIR),
        output_filename_prefix=f"tvb_{condition}",
        monitor_bold=False,
        monitor_spike_rate=True,
    )

    # Both conditions use the SAME synthetic connectome (same random seed)
    patient = PatientData.create_placeholder(patient_id, atlas="DK68")

    if condition == "stroke":
        stroke_params = StrokeParameters(
            lesion_side="left",
            lesion_primary_region="M1",
            excitability_factor=0.3,          # I_o reduced to 30%
            connectivity_reduction_factor=0.4, # DTI weights cut to 40%
            days_post_stroke=7,
        )
        patient.apply_lesion(stroke_params)
    else:
        stroke_params = None

    tvb_model = TVBMotorCortexModelWithExcitability(patient, config, stroke_params)
    tvb_model.build_simulator()

    results = tvb_model.run_simulation()
    output_file = tvb_model.save_results(results)
    stats = tvb_model.summary_statistics(results)

    log.info(f"  CST Left  : {results['corticospinal_drive_left_hz'].mean():.4f} Hz")
    log.info(f"  CST Right : {results['corticospinal_drive_right_hz'].mean():.4f} Hz")
    log.info(f"  Saved     : {output_file}")

    return {
        "condition":  condition,
        "results":    results,
        "stats":      stats,
        "file":       output_file,
        "patient_id": patient_id,
    }


# ── comparison plot ───────────────────────────────────────────────────────────
def make_comparison_plot(healthy: dict, stroke: dict, out_path: Path) -> None:
    """
    6-panel figure comparing healthy and stroke simulations.
    """
    hr = healthy["results"]
    sr = stroke["results"]

    t  = hr["time_s"]
    ds = max(1, len(t) // 2000)
    t_plot = t[::ds]

    hl = hr["corticospinal_drive_left_hz"][::ds]
    hr_ = hr["corticospinal_drive_right_hz"][::ds]
    sl = sr["corticospinal_drive_left_hz"][::ds]
    sright = sr["corticospinal_drive_right_hz"][::ds]

    hfr = hr["firing_rates_hz"]                    # (68, 100000)
    sfr = sr["firing_rates_hz"]

    hfr_mean = hfr.mean(axis=1)                    # (68,)
    sfr_mean = sfr.mean(axis=1)
    diff_mean = sfr_mean - hfr_mean                # signed difference

    # ── figure layout ────────────────────────────────────────────────────────
    DARK  = "#0e1117"
    LIGHT = "#e0e0e0"
    H_COL = "#4bc8ff"    # healthy  blue
    S_COL = "#ff4b4b"    # stroke   red
    INTACT_COL = "#4bc8ff"
    ACCENT = "#f9a825"
    DIFF_COL = "#a855f7"  # purple for difference

    fig = plt.figure(figsize=(19, 15))
    fig.patch.set_facecolor(DARK)
    gs = gridspec.GridSpec(
        3, 3, figure=fig,
        left=0.07, right=0.97, top=0.93, bottom=0.07,
        hspace=0.50, wspace=0.38,
    )

    axes = {
        "cst":    fig.add_subplot(gs[0, :2]),   # CST time series
        "asym":   fig.add_subplot(gs[0, 2]),    # asymmetry summary
        "heat_h": fig.add_subplot(gs[1, 0]),    # healthy heatmap
        "heat_s": fig.add_subplot(gs[1, 1]),    # stroke  heatmap
        "heat_d": fig.add_subplot(gs[1, 2]),    # difference heatmap
        "motor":  fig.add_subplot(gs[2, :2]),   # motor region bar chart
        "metrics":fig.add_subplot(gs[2, 2]),    # scalar metrics table
    }

    for ax in axes.values():
        ax.set_facecolor("#1a1e2e")
        for s in ax.spines.values():
            s.set_edgecolor("#3a3f5c")
        ax.tick_params(colors=LIGHT, labelsize=8)
        ax.xaxis.label.set_color(LIGHT)
        ax.yaxis.label.set_color(LIGHT)

    # ── 1. CST drive time series ─────────────────────────────────────────────
    ax = axes["cst"]
    ax.plot(t_plot, hl,    color=H_COL,    lw=1.4, ls="--", label="Healthy – Left")
    ax.plot(t_plot, hr_,   color=H_COL,    lw=1.4, ls="-",  label="Healthy – Right", alpha=0.7)
    ax.plot(t_plot, sl,    color=S_COL,    lw=1.4, ls="--", label="Stroke  – Left (damaged)")
    ax.plot(t_plot, sright,color=INTACT_COL, lw=1.4, ls=":",  label="Stroke  – Right (intact)", alpha=0.7)

    ax.fill_between(t_plot, sl, hl, where=(hl > sl), color=S_COL, alpha=0.10,
                    label="Left deficit")

    ax.set_xlabel("Time [s]", fontsize=9)
    ax.set_ylabel("CST Drive [Hz]", fontsize=9)
    ax.set_title("Corticospinal Tract Drive — Healthy vs Stroke", fontsize=10,
                 color=LIGHT, pad=6)
    ax.legend(fontsize=7, facecolor="#1a1e2e", edgecolor="#3a3f5c",
               labelcolor=LIGHT, ncol=2, loc="upper right")
    ax.set_xlim(t_plot[0], t_plot[-1])

    hl_m = hr["corticospinal_drive_left_hz"].mean()
    hr_m = hr["corticospinal_drive_right_hz"].mean()
    sl_m = sr["corticospinal_drive_left_hz"].mean()
    sr_m = sr["corticospinal_drive_right_hz"].mean()
    deficit_pct = (hl_m - sl_m) / hl_m * 100 if hl_m else 0

    note = (
        f"Healthy  L={hl_m:.3f} Hz  R={hr_m:.3f} Hz\n"
        f"Stroke   L={sl_m:.3f} Hz  R={sr_m:.3f} Hz\n"
        f"Left CST deficit: {deficit_pct:.1f}%"
    )
    ax.text(0.02, 0.05, note, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=7.5, color=ACCENT,
            bbox=dict(fc="#1a1e2e", ec="#3a3f5c", pad=3, lw=0.8))

    # ── 2. Asymmetry bar ─────────────────────────────────────────────────────
    ax = axes["asym"]
    conditions = ["Healthy\nL", "Healthy\nR", "Stroke\nL", "Stroke\nR"]
    vals  = [hl_m, hr_m, sl_m, sr_m]
    cols  = [H_COL, H_COL, S_COL, INTACT_COL]
    brs   = ax.bar(conditions, vals, color=cols, width=0.55,
                   edgecolor="#3a3f5c", lw=0.8)
    for bar, v in zip(brs, vals):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.005,
                f"{v:.3f}", ha="center", va="bottom", fontsize=7.5,
                color=LIGHT, fontweight="bold")

    # Annotate healthy asymmetry and stroke asymmetry
    h_asym = abs(hl_m - hr_m) / max(hl_m, hr_m) * 100
    s_asym = abs(sl_m - sr_m) / max(sl_m, sr_m) * 100
    ax.text(0.25, 0.90, f"Healthy\nAsym: {h_asym:.1f}%", transform=ax.transAxes,
            ha="center", fontsize=7, color=H_COL,
            bbox=dict(fc="#1a1e2e", ec=H_COL, pad=2, lw=0.7))
    ax.text(0.75, 0.90, f"Stroke\nAsym: {s_asym:.1f}%", transform=ax.transAxes,
            ha="center", fontsize=7, color=S_COL,
            bbox=dict(fc="#1a1e2e", ec=S_COL, pad=2, lw=0.7))

    ax.set_ylim(0, max(vals) * 1.5)
    ax.set_ylabel("Mean CST Drive [Hz]", fontsize=9)
    ax.set_title("Hemispheric Asymmetry\nHealthy vs Stroke", fontsize=10,
                 color=LIGHT, pad=6)

    # ── 3-5. Firing-rate heatmaps ─────────────────────────────────────────────
    ds2 = max(1, hfr.shape[1] // 400)
    n_reg = hfr.shape[0]

    vmax = max(hfr_mean.max(), sfr_mean.max()) * 1.05
    vmax_full = max(hfr[:, ::ds2].max(), sfr[:, ::ds2].max())

    for ax_key, data, title, cmap, norm in [
        ("heat_h", hfr[:, ::ds2], "Healthy — All 68 Regions", "inferno",
         None),
        ("heat_s", sfr[:, ::ds2], "Stroke  — All 68 Regions", "inferno",
         None),
        ("heat_d", (sfr - hfr)[:, ::ds2], "Difference  (Stroke − Healthy)", "RdBu_r",
         TwoSlopeNorm(
             vmin=min((sfr-hfr).min(), -1e-6),
             vcenter=0,
             vmax=max((sfr-hfr).max(),  1e-6),
         )),
    ]:
        ax = axes[ax_key]
        t2 = t[::ds2]
        kw = dict(aspect="auto", origin="lower",
                  extent=[t2[0], t2[-1], 0, n_reg],
                  interpolation="nearest")
        if norm is not None:
            im = ax.imshow(data, cmap=cmap, norm=norm, **kw)
        else:
            im = ax.imshow(data, cmap=cmap, vmin=0, vmax=vmax_full, **kw)
        cb = fig.colorbar(im, ax=ax, pad=0.01, fraction=0.04)
        cb.ax.tick_params(colors=LIGHT, labelsize=6)
        cb.set_label("Hz" if "Diff" not in title else "ΔHz",
                     color=LIGHT, fontsize=7)

        # Mark motor regions
        for name, l_idx, r_idx in MOTOR_PAIRS:
            ax.axhline(l_idx, color=S_COL, lw=0.5, ls="--", alpha=0.7)
            ax.axhline(r_idx, color=INTACT_COL, lw=0.5, ls="--", alpha=0.7)

        ax.set_xlabel("Time [s]", fontsize=8)
        ax.set_ylabel("Region index", fontsize=8)
        ax.set_title(title, fontsize=8.5, color=LIGHT, pad=5)
        ax.set_xlim(t2[0], t2[-1])
        ax.set_ylim(0, n_reg)

    # ── 6. Motor region bar chart ─────────────────────────────────────────────
    ax = axes["motor"]
    x = np.arange(len(MOTOR_PAIRS))
    w = 0.20

    bars = [
        (x - 1.5*w, [hfr_mean[l] for _, l, _ in MOTOR_PAIRS], H_COL,    "Healthy Left"),
        (x - 0.5*w, [hfr_mean[r] for _, _, r in MOTOR_PAIRS], H_COL,    "Healthy Right"),
        (x + 0.5*w, [sfr_mean[l] for _, l, _ in MOTOR_PAIRS], S_COL,    "Stroke Left"),
        (x + 1.5*w, [sfr_mean[r] for _, _, r in MOTOR_PAIRS], INTACT_COL,"Stroke Right"),
    ]
    for xpos, vals2, col, label in bars:
        b = ax.bar(xpos, vals2, w, color=col, label=label,
                   edgecolor="#3a3f5c", lw=0.6,
                   alpha=0.85 if "Right" in label else 1.0)

    region_full = {"M1": "Primary\nMotor", "PMC": "Pre-\nmotor",
                   "SMA": "Suppl.\nMotor", "S1": "Primary\nSensory"}
    ax.set_xticks(x)
    ax.set_xticklabels([region_full[n] for n, *_ in MOTOR_PAIRS], fontsize=8)
    ax.set_ylabel("Mean Firing Rate [Hz]", fontsize=9)
    ax.set_title("Motor Region Firing Rates: Healthy vs Stroke", fontsize=10,
                 color=LIGHT, pad=6)
    ax.legend(fontsize=7, facecolor="#1a1e2e", edgecolor="#3a3f5c",
               labelcolor=LIGHT, ncol=2, loc="upper right")

    all_vals = [v for _, vs, *_ in bars for v in vs]
    ax.set_ylim(0, max(all_vals) * 1.45)

    # Mark stroke M1 deficit
    m1_h_l = hfr_mean[MOTOR.M1_left]
    m1_s_l = sfr_mean[MOTOR.M1_left]
    m1_deficit = (m1_h_l - m1_s_l) / m1_h_l * 100 if m1_h_l else 0
    ax.annotate(
        f"M1 left\ndeficit\n{m1_deficit:.0f}%",
        xy=(x[0] + 0.5*w, m1_s_l),
        xytext=(x[0] + 0.5*w + 0.6, m1_s_l + (max(all_vals)*0.25)),
        fontsize=7, color=ACCENT, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1.0),
    )

    # ── 7. Scalar metrics table ───────────────────────────────────────────────
    ax = axes["metrics"]
    ax.axis("off")

    # Compute metrics
    lat_idx  = MOTOR.M1_left
    cst_deficit   = (hl_m - sl_m) / hl_m * 100 if hl_m else 0
    cst_asym_h    = (hr_m - hl_m) / hr_m * 100 if hr_m else 0
    cst_asym_s    = (sr_m - sl_m) / sr_m * 100 if sr_m else 0
    m1l_deficit   = (hfr_mean[MOTOR.M1_left] - sfr_mean[MOTOR.M1_left]) / hfr_mean[MOTOR.M1_left] * 100
    pmc_deficit   = (hfr_mean[MOTOR.PMC_left] - sfr_mean[MOTOR.PMC_left]) / hfr_mean[MOTOR.PMC_left] * 100
    sma_deficit   = (hfr_mean[MOTOR.SMA_left] - sfr_mean[MOTOR.SMA_left]) / hfr_mean[MOTOR.SMA_left] * 100

    # Cross-hemisphere correlation
    h_lr_corr, _ = pearsonr(hfr_mean[:34], hfr_mean[34:]) if len(hfr_mean) >= 68 else (0, 1)
    s_lr_corr, _ = pearsonr(sfr_mean[:34], sfr_mean[34:]) if len(sfr_mean) >= 68 else (0, 1)

    rows = [
        ("Metric", "Healthy", "Stroke"),
        ("─"*16, "─"*7, "─"*7),
        ("CST Left [Hz]",  f"{hl_m:.3f}",  f"{sl_m:.3f}"),
        ("CST Right [Hz]", f"{hr_m:.3f}",  f"{sr_m:.3f}"),
        ("CST Deficit (L)", "—",           f"{cst_deficit:.1f}%"),
        ("CST Asymmetry",  f"{cst_asym_h:.1f}%", f"{cst_asym_s:.1f}%"),
        ("─"*16, "─"*7, "─"*7),
        ("M1-L firing [Hz]",
         f"{hfr_mean[MOTOR.M1_left]:.3f}",
         f"{sfr_mean[MOTOR.M1_left]:.3f}"),
        ("M1-L deficit", "—", f"{m1l_deficit:.1f}%"),
        ("PMC-L deficit", "—", f"{pmc_deficit:.1f}%"),
        ("SMA-L deficit", "—", f"{sma_deficit:.1f}%"),
        ("─"*16, "─"*7, "─"*7),
        ("L/R FR corr", f"{h_lr_corr:.3f}", f"{s_lr_corr:.3f}"),
    ]

    y0 = 0.98
    dy = 0.072
    col_x = [0.01, 0.56, 0.80]
    col_c = [LIGHT, H_COL, S_COL]

    ax.set_title("Summary Metrics", fontsize=10, color=LIGHT, pad=6)
    for i, row in enumerate(rows):
        for j, (cell, cx, cc) in enumerate(zip(row, col_x, col_c)):
            fw = "bold" if i in (0, 1, 7, 12) else "normal"
            ax.text(cx, y0 - i * dy, cell,
                    transform=ax.transAxes,
                    fontsize=7.5, color=cc, fontweight=fw, va="top")

    # ── super-title ───────────────────────────────────────────────────────────
    fig.suptitle(
        "TVB Motor Cortex — Healthy vs Left-M1 Stroke Paired Simulation",
        fontsize=14, fontweight="bold", color=LIGHT, y=0.975,
    )

    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    log.info(f"Figure saved: {out_path}")
    plt.close()


# ── entry point ───────────────────────────────────────────────────────────────
def main():
    print("=" * 62)
    print("  TVB Healthy vs Stroke Paired Simulation")
    print("=" * 62)

    print("\n[1/4] Running HEALTHY condition...")
    healthy = run_condition("healthy")

    print("\n[2/4] Running STROKE condition...")
    stroke  = run_condition("stroke")

    print("\n[3/4] Computing comparison metrics...")
    hr = healthy["results"]
    sr = stroke["results"]
    hl_m = hr["corticospinal_drive_left_hz"].mean()
    hr_m = hr["corticospinal_drive_right_hz"].mean()
    sl_m = sr["corticospinal_drive_left_hz"].mean()
    sr_m = sr["corticospinal_drive_right_hz"].mean()

    print(f"\n  {'Metric':<30} {'Healthy':>10} {'Stroke':>10} {'Δ':>10}")
    print(f"  {'─'*60}")
    print(f"  {'CST Left [Hz]':<30} {hl_m:>10.4f} {sl_m:>10.4f} {sl_m-hl_m:>+10.4f}")
    print(f"  {'CST Right [Hz]':<30} {hr_m:>10.4f} {sr_m:>10.4f} {sr_m-hr_m:>+10.4f}")
    print(f"  {'CST Left deficit':<30} {'—':>10} {(hl_m-sl_m)/hl_m*100:>9.1f}% {'':>10}")
    print(f"  {'CST Asymmetry (L/R)':<30} "
          f"{abs(hl_m-hr_m)/max(hl_m,hr_m)*100:>9.1f}% "
          f"{abs(sl_m-sr_m)/max(sl_m,sr_m)*100:>9.1f}% {'':>10}")

    hfr = hr["firing_rates_hz"].mean(axis=1)
    sfr = sr["firing_rates_hz"].mean(axis=1)
    print(f"\n  Motor region mean firing rates:")
    for name, li, ri in MOTOR_PAIRS:
        print(f"    {name:<6} L: H={hfr[li]:.3f} S={sfr[li]:.3f} "
              f"({(sfr[li]-hfr[li])/hfr[li]*100:+.1f}%)   "
              f"R: H={hfr[ri]:.3f} S={sfr[ri]:.3f} "
              f"({(sfr[ri]-hfr[ri])/hfr[ri]*100:+.1f}%)")

    print("\n[4/4] Generating comparison plot...")
    fig_path = OUTPUT_DIR / "healthy_vs_stroke_comparison.png"
    make_comparison_plot(healthy, stroke, fig_path)

    print(f"\n{'='*62}")
    print(f"  Done.")
    print(f"  Healthy HDF5 : {healthy['file']}")
    print(f"  Stroke  HDF5 : {stroke['file']}")
    print(f"  Figure       : {fig_path}")
    print(f"{'='*62}")


if __name__ == "__main__":
    main()
