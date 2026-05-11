#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TVB Phase 3 — Gait Pattern Gallery
====================================

Shows the actual left/right extensor force waveforms at every key
rehabilitation week so you can *see* the walking pattern becoming
symmetric step by step.

Layout
------
  • Two large panels side-by-side (Left leg | Right leg)
  • Each panel is a vertical stack of N_SHOW_WEEKS sub-traces
  • Each sub-trace shows a 6-second steady-state window of extensor force
  • Colour fades from red (acute, week 0) → teal (full recovery, week 12)
  • A translucent fill between the L and R traces in each row shows the
    force-deficit gap; the fill vanishes as recovery progresses
  • Asymmetry index (AmpAI %) annotated on every row

Additionally, a compact "overlay" panel underneath shows all weeks
simultaneously on one axis to highlight the convergence.

Run with:  python3 tvb_phase3_gait_gallery.py
Output  :  tvb_output/phase3/phase3_gait_gallery.png
"""

import numpy as np
import h5py
import importlib.util
from pathlib import Path
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from scipy.signal import butter, filtfilt, find_peaks

# ── Re-use Phase 2 CPG simulator ─────────────────────────────────────────────
_spec = importlib.util.spec_from_file_location(
    "p2", Path(__file__).parent / "tvb_phase2_spinal_cpg.py")
_p2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_p2)
simulate           = _p2.simulate
load_phase1_scales = _p2.load_phase1_scales
C0                 = _p2.C0

OUTPUT_DIR = Path("./tvb_output/phase3")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Key weeks to show ─────────────────────────────────────────────────────────
KEY_WEEKS   = [0, 2, 4, 6, 8, 10, 12]
SIM_S       = 25.0      # simulate 25 s so we have plenty of steady-state
DT_S        = 0.001
SHOW_WIN_S  = 6.0       # seconds of waveform to display per row
LPF_HZ      = 8.0       # low-pass for display smoothing

# ── Full-recovery logistic ─────────────────────────────────────────────────────
def logistic(t, k, t0):
    return 1.0 / (1.0 + np.exp(-k * (t - t0)))

def recovery_fraction_at_week(wk, all_weeks=np.linspace(0,12,13)):
    r_raw = logistic(all_weeks, k=0.7, t0=4.0)
    r     = (r_raw - r_raw[0]) / (r_raw[-1] - r_raw[0])
    idx   = np.argmin(np.abs(all_weeks - wk))
    return float(np.clip(r[idx], 0, 1))


def peak_amplitude(t, force, min_height=0.5):
    dt = float(np.diff(t[:10]).mean())
    half = len(force) // 2
    b, a = butter(2, LPF_HZ * dt * 2, btype="low")
    fs   = filtfilt(b, a, force[half:])
    pk, _= find_peaks(fs, height=min_height, distance=max(1, int(0.3/dt)))
    return float(fs[pk].mean()) if len(pk) >= 1 else float(fs.max())


# ══════════════════════════════════════════════════════════════════════════════
#  SIMULATE ALL KEY WEEKS
# ══════════════════════════════════════════════════════════════════════════════
def run_gallery_sims(c_stroke_L, c_healthy_L, c_R):
    all_weeks = np.linspace(0, 12, 13)
    results   = {}
    print(f"  Simulating {len(KEY_WEEKS)} key weeks …")
    for wk in KEY_WEEKS:
        r   = recovery_fraction_at_week(wk, all_weeks)
        c_L = c_stroke_L + r * (c_healthy_L - c_stroke_L)
        deficit = (1 - c_L / c_healthy_L) * 100
        print(f"    week {wk:2d}  r={r:.2f}  c_L={c_L:.3f}  "
              f"deficit={deficit:.1f}%")
        res = simulate(c_L, c_R, sim_s=SIM_S, dt_s=DT_S)
        aL  = peak_amplitude(res["time_s"], res["force_LE"])
        aR  = peak_amplitude(res["time_s"], res["force_RE"])
        amp_ai = (aR - aL) / max(aR, aL) if max(aR, aL) > 0 else 0.0
        results[wk] = dict(res=res, c_L=c_L, r=r,
                           amp_ai=amp_ai, amp_L=aL, amp_R=aR)
    return results


# ══════════════════════════════════════════════════════════════════════════════
#  GALLERY FIGURE
# ══════════════════════════════════════════════════════════════════════════════
def make_gallery(gallery: dict, c_stroke_L: float, c_R: float,
                 out_path: Path):
    DARK  = "#0e1117"
    LIGHT = "#e0e0e0"
    ACCENT = "#f9a825"

    N = len(KEY_WEEKS)

    # Colour ramp: red (acute) → teal/blue (recovered)
    CMAP = LinearSegmentedColormap.from_list(
        "rehab", ["#ff3333", "#ff9933", "#ffdd33",
                  "#66dd44", "#33ccaa", "#33aaff"])
    week_cols = {wk: CMAP(i / (N - 1)) for i, wk in enumerate(KEY_WEEKS)}

    # ── Layout ────────────────────────────────────────────────────────────────
    # Rows: one per week.   Cols: [Left-leg waveform | Right-leg waveform]
    # Plus a wide "convergence overlay" strip at the bottom.

    fig = plt.figure(figsize=(18, 3.5 * N + 3.5))
    fig.patch.set_facecolor(DARK)

    outer = gridspec.GridSpec(
        2, 1, figure=fig,
        height_ratios=[N * 3.5, 3.0],
        hspace=0.12,
        left=0.07, right=0.97, top=0.95, bottom=0.04)

    # Top block: N rows × 2 cols
    gs_top = gridspec.GridSpecFromSubplotSpec(
        N, 2, subplot_spec=outer[0],
        hspace=0.08, wspace=0.07)

    # Bottom block: 1 row × 2 cols (overlay)
    gs_bot = gridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=outer[1],
        wspace=0.07)

    ax_L_ov = fig.add_subplot(gs_bot[0, 0])   # overlay Left
    ax_R_ov = fig.add_subplot(gs_bot[0, 1])   # overlay Right

    row_axs = []   # (ax_L, ax_R) per week row
    for i in range(N):
        ax_L = fig.add_subplot(gs_top[i, 0])
        ax_R = fig.add_subplot(gs_top[i, 1])
        row_axs.append((ax_L, ax_R))

    def style(ax):
        ax.set_facecolor("#12172a")
        for sp in ax.spines.values():
            sp.set_edgecolor("#2a305a")
        ax.tick_params(colors=LIGHT, labelsize=7)

    for pair in row_axs:
        for ax in pair: style(ax)
    for ax in (ax_L_ov, ax_R_ov): style(ax)

    # ── Helper: extract a steady-state window ────────────────────────────────
    def steady_window(res, side_key):
        t  = res["time_s"]
        f  = res[side_key]
        dt = float(np.diff(t[:10]).mean())
        b, a = butter(2, LPF_HZ * dt * 2, btype="low")
        # Use last SHOW_WIN_S of the simulation
        n_win = int(SHOW_WIN_S / dt)
        t_w   = t[-n_win:] - t[-n_win]     # shift to start at 0
        f_w   = filtfilt(b, a, f[-n_win:])
        return t_w, f_w

    y_max = max(
        gallery[wk]["res"]["force_RE"].max() for wk in KEY_WEEKS) * 1.10

    # ── Draw each week row ────────────────────────────────────────────────────
    for i, wk in enumerate(KEY_WEEKS):
        g     = gallery[wk]
        res   = g["res"]
        col   = week_cols[wk]
        ai    = g["amp_ai"]
        c_L   = g["c_L"]
        ax_L, ax_R = row_axs[i]

        t_L, fL = steady_window(res, "force_LE")
        t_R, fR = steady_window(res, "force_RE")

        # ── Left panel ──────────────────────────────────────────────────────
        ax_L.fill_between(t_L, fL, fR[:len(fL)],
                          alpha=0.18, color=col, zorder=1)
        ax_L.plot(t_L, fR[:len(fL)], color="#ff6666", lw=1.1,
                  alpha=0.55, ls="--", label="Right" if i == 0 else "")
        ax_L.plot(t_L, fL,           color=col,      lw=1.8,
                  label="Left" if i == 0 else "")
        ax_L.set_ylim(0, y_max)
        ax_L.set_xlim(0, SHOW_WIN_S)
        if i < N - 1:
            ax_L.set_xticklabels([])
        else:
            ax_L.set_xlabel("Time [s]", fontsize=8, color=LIGHT)
        ax_L.set_ylabel("Force [N]", fontsize=7, color=LIGHT)

        # Week label + metrics
        deficit = (1 - c_L / C0) * 100
        tag = (f"Wk {wk:2d}  |  c_L={c_L:.3f}"
               f"  deficit={deficit:.1f}%"
               f"  AmpAI={ai*100:.1f}%")
        ax_L.text(0.01, 0.96, tag, transform=ax_L.transAxes,
                  fontsize=7.5, color=col, va="top", fontweight="bold")

        # ── Right panel ─────────────────────────────────────────────────────
        ax_R.fill_between(t_R, fR, fL[:len(fR)],
                          alpha=0.18, color=col, zorder=1)
        ax_R.plot(t_R, fL[:len(fR)], color=col,      lw=1.1,
                  alpha=0.55, ls="--", label="Left" if i == 0 else "")
        ax_R.plot(t_R, fR,           color="#ff6666", lw=1.8,
                  label="Right" if i == 0 else "")
        ax_R.set_ylim(0, y_max)
        ax_R.set_xlim(0, SHOW_WIN_S)
        ax_R.yaxis.tick_right()
        ax_R.yaxis.set_label_position("right")
        if i < N - 1:
            ax_R.set_xticklabels([])
        else:
            ax_R.set_xlabel("Time [s]", fontsize=8, color=LIGHT)
        ax_R.set_ylabel("Force [N]", fontsize=7, color=LIGHT)

        # ── Overlay panels ───────────────────────────────────────────────────
        lw_ov = 1.4 if wk in (0, 12) else 0.9
        al_ov = 1.0 if wk in (0, 12) else 0.60
        ax_L_ov.plot(t_L, fL, color=col, lw=lw_ov, alpha=al_ov,
                     label=f"Wk {wk}" if wk in (0, 4, 8, 12) else "")
        ax_R_ov.plot(t_R, fR, color="#ff6666" if wk == 0 else col,
                     lw=lw_ov, alpha=al_ov,
                     label=f"Wk {wk}" if wk in (0, 4, 8, 12) else "")

    # ── Column headers ────────────────────────────────────────────────────────
    row_axs[0][0].set_title(
        "LEFT Extensor Force  (solid = Left,  dashed = Right reference)",
        fontsize=10, color=LIGHT, pad=8)
    row_axs[0][1].set_title(
        "RIGHT Extensor Force  (solid = Right,  dashed = Left reference)",
        fontsize=10, color=LIGHT, pad=8)

    # Legend on first row
    for ax in row_axs[0]:
        ax.legend(fontsize=7, facecolor="#12172a", edgecolor="#2a305a",
                  labelcolor=LIGHT, loc="upper right")

    # ── Overlay panels styling ─────────────────────────────────────────────────
    for ax, side in [(ax_L_ov, "LEFT  — all weeks overlaid"),
                     (ax_R_ov, "RIGHT — all weeks overlaid")]:
        ax.set_facecolor("#12172a")
        for sp in ax.spines.values(): sp.set_edgecolor("#2a305a")
        ax.tick_params(colors=LIGHT, labelsize=7)
        ax.set_ylim(0, y_max)
        ax.set_xlim(0, SHOW_WIN_S)
        ax.set_xlabel("Time [s]", fontsize=8, color=LIGHT)
        ax.set_ylabel("Force [N]", fontsize=8, color=LIGHT)
        ax.set_title(side, fontsize=9, color=LIGHT, pad=5)
        ax.legend(fontsize=7.5, facecolor="#12172a", edgecolor="#2a305a",
                  labelcolor=LIGHT, loc="upper right")
    ax_R_ov.yaxis.tick_right()
    ax_R_ov.yaxis.set_label_position("right")

    # Colour-bar annotation: red→teal ramp label
    cbar_ax = fig.add_axes([0.005, 0.08, 0.007, 0.82])
    sm = plt.cm.ScalarMappable(
        cmap=CMAP,
        norm=plt.Normalize(vmin=0, vmax=12))
    sm.set_array([])
    cb = plt.colorbar(sm, cax=cbar_ax)
    cb.set_label("Rehab week", fontsize=8, color=LIGHT, rotation=90)
    cb.ax.yaxis.set_tick_params(color=LIGHT, labelsize=7)
    plt.setp(cb.ax.yaxis.get_ticklabels(), color=LIGHT)
    cbar_ax.yaxis.label.set_color(LIGHT)

    fig.suptitle(
        "Walking Pattern Recovery — Gait Gallery\n"
        "L/R Extensor Force Waveforms Week by Week  "
        "(Full-Recovery Trajectory)",
        fontsize=13, fontweight="bold", color=LIGHT, y=0.975)

    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"[Plot] Saved → {out_path}")
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    np.random.seed(42)
    print("=" * 62)
    print("  TVB Phase 3 — Gait Pattern Gallery")
    print("=" * 62)

    print("\n[1/3] Loading Phase 1 TVB scales...")
    scales = load_phase1_scales()
    c_stroke_L  = C0 * scales["scale_L"]
    c_healthy_L = C0
    c_R         = C0 * scales["scale_R"]

    print(f"  Stroke c_L={c_stroke_L:.3f}  Healthy c_L={c_healthy_L:.3f}")

    print("\n[2/3] Simulating key weeks (Full-Recovery trajectory)...")
    gallery = run_gallery_sims(c_stroke_L, c_healthy_L, c_R)

    print("\n  Week  c_L    AmpAI   PeakF_L  PeakF_R")
    print("  " + "─"*42)
    for wk in KEY_WEEKS:
        g = gallery[wk]
        print(f"  {wk:4d}  {g['c_L']:.3f}  "
              f"{g['amp_ai']*100:5.1f}%  "
              f"{g['amp_L']:6.1f}N  {g['amp_R']:6.1f}N")

    print("\n[3/3] Generating gait gallery figure...")
    out_path = OUTPUT_DIR / "phase3_gait_gallery.png"
    make_gallery(gallery, c_stroke_L, c_R, out_path)

    print(f"\n{'='*62}")
    print(f"  Done.  Figure → {out_path}")
    print(f"{'='*62}\n")


if __name__ == "__main__":
    main()
