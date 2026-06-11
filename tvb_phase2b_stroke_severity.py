#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TVB Phase 2B — Graded acute-stroke severity with motoneuron recruitment
========================================================================

Demonstrates the bio-plausibility upgrade to the stroke force model:

  Acute hemiparesis is frequently a FLACCID limb — force ≈ 0 — not merely a
  modestly weakened one.  We add a motoneuron-pool recruitment threshold
  (Henneman size principle) so that below a critical descending drive the
  affected limb produces negligible rhythmic force.

This figure sweeps clinical severity (healthy → complete) and shows:
  • LEFT (affected) extensor force collapsing to ~0 as severity rises
  • RIGHT (unaffected) extensor force preserved
  • Peak-force and amplitude-asymmetry summary across severities
  • The recruitment curve R(c) that produces the threshold behaviour

Run with:  python3 tvb_phase2b_stroke_severity.py
"""

import numpy as np
from pathlib import Path
from scipy.signal import find_peaks, butter, filtfilt

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import importlib.util
spec = importlib.util.spec_from_file_location(
    "p2", Path(__file__).parent / "tvb_phase2_spinal_cpg.py")
p2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p2)

OUTPUT_DIR = Path("./tvb_output/phase2b")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

C0 = p2.C0

# Severity order for plotting (healthy first → most severe last)
SEVERITY_ORDER = ["healthy", "mild", "moderate", "severe", "complete"]
SEVERITY_COLORS = {
    "healthy":  "#3498db",
    "mild":     "#88cc88",
    "moderate": "#f9a825",
    "severe":   "#e67e22",
    "complete": "#e74c3c",
}


def peak_force(t, force):
    """Mean peak extensor force in the steady-state second half of the run."""
    dt = float(np.diff(t[:10]).mean())
    half = len(force) // 2
    b, a = butter(2, 4.0 * dt * 2, btype="low")
    fs = filtfilt(b, a, force[half:])
    pk, _ = find_peaks(fs, height=0.5, distance=max(1, int(0.3 / dt)))
    return float(fs[pk].mean()) if len(pk) else float(max(fs.max(), 0.0))


def run_severities():
    """Simulate each severity level.  Left side affected, right side healthy."""
    results = {}
    print("Simulating graded stroke severities (left affected, right healthy):")
    print(f"  {'severity':>9s}  {'c_L':>5s}  {'R(c_L)':>6s}  "
          f"{'F_L[N]':>7s}  {'F_R[N]':>7s}  {'AmpAI':>6s}")
    for sev in SEVERITY_ORDER:
        scale = p2.STROKE_SEVERITY[sev]
        c_L = C0 * scale
        c_R = C0                      # unaffected side at healthy drive
        res = p2.simulate(c_L, c_R, sim_s=20.0)
        t = res["time_s"]
        fL = res["force_LE"] + res["force_LF"]
        fR = res["force_RE"] + res["force_RF"]
        pL = peak_force(t, fL)
        pR = peak_force(t, fR)
        ampai = (pR - pL) / max(pR, pL) if max(pR, pL) > 0 else 0.0
        results[sev] = {
            "c_L": c_L, "c_R": c_R,
            "R": p2.recruitment_gain(c_L),
            "t": t, "fL": fL, "fR": fR,
            "peak_L": pL, "peak_R": pR, "ampai": ampai,
        }
        print(f"  {sev:>9s}  {c_L:>5.2f}  {results[sev]['R']:>6.3f}  "
              f"{pL:>7.1f}  {pR:>7.1f}  {ampai*100:>5.0f}%")
    return results


def make_figure(results, out_path):
    DARK = "#0e1117"; LIGHT = "#e0e0e0"; ACCENT = "#f9a825"

    fig = plt.figure(figsize=(20, 12))
    fig.patch.set_facecolor(DARK)
    gs = gridspec.GridSpec(3, 2, figure=fig,
                           left=0.07, right=0.97, top=0.91, bottom=0.07,
                           hspace=0.42, wspace=0.24, height_ratios=[1, 1, 1])

    def mk(r, c, rs=1, cs=1):
        ax = fig.add_subplot(gs[r:r+rs, c:c+cs])
        ax.set_facecolor("#1a1e2e")
        for sp in ax.spines.values():
            sp.set_edgecolor("#3a3f5c")
        ax.tick_params(colors=LIGHT, labelsize=8)
        ax.xaxis.label.set_color(LIGHT); ax.yaxis.label.set_color(LIGHT)
        return ax

    def idx(ax, letter):
        ax.text(-0.09, 1.07, letter, fontsize=15, fontweight="bold",
                color=ACCENT, transform=ax.transAxes, ha="left", va="top")

    fig.suptitle(
        "Phase 2B: Graded Acute-Stroke Severity with Motoneuron Recruitment\n"
        "Below a critical descending drive the affected limb is FLACCID "
        "(force ≈ 0) — clinically realistic acute hemiparesis",
        fontsize=13, fontweight="bold", color=ACCENT, y=0.975)

    # Window for waveform display
    def window(t, f):
        i0 = int(np.argmax(t >= 10.0))
        return t[i0:], f[i0:]

    # ── A: LEFT (affected) extensor force overlay ────────────────────────
    axA = mk(0, 0)
    for sev in SEVERITY_ORDER:
        r = results[sev]
        tt, ff = window(r["t"], r["fL"])
        axA.plot(tt, ff, color=SEVERITY_COLORS[sev], lw=2,
                 label=f"{sev} (peak {r['peak_L']:.1f} N)", alpha=0.9)
    axA.set_title("LEFT (affected) extensor force — collapses with severity",
                  fontsize=10, color=LIGHT, fontweight="bold", pad=6)
    idx(axA, "A")
    axA.set_xlabel("Time [s]", fontsize=9); axA.set_ylabel("Force [N]", fontsize=9)
    axA.legend(fontsize=7.5, facecolor="#1a1e2e", edgecolor="#3a3f5c",
               labelcolor=LIGHT, loc="upper right")
    axA.grid(True, alpha=0.15, color=LIGHT)
    axA.set_ylim(-1, p2.FORCE_MAX * 1.4)

    # ── B: RIGHT (unaffected) extensor force overlay ─────────────────────
    axB = mk(0, 1)
    for sev in SEVERITY_ORDER:
        r = results[sev]
        tt, ff = window(r["t"], r["fR"])
        axB.plot(tt, ff, color=SEVERITY_COLORS[sev], lw=2,
                 label=f"{sev} (peak {r['peak_R']:.1f} N)", alpha=0.9)
    axB.set_title("RIGHT (unaffected) extensor force — preserved",
                  fontsize=10, color=LIGHT, fontweight="bold", pad=6)
    idx(axB, "B")
    axB.set_xlabel("Time [s]", fontsize=9); axB.set_ylabel("Force [N]", fontsize=9)
    axB.legend(fontsize=7.5, facecolor="#1a1e2e", edgecolor="#3a3f5c",
               labelcolor=LIGHT, loc="upper right")
    axB.grid(True, alpha=0.15, color=LIGHT)
    axB.set_ylim(-1, p2.FORCE_MAX * 1.4)

    # ── C: peak force (L vs R) per severity ──────────────────────────────
    axC = mk(1, 0)
    x = np.arange(len(SEVERITY_ORDER)); w = 0.38
    pk_L = [results[s]["peak_L"] for s in SEVERITY_ORDER]
    pk_R = [results[s]["peak_R"] for s in SEVERITY_ORDER]
    axC.bar(x - w/2, pk_L, w, label="Left (affected)", color="#e74c3c",
            alpha=0.9, edgecolor=LIGHT, linewidth=0.5)
    axC.bar(x + w/2, pk_R, w, label="Right (unaffected)", color="#4ecdc4",
            alpha=0.9, edgecolor=LIGHT, linewidth=0.5)
    for xi, v in zip(x - w/2, pk_L):
        axC.text(xi, v + 0.3, f"{v:.1f}", ha="center", va="bottom",
                 fontsize=7.5, color="#ff9b8e")
    axC.set_title("Peak extensor force by severity", fontsize=10,
                  color=LIGHT, fontweight="bold", pad=6)
    idx(axC, "C")
    axC.set_xticks(x); axC.set_xticklabels(SEVERITY_ORDER, fontsize=8)
    axC.set_ylabel("Peak force [N]", fontsize=9)
    axC.legend(fontsize=8, facecolor="#1a1e2e", edgecolor="#3a3f5c",
               labelcolor=LIGHT, loc="upper right")
    axC.grid(True, alpha=0.15, color=LIGHT, axis="y")

    # ── D: amplitude asymmetry index per severity ────────────────────────
    axD = mk(1, 1)
    ampai = [results[s]["ampai"] * 100 for s in SEVERITY_ORDER]
    bars = axD.bar(x, ampai, color=[SEVERITY_COLORS[s] for s in SEVERITY_ORDER],
                   alpha=0.9, edgecolor=LIGHT, linewidth=0.5)
    for b, v in zip(bars, ampai):
        axD.text(b.get_x() + b.get_width()/2, v + 1, f"{v:.0f}%",
                 ha="center", va="bottom", fontsize=8, color=LIGHT)
    axD.axhline(10, color="#88ff88", lw=1.2, ls="--", alpha=0.6,
                label="Near-normal gait (<10%)")
    axD.axhline(100, color="#888", lw=1, ls=":", alpha=0.5)
    axD.set_title("Force amplitude asymmetry index (AmpAI)", fontsize=10,
                  color=LIGHT, fontweight="bold", pad=6)
    idx(axD, "D")
    axD.set_xticks(x); axD.set_xticklabels(SEVERITY_ORDER, fontsize=8)
    axD.set_ylabel("AmpAI [%]  (100% = flaccid)", fontsize=9)
    axD.legend(fontsize=8, facecolor="#1a1e2e", edgecolor="#3a3f5c",
               labelcolor=LIGHT, loc="upper left")
    axD.grid(True, alpha=0.15, color=LIGHT, axis="y")
    axD.set_ylim(0, 110)

    # ── E: recruitment curve R(c) with severity markers ──────────────────
    axE = mk(2, 0, cs=2)
    cc = np.linspace(0, C0 * 1.05, 400)
    RR = [p2.recruitment_gain(c) for c in cc]
    axE.plot(cc, np.array(RR) * 100, color=ACCENT, lw=2.5,
             label=f"R(c) = 1/(1+e^(−{p2.K_RECRUIT:.0f}(c−{p2.C_CRIT:.2f})))")
    axE.axvline(p2.C_CRIT, color="#888", lw=1, ls=":", alpha=0.6)
    axE.text(p2.C_CRIT + 0.02, 8, f"C_CRIT = {p2.C_CRIT:.2f}",
             color="#aaa", fontsize=8)
    for sev in SEVERITY_ORDER:
        r = results[sev]
        axE.plot(r["c_L"], r["R"] * 100, "o", ms=11,
                 color=SEVERITY_COLORS[sev], markeredgecolor="white",
                 markeredgewidth=0.8, zorder=5)
        axE.annotate(sev, (r["c_L"], r["R"] * 100),
                     textcoords="offset points", xytext=(0, 12),
                     ha="center", fontsize=8, color=SEVERITY_COLORS[sev],
                     fontweight="bold")
    axE.set_title("Motoneuron-pool recruitment curve — the flaccidity threshold",
                  fontsize=10, color=LIGHT, fontweight="bold", pad=6)
    idx(axE, "E")
    axE.set_xlabel("Descending drive c  (C0 = healthy = %.2f)" % C0, fontsize=9)
    axE.set_ylabel("Recruitable force [%]", fontsize=9)
    axE.legend(fontsize=9, facecolor="#1a1e2e", edgecolor="#3a3f5c",
               labelcolor=LIGHT, loc="lower right")
    axE.grid(True, alpha=0.15, color=LIGHT)
    axE.set_xlim(0, C0 * 1.05); axE.set_ylim(-3, 105)

    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"\n[Plot] Saved → {out_path}")


def main():
    np.random.seed(42)
    print("=" * 72)
    print("  TVB Phase 2B — Graded Acute-Stroke Severity + Motoneuron Recruitment")
    print("=" * 72)
    results = run_severities()
    fig_path = OUTPUT_DIR / "phase2b_stroke_severity.png"
    make_figure(results, fig_path)
    print(f"\n{'='*72}")
    print("  Phase 2B complete.")
    print(f"  Figure: {fig_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
