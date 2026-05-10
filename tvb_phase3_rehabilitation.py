#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TVB Phase 3 — Rehabilitation Simulation
========================================

Clinical scenario
-----------------
After a left-hemisphere stroke that silences M1_L, the patient presents
with hemiparetic gait: reduced CST drive to the left spinal half-centre
→ lower extensor force amplitude, shorter stance phase, and asymmetric
locomotion.

Rehabilitation (physiotherapy, constraint-induced movement therapy,
non-invasive brain stimulation, etc.) gradually restores cortical
excitability and with it the CST drive.  We model this as:

    c_L(r) = c_stroke_L  +  r × (c_healthy_L − c_stroke_L)
    c_R    = c_healthy_R  (unaffected throughout)

where r ∈ [0, 1] is the **recovery fraction** (0 = acute stroke,
1 = full neurological recovery).

Recovery trajectories
---------------------
Three trajectories are simulated — matching common clinical outcomes:

  Fast   : logistic function, 80% recovered by week 8
  Slow   : linear, 60% at 12 weeks, plateau at 70%
  Full   : logistic reaching 100% by week 12

Gait metrics tracked at each recovery step
-------------------------------------------
• Peak extensor force amplitude L/R  (primary impairment marker)
• Amplitude asymmetry index  (AmpAI) = (F_R − F_L) / max(F_R, F_L)
• Duty-cycle asymmetry = duty_R − duty_L  (stance-phase imbalance)
• Step frequency (both sides, to show dose-response recovery)

Clinical milestones (literature-based thresholds)
--------------------------------------------------
  AmpAI < 0.25  → community ambulation
  AmpAI < 0.10  → near-normal symmetric gait

Outputs (tvb_output/phase3/)
----------------------------
  phase3_recovery_<trajectory>_<ts>.h5   — one file per trajectory
  phase3_rehabilitation.png              — 6-panel recovery figure

Run with:  python3 tvb_phase3_rehabilitation.py
"""

import numpy as np
import h5py
import glob
from pathlib import Path
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from scipy.signal import find_peaks, butter, filtfilt
from scipy.ndimage import uniform_filter1d

# ── Import shared CPG machinery from Phase 2 ─────────────────────────────────
# We re-use the same Matsuoka + muscle-proxy simulator verbatim so results
# are directly comparable.  The only thing that changes between Phase 2
# and Phase 3 is c_L (the left tonic input that encodes cortical recovery).
import sys, importlib, types

# Load Phase 2 as a module without executing main()
_spec  = importlib.util.spec_from_file_location(
    "p2", Path(__file__).parent / "tvb_phase2_spinal_cpg.py")
_p2    = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_p2)

simulate          = _p2.simulate
load_phase1_scales = _p2.load_phase1_scales
C0                = _p2.C0

# ── Output directory ──────────────────────────────────────────────────────────
OUTPUT_DIR = Path("./tvb_output/phase3")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Recovery resolution & simulation params ───────────────────────────────────
N_STEPS   = 13          # 0..12 weeks
SIM_S     = 20.0        # CPG integration time per step [s]
DT_S      = 0.001


# ══════════════════════════════════════════════════════════════════════════════
#  RECOVERY TRAJECTORIES
# ══════════════════════════════════════════════════════════════════════════════
def logistic(t, k, t0):
    return 1.0 / (1.0 + np.exp(-k * (t - t0)))

def make_trajectories(n=N_STEPS):
    """Return dict {name: array of recovery fractions, shape (n,)}."""
    wk = np.linspace(0, 12, n)          # weeks 0..12

    # Full recovery — fast logistic (k=0.7, inflection at week 4)
    full  = logistic(wk, k=0.7, t0=4.0)
    full  = (full - full[0]) / (full[-1] - full[0])  # normalise 0→1

    # Fast partial — logistic plateau at 80%
    fast  = logistic(wk, k=0.8, t0=4.0)
    fast  = (fast - fast[0]) / (1.05*(fast[-1] - fast[0]))
    fast  = np.clip(fast, 0, 0.80)

    # Slow partial — linear plateau at 60%
    slow  = np.clip(np.linspace(0, 0.70, n), 0, 0.70)

    return {"Full recovery": full,
            "Fast partial (80%)": fast,
            "Slow partial (60%)": slow}, wk


# ══════════════════════════════════════════════════════════════════════════════
#  GAIT METRICS FROM ONE CPG RUN
# ══════════════════════════════════════════════════════════════════════════════
def _smooth(t, force):
    dt = float(np.diff(t[:10]).mean())
    half = len(force) // 2
    b, a  = butter(2, 4.0 * dt * 2, btype="low")
    return filtfilt(b, a, force[half:]), t[half:], dt

def peak_amplitude(t, force):
    fs, ts, dt = _smooth(t, force)
    pk, _ = find_peaks(fs, height=0.5, distance=max(1, int(0.3/dt)))
    return float(fs[pk].mean()) if len(pk) >= 1 else float(fs.max())

def step_frequency(t, force):
    fs, ts, dt = _smooth(t, force)
    pk, _ = find_peaks(fs, height=0.5, distance=max(1, int(0.3/dt)))
    if len(pk) < 2:
        return 0.0
    return float(1000.0 / (np.diff(ts[pk]).mean() * 1000.0))

def duty_cycle(t, force):
    fs, ts, dt = _smooth(t, force)
    return float(np.mean(fs > fs.max() * 0.5)) if fs.max() > 0.5 else 0.0

def gait_metrics(res):
    t = res["time_s"]
    aL = peak_amplitude(t, res["force_LE"])
    aR = peak_amplitude(t, res["force_RE"])
    fL = step_frequency(t, res["force_LE"])
    fR = step_frequency(t, res["force_RE"])
    dL = duty_cycle(t, res["force_LE"])
    dR = duty_cycle(t, res["force_RE"])
    amp_ai  = (aR - aL) / max(aR, aL) if max(aR, aL) > 0 else 0.0
    duty_ai = dR - dL
    return dict(amp_L=aL, amp_R=aR, freq_L=fL, freq_R=fR,
                duty_L=dL, duty_R=dR,
                amp_ai=amp_ai, duty_ai=duty_ai)


# ══════════════════════════════════════════════════════════════════════════════
#  RUN ONE REHABILITATION TRAJECTORY
# ══════════════════════════════════════════════════════════════════════════════
def run_trajectory(name: str, r_arr: np.ndarray,
                   c_stroke_L: float, c_healthy_L: float,
                   c_R: float, weeks: np.ndarray) -> dict:
    """
    Simulate CPG for every recovery fraction in r_arr.
    Returns dict of metric arrays (one value per recovery step).
    """
    metrics_list = []
    c_L_arr = []

    for i, r in enumerate(r_arr):
        c_L = c_stroke_L + r * (c_healthy_L - c_stroke_L)
        c_L_arr.append(c_L)
        pct = (1 - (c_stroke_L + r*(c_healthy_L - c_stroke_L)) /
               c_healthy_L) * 100
        print(f"  [{i+1:2d}/{len(r_arr)}]  week={weeks[i]:.0f}  "
              f"r={r:.2f}  c_L={c_L:.3f}  deficit={pct:.1f}%")
        res = simulate(c_L, c_R, sim_s=SIM_S, dt_s=DT_S)
        metrics_list.append(gait_metrics(res))

    # Transpose list-of-dicts → dict-of-arrays
    keys = metrics_list[0].keys()
    out  = {k: np.array([m[k] for m in metrics_list]) for k in keys}
    out["c_L"]   = np.array(c_L_arr)
    out["r"]     = r_arr
    out["weeks"] = weeks
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  SAVE HDF5
# ══════════════════════════════════════════════════════════════════════════════
def save_trajectory_h5(name: str, data: dict) -> Path:
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("%","pct")
    path = OUTPUT_DIR / f"phase3_recovery_{slug}_{ts}.h5"
    with h5py.File(path, "w") as f:
        f.attrs["trajectory"] = name
        f.attrs["model"]      = "Matsuoka+Rehab"
        for k, v in data.items():
            f.create_dataset(k, data=np.asarray(v, dtype=np.float32))
    print(f"  Saved → {path}")
    return path


# ══════════════════════════════════════════════════════════════════════════════
#  6-PANEL REHABILITATION FIGURE
# ══════════════════════════════════════════════════════════════════════════════
def make_rehab_plot(all_traj: dict, scales: dict, weeks: np.ndarray,
                    out_path: Path):
    DARK  = "#0e1117"; LIGHT = "#e0e0e0"; ACCENT = "#f9a825"
    # Trajectory colours
    TCOLS = {
        "Full recovery":      "#4bc8ff",   # blue
        "Fast partial (80%)": "#ffa040",   # orange
        "Slow partial (60%)": "#cc44cc",   # purple
    }
    # Clinical threshold colours
    TH_COM  = "#44cc88"   # community ambulation AmpAI < 0.25
    TH_NORM = "#88ff88"   # near-normal           AmpAI < 0.10

    fig = plt.figure(figsize=(19, 17))
    fig.patch.set_facecolor(DARK)
    gs  = gridspec.GridSpec(4, 3, figure=fig,
                            left=0.07, right=0.97, top=0.93, bottom=0.06,
                            hspace=0.55, wspace=0.38)

    axs = {
        "diag":  fig.add_subplot(gs[0, :]),
        "ampai": fig.add_subplot(gs[1, :2]),   # amplitude asymmetry recovery
        "dutyai":fig.add_subplot(gs[2, :2]),   # duty-cycle asymmetry
        "ampabs":fig.add_subplot(gs[1, 2]),    # absolute L/R peak force
        "freq":  fig.add_subplot(gs[2, 2]),    # step frequency recovery
        "cL":    fig.add_subplot(gs[3, :2]),   # CST drive / c_L recovery
        "tbl":   fig.add_subplot(gs[3, 2]),    # metrics table at key weeks
    }

    for ax in axs.values():
        ax.set_facecolor("#1a1e2e")
        for s in ax.spines.values(): s.set_edgecolor("#3a3f5c")
        ax.tick_params(colors=LIGHT, labelsize=8)
        ax.xaxis.label.set_color(LIGHT); ax.yaxis.label.set_color(LIGHT)

    # ── Panel 0: Rehab pipeline diagram ──────────────────────────────────────
    ax = axs["diag"]
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    boxes = [
        (0.07,  "Stroke\n(acute)\nL-M1 silent",       "#6e2e2e"),
        (0.22,  "Rehabilitation\n(physio / CIMT /\nrTMS)",  "#2e3b6e"),
        (0.37,  "CST drive\nrecovery\nc_L(r)",         "#1a4a2e"),
        (0.52,  "Matsuoka\nCPG\n(4 neurons)",          "#2e3b6e"),
        (0.67,  "Muscle\nForce\nproxy",                "#1a4a2e"),
        (0.82,  "Gait\nMetrics\n(AmpAI, duty)",        "#2a4a1e"),
    ]
    for x0, label, col in boxes:
        rect = FancyBboxPatch((x0-0.07, 0.12), 0.13, 0.76,
                              boxstyle="round,pad=0.02", lw=1.0,
                              edgecolor="#5a6090", facecolor=col,
                              transform=ax.transAxes, clip_on=False)
        ax.add_patch(rect)
        ax.text(x0, 0.50, label, ha="center", va="center", fontsize=7.5,
                color=LIGHT, transform=ax.transAxes, fontweight="bold",
                linespacing=1.35)
    for x0, x1 in [(0.14,0.22),(0.29,0.37),(0.44,0.52),(0.59,0.67),(0.74,0.82)]:
        ax.annotate("", xy=(x1-0.06, 0.50), xytext=(x0+0.01, 0.50),
                    xycoords="axes fraction", textcoords="axes fraction",
                    arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1.5))
    note = (f"TVB stroke: CST_L {scales['healthy_L']:.3f}→{scales['stroke_L']:.3f} Hz "
            f"(−{(1-scales['scale_L'])*100:.1f}% deficit) → c_L: "
            f"{C0:.3f}→{C0*scales['scale_L']:.3f}.  "
            f"Recovery r∈[0,1] linearly restores c_L → monitors gait symmetry "
            f"across 0–12 rehabilitation weeks.")
    ax.text(0.50, 0.01, note, ha="center", va="bottom", fontsize=7.5,
            color=ACCENT, transform=ax.transAxes,
            bbox=dict(fc="#1a1e2e", ec=ACCENT, pad=3, lw=0.8))

    # ── Helper: clinical milestone bands ─────────────────────────────────────
    def _threshold_bands(ax, xmin, xmax):
        ax.axhspan(0.10, 0.25, alpha=0.10, color=TH_COM,  zorder=0)
        ax.axhspan(0.00, 0.10, alpha=0.15, color=TH_NORM, zorder=0)
        ax.axhline(0.25, color=TH_COM,  lw=0.9, ls="--", alpha=0.7,
                   label="Community ambulation  (AmpAI<0.25)")
        ax.axhline(0.10, color=TH_NORM, lw=0.9, ls="--", alpha=0.7,
                   label="Near-normal gait  (AmpAI<0.10)")

    # ── Panel 1: Amplitude asymmetry recovery ─────────────────────────────────
    ax = axs["ampai"]
    _threshold_bands(ax, weeks[0], weeks[-1])
    for name, data in all_traj.items():
        ax.plot(data["weeks"], data["amp_ai"],
                color=TCOLS[name], lw=2.0, marker="o", ms=4,
                label=name)
    # Mark acute stroke and healthy reference
    ax.axhline(0.0, color=LIGHT, lw=0.6, ls=":", alpha=0.4)
    ax.set_xlabel("Rehabilitation week", fontsize=9)
    ax.set_ylabel("Amplitude Asymmetry Index\n(R−L)/max(R,L)", fontsize=9)
    ax.set_title(
        "Force Amplitude Asymmetry Recovery\n"
        "(primary hemiparetic gait metric)",
        fontsize=10, color=LIGHT, pad=5)
    ax.legend(fontsize=7.5, facecolor="#1a1e2e", edgecolor="#3a3f5c",
              labelcolor=LIGHT, loc="upper right", ncol=1)
    ax.set_xlim(weeks[0], weeks[-1])
    ax.set_ylim(-0.05, max(d["amp_ai"].max() for d in all_traj.values()) + 0.05)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    # Annotate final week values
    for name, data in all_traj.items():
        ax.annotate(f"  {data['amp_ai'][-1]*100:.0f}%",
                    xy=(weeks[-1], data["amp_ai"][-1]),
                    fontsize=7, color=TCOLS[name])

    # ── Panel 2: Duty-cycle asymmetry ─────────────────────────────────────────
    ax = axs["dutyai"]
    for name, data in all_traj.items():
        ax.plot(data["weeks"], data["duty_ai"] * 100,
                color=TCOLS[name], lw=2.0, marker="s", ms=4,
                label=name)
    ax.axhline(0, color=LIGHT, lw=0.6, ls=":", alpha=0.4)
    ax.axhline(5, color=TH_COM,  lw=0.9, ls="--", alpha=0.7,
               label="Community ambulation  (ΔDuty<5%)")
    ax.axhline(2, color=TH_NORM, lw=0.9, ls="--", alpha=0.7,
               label="Near-normal  (ΔDuty<2%)")
    ax.set_xlabel("Rehabilitation week", fontsize=9)
    ax.set_ylabel("Duty-Cycle Asymmetry  Δ(R−L)  [%]", fontsize=9)
    ax.set_title("Stance-Phase Asymmetry Recovery", fontsize=10,
                 color=LIGHT, pad=5)
    ax.legend(fontsize=7.5, facecolor="#1a1e2e", edgecolor="#3a3f5c",
              labelcolor=LIGHT, loc="upper right")
    ax.set_xlim(weeks[0], weeks[-1])
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    # ── Panel 3: Absolute peak force L and R ──────────────────────────────────
    ax = axs["ampabs"]
    for name, data in all_traj.items():
        ax.plot(data["weeks"], data["amp_L"],
                color=TCOLS[name], lw=1.8, ls="-", marker="o", ms=3.5,
                label=f"{name} L")
        ax.plot(data["weeks"], data["amp_R"],
                color=TCOLS[name], lw=1.2, ls="--", marker="x", ms=4,
                alpha=0.6, label=f"{name} R")
    ax.set_xlabel("Rehabilitation week", fontsize=9)
    ax.set_ylabel("Peak Extensor Force [N]", fontsize=9)
    ax.set_title("Absolute Force Recovery\n(solid=Left, dashed=Right)",
                 fontsize=9, color=LIGHT, pad=5)
    ax.legend(fontsize=6.0, facecolor="#1a1e2e", edgecolor="#3a3f5c",
              labelcolor=LIGHT, ncol=1)
    ax.set_xlim(weeks[0], weeks[-1])
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    # ── Panel 4: Step frequency ────────────────────────────────────────────────
    ax = axs["freq"]
    for name, data in all_traj.items():
        ax.plot(data["weeks"], data["freq_L"],
                color=TCOLS[name], lw=1.8, ls="-", marker="o", ms=3.5,
                label=f"{name} L")
        ax.plot(data["weeks"], data["freq_R"],
                color=TCOLS[name], lw=1.2, ls="--", marker="x", ms=4,
                alpha=0.6)
    ax.set_xlabel("Rehabilitation week", fontsize=9)
    ax.set_ylabel("Step Frequency [Hz]", fontsize=9)
    ax.set_title("Cadence Recovery\n(solid=Left, dashed=Right)",
                 fontsize=9, color=LIGHT, pad=5)
    ax.legend(fontsize=6.0, facecolor="#1a1e2e", edgecolor="#3a3f5c",
              labelcolor=LIGHT, ncol=1)
    ax.set_xlim(weeks[0], weeks[-1])
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    # ── Panel 5: CST drive / c_L recovery curve ───────────────────────────────
    ax = axs["cL"]
    for name, data in all_traj.items():
        ax.plot(data["weeks"], data["c_L"],
                color=TCOLS[name], lw=2.0, label=name)
    ax.axhline(C0, color=LIGHT, lw=0.8, ls="--", alpha=0.5,
               label=f"Healthy c_L = {C0:.2f}")
    ax.axhline(C0 * scales["scale_L"], color="#ff4b4b", lw=0.8, ls=":",
               alpha=0.7, label=f"Acute stroke c_L = {C0*scales['scale_L']:.3f}")
    ax.fill_between(weeks,
                    C0 * scales["scale_L"], C0,
                    alpha=0.07, color=LIGHT)
    ax.set_xlabel("Rehabilitation week", fontsize=9)
    ax.set_ylabel("Left CST input  c_L", fontsize=9)
    ax.set_title("Cortical Drive Recovery Trajectories\n"
                 "(TVB Phase 1 → Phase 2 → Phase 3 link)",
                 fontsize=10, color=LIGHT, pad=5)
    ax.legend(fontsize=7.5, facecolor="#1a1e2e", edgecolor="#3a3f5c",
              labelcolor=LIGHT)
    ax.set_xlim(weeks[0], weeks[-1])
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    # ── Panel 6: Key-week summary table ───────────────────────────────────────
    ax = axs["tbl"]
    ax.axis("off")
    ax.set_title("Key-Week Metrics", fontsize=10, color=LIGHT, pad=6)

    key_wks = [0, 4, 8, 12]   # weeks to tabulate
    col_c   = [LIGHT] + [TCOLS[n] for n in all_traj]
    col_x   = np.linspace(0.01, 0.98, 1 + len(all_traj))

    header = ["Wk / Metric"] + list(all_traj.keys())
    y0, dy = 0.99, 0.055

    def tbl_text(ax, row_i, texts, colors, bold=False):
        for j, (txt, cx, cc) in enumerate(zip(texts, col_x, colors)):
            ax.text(cx, y0 - row_i*dy, txt, transform=ax.transAxes,
                    fontsize=6.2, color=cc,
                    fontweight="bold" if bold else "normal", va="top",
                    ha="left")

    tbl_text(ax, 0, header, col_c, bold=True)
    row = 1
    tbl_text(ax, row, ["─"*12]*len(header), col_c); row += 1

    for wk in key_wks:
        # Find nearest index
        idx = int(np.argmin(np.abs(weeks - wk)))
        tbl_text(ax, row,
                 [f"Week {wk:2d}"] + [""]*len(all_traj),
                 col_c, bold=True); row += 1
        for metric, label in [
                ("amp_L",  "  PeakF L [N]"),
                ("amp_ai", "  AmpAI [%]"),
                ("duty_ai","  ΔDuty [%]"),
                ("freq_L", "  Freq L [Hz]"),
        ]:
            vals = [label]
            for name, data in all_traj.items():
                v = data[metric][idx]
                if metric == "amp_ai":
                    vals.append(f"{v*100:.1f}%")
                elif metric == "duty_ai":
                    vals.append(f"{v*100:.1f}%")
                elif metric == "freq_L":
                    vals.append(f"{v:.3f}")
                else:
                    vals.append(f"{v:.1f}")
            tbl_text(ax, row, vals, col_c); row += 1
        tbl_text(ax, row, [""]*len(header), col_c); row += 1

    fig.suptitle(
        "Phase 3: Rehabilitation Simulation\n"
        "Cortical Recovery → Restored CST Drive → Symmetric Gait",
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
    print("  TVB Phase 3 — Rehabilitation Simulation")
    print("=" * 62)

    print("\n[1/4] Loading Phase 1 TVB drive scales...")
    scales = load_phase1_scales()

    c_stroke_L  = C0 * scales["scale_L"]    # ~0.967
    c_healthy_L = C0                         # 1.500
    c_R         = C0 * scales["scale_R"]    # ~1.500

    print(f"  Acute stroke:  c_L = {c_stroke_L:.3f}  c_R = {c_R:.3f}")
    print(f"  Full recovery: c_L = {c_healthy_L:.3f}  c_R = {c_R:.3f}")

    trajectories, weeks = make_trajectories(N_STEPS)

    print(f"\n[2/4] Running {len(trajectories)} recovery trajectories "
          f"× {N_STEPS} weeks each ({N_STEPS*len(trajectories)} CPG sims)...")

    all_traj  = {}
    traj_paths = {}
    for name, r_arr in trajectories.items():
        print(f"\n  ── {name} ──")
        data = run_trajectory(name, r_arr, c_stroke_L, c_healthy_L,
                              c_R, weeks)
        all_traj[name] = data
        traj_paths[name] = save_trajectory_h5(name, data)

    print("\n[3/4] Printing recovery summary...")
    print(f"\n  {'Week':>4s}  {'Trajectory':25s}  "
          f"{'c_L':6s}  {'AmpAI':7s}  {'ΔDuty':7s}  "
          f"{'Freq L':7s}  {'PeakFL':7s}")
    print(f"  {'─'*75}")
    for wk in [0, 2, 4, 6, 8, 10, 12]:
        idx = int(np.argmin(np.abs(weeks - wk)))
        for name, data in all_traj.items():
            print(f"  {wk:4.0f}  {name:25s}  "
                  f"{data['c_L'][idx]:.3f}  "
                  f"{data['amp_ai'][idx]*100:6.1f}%  "
                  f"{data['duty_ai'][idx]*100:6.1f}%  "
                  f"{data['freq_L'][idx]:.3f}Hz  "
                  f"{data['amp_L'][idx]:.1f}N")

    # Clinical milestone weeks
    print("\n  Clinical milestones (AmpAI thresholds):")
    for name, data in all_traj.items():
        ai = data["amp_ai"]
        com  = next((weeks[i] for i, v in enumerate(ai) if v < 0.25), None)
        norm = next((weeks[i] for i, v in enumerate(ai) if v < 0.10), None)
        print(f"    {name:25s}  community wk: "
              f"{f'{com:.0f}' if com is not None else 'never':>5s}  "
              f"near-normal wk: "
              f"{f'{norm:.0f}' if norm is not None else 'never':>5s}")

    print("\n[4/4] Generating rehabilitation figure...")
    fig_path = OUTPUT_DIR / "phase3_rehabilitation.png"
    make_rehab_plot(all_traj, scales, weeks, fig_path)

    print(f"\n{'='*62}")
    print("  Phase 3 complete.")
    for name, p in traj_paths.items():
        print(f"  {name:25s}: {p}")
    print(f"  Figure: {fig_path}")
    print(f"{'='*62}\n")


if __name__ == "__main__":
    main()
