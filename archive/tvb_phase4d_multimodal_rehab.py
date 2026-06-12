#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TVB Phase 4D — Multi-level Rehabilitation Comparison
======================================================

Compare four evidence-based rehabilitation protocols over 0–12 weeks:

  A. TMS only (Transcranial Magnetic Stimulation on motor cortex)
  B. Epidural SCS only (Spinal Cord Stimulation at L4/L5)
  C. Combined TMS + Epidural SCS (simultaneous)
  D. Sequential: TMS weeks 0–2, then add epidural SCS weeks 3–12

Each protocol modifies the Matsuoka CPG via different pathways:
  - TMS: increases cortical CST drive c_L, c_R (top-down)
  - Epidural SCS: periodic oscillatory input to CPG interneurons (bottom-up)
  - Combined: both pathways active
  - Sequential: leverages timing of cortical plasticity + spinal learning

Clinical grounding:
  Rejc et al. 2017 (Nat Med) — epidural SCS + voluntary effort → stepping in SCI
  Khedr et al. 2005 — TMS alone produces 2-week acceleration
  (Combined effect: synergistic, better than either alone)

Outputs (tvb_output/phase4d/):
  phase4d_multimodal_rehab.png   — 8-panel figure comparing all protocols
  phase4d_recovery_metrics.h5    — detailed metrics over 12 weeks

Run with:  python3 tvb_phase4d_multimodal_rehab.py
"""

import numpy as np
import h5py
from pathlib import Path
from datetime import datetime
from scipy.signal import find_peaks, butter, filtfilt

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

OUTPUT_DIR = Path("./tvb_output/phase4d")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
#  IMPORT PHASE 2 CPG SIMULATOR
# ─────────────────────────────────────────────────────────────────────────────
import importlib.util
spec = importlib.util.spec_from_file_location(
    "p2", Path(__file__).parent / "tvb_phase2_spinal_cpg.py")
p2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p2)

# Load baseline parameters from Phase 1 + Phase 2
scales = p2.load_phase1_scales()
C0 = p2.C0
c_stroke = C0 * scales["scale_L"]
c_healthy = C0
c_R = C0 * scales["scale_R"]

# ══════════════════════════════════════════════════════════════════════════════
#  EXTENDED CPG SIMULATOR: TWO-LEVEL (CORTICAL + SPINAL)
# ══════════════════════════════════════════════════════════════════════════════

def simulate_cpg_with_scs(c_L: float, c_R: float,
                           scs_amp_L: float = 0.0,
                           scs_freq_L: float = 20.0,
                           scs_amp_R: float = 0.0,
                           scs_freq_R: float = 20.0,
                           sim_s: float = 20.0) -> dict:
    """
    Matsuoka bilateral CPG with epidural spinal cord stimulation (SCS).

    Two-level drive model:
      c_total_L(t) = c_cortical_L + scs_amp_L · sin(2π·scs_freq_L·t)
      c_total_R(t) = c_cortical_R + scs_amp_R · sin(2π·scs_freq_R·t)

    Parameters
    ----------
    c_L, c_R        : cortical CST drives [Hz] (from TVB)
    scs_amp_L/R     : epidural SCS amplitude [Hz] (0 = no SCS)
    scs_freq_L/R    : epidural SCS frequency [Hz] (typical 10–30 Hz)
    sim_s           : simulation duration [s]

    Returns
    -------
    dict with keys: time_s, y (neuron states), force_LE, force_RE, etc.

    Physiological basis:
      Epidural SCS excites propriospinal interneurons and CPG neurons,
      providing a rhythmic drive that entrains the CPG to the stimulation
      frequency. Combined with cortical CST drive, produces synergistic
      stepping response (Rejc et al. 2017, Gerasimenko et al. 2015).
    """
    # Matsuoka parameters (from Phase 2)
    TAU1 = 0.08
    TAU2 = 0.48
    BETA = 2.5
    W_MAT = np.array([[0, -2.5, 0, 0],      # LE → LE (self), LE → LF (inhibit)
                      [-2.5, 0, 0, 0],      # LF → LF (self), LF ← LE (inhibit)
                      [0, 0, 0, -2.5],      # RE ← RE (self), RE ← RF (inhibit)
                      [0, 0, -2.5, 0]])     # RF → RF (self), RF ← RE (inhibit)
    W_COM = 0.3  # commissural coupling (reduced from 0.8 to eliminate overcomp)

    # Commissural coupling matrix (L ↔ R)
    W_COMM_MAT = np.array([[0, 0, -W_COM, 0],      # LE ← RE
                           [0, 0, 0, -W_COM],        # LF ← RF
                           [-W_COM, 0, 0, 0],        # RE ← LE
                           [0, -W_COM, 0, 0]])       # RF ← LF

    # Drive-dependent time constants (per-neuron τ scaling)
    tau1 = np.array([TAU1*(C0/c_L), TAU1*(C0/c_L),
                     TAU1*(C0/c_R), TAU1*(C0/c_R)])
    tau2 = np.array([TAU2*(C0/c_L), TAU2*(C0/c_L),
                     TAU2*(C0/c_R), TAU2*(C0/c_R)])

    # Initial conditions (balanced)
    u = np.array([0.80, -0.50, -0.50, 0.80])
    v = np.array([0.10, 0.0, 0.0, 0.10])
    y = np.maximum(u, 0.0)

    # Time stepping
    dt = 0.001
    N = int(sim_s / dt)
    t = np.arange(N) * dt

    # Storage
    y_arr = np.zeros((4, N))
    u_arr = np.zeros((4, N))

    for k in range(N):
        # ── Epidural SCS inputs (periodic forcing) ───────────────────────────
        scs_L = scs_amp_L * np.sin(2*np.pi * scs_freq_L * t[k])
        scs_R = scs_amp_R * np.sin(2*np.pi * scs_freq_R * t[k])
        scs_drives = np.array([scs_L, scs_L, scs_R, scs_R])  # applied symmetrically

        # ── Matsuoka dynamics ──────────────────────────────────────────────────
        c = np.array([c_L, c_L, c_R, c_R])  # cortical CST drives
        c_total = c + scs_drives               # combined drive: cortical + spinal

        du = (-u - W_MAT @ y - W_COMM_MAT @ y - BETA * v + c_total) / tau1
        dv = (-v + y) / tau2

        u = u + dt * du
        v = v + dt * dv
        y = np.maximum(u, 0.0)  # rectification (rate > 0)

        y_arr[:, k] = y
        u_arr[:, k] = u

    # ── Muscle forces (quadratic law: F ∝ y²) ───────────────────────────────
    force_scale = 20.0  # empirical scaling
    force_LE = force_scale * y_arr[0, :]**2
    force_LF = force_scale * y_arr[1, :]**2
    force_RE = force_scale * y_arr[2, :]**2
    force_RF = force_scale * y_arr[3, :]**2

    return {
        "time_s": t,
        "y": y_arr,
        "u": u_arr,
        "force_LE": force_LE,
        "force_LF": force_LF,
        "force_RE": force_RE,
        "force_RF": force_RF,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  REHABILITATION PROTOCOLS (A, B, C, D)
# ══════════════════════════════════════════════════════════════════════════════

PROTOCOLS = {
    "A: TMS only": {
        "label": "TMS (cortical, 10 Hz, 5×/wk)",
        "tms_active": True,
        "scs_active": False,
        "tms_delay_weeks": 0,
        "scs_delay_weeks": 999,
        "color": "#ff6b6b",
        "description": "Cortical TMS alone; increases CST drive via magnetic stimulation",
    },
    "B: SCS only": {
        "label": "Epidural SCS (spinal, 20 Hz, continuous)",
        "tms_active": False,
        "scs_active": True,
        "tms_delay_weeks": 999,
        "scs_delay_weeks": 0,
        "color": "#4ecdc4",
        "description": "Epidural SCS alone; direct CPG entrainment",
    },
    "C: TMS + SCS combined": {
        "label": "TMS + Epidural SCS (simultaneous)",
        "tms_active": True,
        "scs_active": True,
        "tms_delay_weeks": 0,
        "scs_delay_weeks": 0,
        "color": "#45b7d1",
        "description": "Both pathways active from week 0; dual-level drive",
    },
    "D: Sequential (TMS→SCS)": {
        "label": "TMS weeks 0–2, then add SCS weeks 3–12",
        "tms_active": True,
        "scs_active": True,
        "tms_delay_weeks": 0,
        "scs_delay_weeks": 3,
        "color": "#96ceb4",
        "description": "TMS early (cortical plasticity phase), add SCS later (spinal learning)",
    },
}


def recovery_c_L(weeks: np.ndarray, c_stroke: float, c_healthy: float,
                  k: float = 0.70, t0: float = 4.0) -> np.ndarray:
    """Logistic recovery of left CST drive over weeks (baseline, no NIBS offset)."""
    r = 1.0 / (1.0 + np.exp(-k * (weeks - t0)))
    # Normalize to [0, 1] over full trajectory (week 0 to week 12)
    r_all = 1.0 / (1.0 + np.exp(-k * (np.arange(13) - t0)))
    r_norm = (r_all - r_all[0]) / np.clip(r_all[-1] - r_all[0], 1e-6, None)
    # Interpolate
    r = 1.0 / (1.0 + np.exp(-k * (weeks - t0)))
    r = (r - r_all[0]) / np.clip(r_all[-1] - r_all[0], 1e-6, None)
    return np.clip(c_stroke + r * (c_healthy - c_stroke), c_stroke, c_healthy)


def run_multimodal_trajectories(weeks: np.ndarray) -> dict:
    """Run all four protocols over 0–12 weeks."""
    results = {}
    global c_stroke, c_healthy, c_R

    for proto_name, proto_spec in PROTOCOLS.items():
        print(f"\n  ── {proto_name} ──")
        tms_active = proto_spec["tms_active"]
        scs_active = proto_spec["scs_active"]
        tms_delay = proto_spec["tms_delay_weeks"]
        scs_delay = proto_spec["scs_delay_weeks"]

        amp_ai_arr, amp_L_arr, amp_R_arr, freq_arr = [], [], [], []

        for wk in weeks:
            # Baseline CST recovery (same for all protocols)
            c_L = recovery_c_L(np.array([wk]), c_stroke, c_healthy)[0]
            c_R_val = c_R  # right side mostly unaffected by stroke

            # TMS boost (if active and after delay)
            tms_boost_L = 0.0
            if tms_active and wk >= tms_delay:
                # TMS effects: increases I_o by ~7.5%, translates to CST boost
                # Decay over time as cortical plasticity accumulates
                weeks_since_start = wk - tms_delay
                tms_boost_L = 0.04 * max(0, 1.0 - weeks_since_start / 8.0)

            # Epidural SCS (if active and after delay)
            scs_amp = 0.0
            scs_freq = 20.0  # Hz (propriospinal frequency range)
            if scs_active and wk >= scs_delay:
                # SCS amplitude grows over weeks as patient learns to use it
                weeks_since_scs = wk - scs_delay
                scs_amp = 0.1 * (1.0 + weeks_since_scs / 6.0)  # slower growth
                scs_amp = np.clip(scs_amp, 0, 0.2)  # lower cap

            c_L_stim = np.clip(c_L + tms_boost_L, c_stroke, c_healthy)  # cap at healthy
            c_R_stim = np.clip(c_R_val, c_stroke, c_healthy)

            # Simulate CPG with two-level drive
            res = simulate_cpg_with_scs(
                c_L_stim, c_R_stim,
                scs_amp_L=scs_amp*0.1, scs_freq_L=scs_freq,  # reduce SCS amplitude
                scs_amp_R=scs_amp*0.1, scs_freq_R=scs_freq,
                sim_s=20.0
            )

            # Extract metrics
            dt = float(np.diff(res["time_s"][:10]).mean())
            half = len(res["time_s"]) // 2
            b, a = butter(2, 4.0*dt*2, btype="low")

            # Left leg force
            force_L = res["force_LE"] + res["force_LF"]
            fs_L = filtfilt(b, a, force_L[half:])
            pk_L, _ = find_peaks(fs_L, height=0.5, distance=max(1, int(0.3/dt)))
            amp_L = float(fs_L[pk_L].mean()) if len(pk_L) else float(fs_L.max())

            # Right leg force
            force_R = res["force_RE"] + res["force_RF"]
            fs_R = filtfilt(b, a, force_R[half:])
            pk_R, _ = find_peaks(fs_R, height=0.5, distance=max(1, int(0.3/dt)))
            amp_R = float(fs_R[pk_R].mean()) if len(pk_R) else float(fs_R.max())

            # Asymmetry index
            ai = (amp_R - amp_L) / max(amp_R, amp_L) if max(amp_R, amp_L) > 0 else 0.0

            # Frequency (from peak detection)
            if len(pk_L) > 0:
                freq = float(len(pk_L) / (res["time_s"][-1] - res["time_s"][half]))
            else:
                freq = 0.0

            amp_ai_arr.append(ai)
            amp_L_arr.append(amp_L)
            amp_R_arr.append(amp_R)
            freq_arr.append(freq)

            print(f"    wk={wk:.0f}  c_L={c_L_stim:.3f}  TMS_boost={tms_boost_L:.3f}  "
                  f"SCS_amp={scs_amp:.2f}  AmpAI={ai*100:.1f}%  "
                  f"F_L={amp_L:.1f} N  F_R={amp_R:.1f} N  freq={freq:.2f} Hz")

        results[proto_name] = {
            "amp_ai": np.array(amp_ai_arr),
            "amp_L": np.array(amp_L_arr),
            "amp_R": np.array(amp_R_arr),
            "freq": np.array(freq_arr),
            "color": proto_spec["color"],
            "label": proto_spec["label"],
        }

    return results


# ══════════════════════════════════════════════════════════════════════════════
#  SAVE RESULTS
# ══════════════════════════════════════════════════════════════════════════════

def save_results_h5(results: dict, weeks: np.ndarray, out_path: Path):
    """Save recovery metrics to HDF5."""
    with h5py.File(out_path, "w") as f:
        f.attrs["title"] = "Phase 4D: Multi-level Rehabilitation Comparison"
        f.attrs["protocols"] = list(results.keys())
        f.create_dataset("weeks", data=weeks)
        for proto_name, data in results.items():
            grp = f.create_group(proto_name.replace(":", "_").replace(" ", "_"))
            grp.create_dataset("amp_ai", data=data["amp_ai"])
            grp.create_dataset("amp_L", data=data["amp_L"])
            grp.create_dataset("amp_R", data=data["amp_R"])
            grp.create_dataset("freq", data=data["freq"])
    print(f"  Saved → {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
#  8-PANEL COMPARISON FIGURE
# ══════════════════════════════════════════════════════════════════════════════

def make_comparison_figure(results: dict, weeks: np.ndarray, out_path: Path):
    """Generate 8-panel figure comparing all four protocols."""
    DARK = "#0e1117"
    LIGHT = "#e0e0e0"
    ACCENT = "#f9a825"
    TH_NORM = "#88ff88"
    TH_COM = "#44cc88"

    fig = plt.figure(figsize=(20, 14))
    fig.patch.set_facecolor(DARK)

    gs = gridspec.GridSpec(
        4, 4, figure=fig,
        left=0.06, right=0.97, top=0.94, bottom=0.06,
        hspace=0.45, wspace=0.35
    )

    def mk(r, c, rs=1, cs=1):
        ax = fig.add_subplot(gs[r:r+rs, c:c+cs])
        ax.set_facecolor("#1a1e2e")
        for sp in ax.spines.values():
            sp.set_edgecolor("#3a3f5c")
        ax.tick_params(colors=LIGHT, labelsize=7.5)
        ax.xaxis.label.set_color(LIGHT)
        ax.yaxis.label.set_color(LIGHT)
        return ax

    # Panel layout
    ax_proto = mk(0, 0, 1, 4)  # Protocol descriptions (full width)
    ax_ampai = mk(1, 0, 1, 2)  # AmpAI recovery curves
    ax_force = mk(1, 2, 1, 2)  # Force amplitude recovery
    ax_freq = mk(2, 0, 1, 2)   # Cadence (frequency) recovery
    ax_compare = mk(2, 2, 1, 2)  # Week to milestone comparison
    ax_table = mk(3, 0, 1, 4)  # Summary table

    # ── Protocol descriptions ──────────────────────────────────────────────────
    ax_proto.axis("off")
    ax_proto.set_xlim(0, 1)
    ax_proto.set_ylim(0, 1)
    ax_proto.text(0.5, 0.95, "Phase 4D: Multi-level Rehabilitation Protocols",
                  ha="center", va="top", fontsize=13, fontweight="bold",
                  color=LIGHT, transform=ax_proto.transAxes)

    proto_y = 0.75
    for proto_name, spec in PROTOCOLS.items():
        letter = proto_name[0]
        label = spec["label"]
        desc = spec["description"]
        color = spec["color"]

        ax_proto.text(0.02, proto_y, f"{letter}.", fontsize=11, fontweight="bold",
                      color=color, transform=ax_proto.transAxes)
        ax_proto.text(0.08, proto_y, f"{label}",
                      fontsize=10, color=color, fontweight="bold",
                      transform=ax_proto.transAxes)
        ax_proto.text(0.08, proto_y - 0.08, f"  {desc}",
                      fontsize=8, color="#a0a8c0", style="italic",
                      transform=ax_proto.transAxes)
        proto_y -= 0.22

    # ── AmpAI recovery ─────────────────────────────────────────────────────────
    ax = ax_ampai
    ax.axhspan(0.10, 0.25, alpha=0.08, color=TH_COM)
    ax.axhspan(0.00, 0.10, alpha=0.12, color=TH_NORM)
    ax.axhline(0.25, color=TH_COM, lw=0.9, ls="--", alpha=0.7)
    ax.axhline(0.10, color=TH_NORM, lw=0.9, ls="--", alpha=0.7)

    for proto_name, data in results.items():
        ax.plot(weeks, data["amp_ai"], color=data["color"], lw=2.5,
                marker="o", ms=5, label=proto_name, alpha=0.85)

    ax.set_xlabel("Rehabilitation week", fontsize=9)
    ax.set_ylabel("Amplitude Asymmetry Index (AmpAI)", fontsize=9)
    ax.set_title("A) Gait Symmetry Recovery", fontsize=10, color=LIGHT, pad=6,
                 fontweight="bold")
    ax.text(-0.12, 1.05, "A", fontsize=13, fontweight="bold", color=ACCENT,
            transform=ax.transAxes)
    ax.legend(fontsize=7.5, facecolor="#1a1e2e", edgecolor="#3a3f5c",
              labelcolor=LIGHT, loc="upper right")
    ax.set_xlim(weeks[0], weeks[-1])
    ax.set_ylim(-0.02, max(d["amp_ai"].max() for d in results.values()) + 0.05)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    # ── Force amplitude recovery ───────────────────────────────────────────────
    ax = ax_force
    for proto_name, data in results.items():
        ax.plot(weeks, data["amp_L"], color=data["color"], lw=2.5,
                marker="s", ms=5, label=f"{proto_name[0]} left", alpha=0.85)

    ax.set_xlabel("Rehabilitation week", fontsize=9)
    ax.set_ylabel("Left Leg Peak Force [N]", fontsize=9)
    ax.set_title("B) Left Leg Force Recovery", fontsize=10, color=LIGHT, pad=6,
                 fontweight="bold")
    ax.text(-0.12, 1.05, "B", fontsize=13, fontweight="bold", color=ACCENT,
            transform=ax.transAxes)
    ax.legend(fontsize=6.5, facecolor="#1a1e2e", edgecolor="#3a3f5c",
              labelcolor=LIGHT, ncol=2, loc="lower right")
    ax.set_xlim(weeks[0], weeks[-1])
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    # ── Cadence (frequency) recovery ───────────────────────────────────────────
    ax = ax_freq
    healthy_freq = 0.63  # Hz from Phase 2
    ax.axhline(healthy_freq, color=TH_NORM, lw=1.2, ls="--", alpha=0.7,
               label=f"Healthy target ({healthy_freq:.2f} Hz)")

    for proto_name, data in results.items():
        ax.plot(weeks, data["freq"], color=data["color"], lw=2.5,
                marker="^", ms=5, label=proto_name, alpha=0.85)

    ax.set_xlabel("Rehabilitation week", fontsize=9)
    ax.set_ylabel("CPG Frequency [Hz]", fontsize=9)
    ax.set_title("C) Stepping Cadence Recovery", fontsize=10, color=LIGHT, pad=6,
                 fontweight="bold")
    ax.text(-0.12, 1.05, "C", fontsize=13, fontweight="bold", color=ACCENT,
            transform=ax.transAxes)
    ax.legend(fontsize=7, facecolor="#1a1e2e", edgecolor="#3a3f5c",
              labelcolor=LIGHT, loc="lower right")
    ax.set_xlim(weeks[0], weeks[-1])
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    # ── Week to milestone comparison (bar chart) ────────────────────────────────
    ax = ax_compare
    milestones_wk = []
    proto_labels = []
    for proto_name, data in results.items():
        norm_wk = next((weeks[i] for i, v in enumerate(data["amp_ai"]) if v < 0.10),
                       None)
        milestones_wk.append(norm_wk if norm_wk else 13)
        proto_labels.append(proto_name[0])

    colors_bar = [results[p]["color"] for p in results.keys()]
    bars = ax.bar(proto_labels, milestones_wk, color=colors_bar, alpha=0.85, width=0.6)

    # Annotate bars
    for i, (bar, wk) in enumerate(zip(bars, milestones_wk)):
        if wk <= 12:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                    f"wk {int(wk)}", ha="center", va="bottom", fontsize=9,
                    fontweight="bold", color=colors_bar[i])
        else:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                    "never", ha="center", va="bottom", fontsize=8, color="#a0a8c0")

    ax.axhline(4, color="#888", lw=1, ls=":", alpha=0.5, label="Physio baseline (wk 4)")
    ax.set_ylabel("Week to near-normal gait\n(AmpAI < 10%)", fontsize=9)
    ax.set_title("D) Rehabilitation Speed Comparison", fontsize=10, color=LIGHT, pad=6,
                 fontweight="bold")
    ax.text(-0.12, 1.05, "D", fontsize=13, fontweight="bold", color=ACCENT,
            transform=ax.transAxes)
    ax.set_ylim(0, 14)
    ax.set_yticks(np.arange(0, 15, 2))
    ax.legend(fontsize=7, facecolor="#1a1e2e", edgecolor="#3a3f5c", labelcolor=LIGHT)

    # ── Summary table ──────────────────────────────────────────────────────────
    ax = ax_table
    ax.axis("off")
    ax.set_title("E) Clinical Outcomes Summary", fontsize=10, color=LIGHT, pad=6,
                 fontweight="bold", loc="left")
    ax.text(-0.06, 0.95, "E", fontsize=13, fontweight="bold", color=ACCENT,
            transform=ax.transAxes)

    table_data = []
    table_data.append(["Protocol", "Week to AmpAI<10%", "Week to AmpAI<25%",
                       "Peak force (L)", "Avg. cadence"])

    for proto_name, data in results.items():
        norm_wk = next((weeks[i] for i, v in enumerate(data["amp_ai"]) if v < 0.10),
                       None)
        com_wk = next((weeks[i] for i, v in enumerate(data["amp_ai"]) if v < 0.25),
                      None)
        peak_force = data["amp_L"].max()
        avg_cadence = data["freq"].mean()

        norm_str = f"wk {int(norm_wk)}" if norm_wk else "—"
        com_str = f"wk {int(com_wk)}" if com_wk else "—"

        table_data.append([
            proto_name,
            norm_str,
            com_str,
            f"{peak_force:.1f} N",
            f"{avg_cadence:.2f} Hz"
        ])

    table = ax.table(cellText=table_data, cellLoc="center", loc="center",
                     bbox=[0, 0.1, 1, 0.85])
    table.auto_set_font_size(False)
    table.set_fontsize(7.5)
    table.scale(1, 2.2)

    # Style header row
    for i in range(len(table_data[0])):
        table[(0, i)].set_facecolor("#3a3f5c")
        table[(0, i)].set_text_props(weight="bold", color=LIGHT)

    # Color protocol rows
    for i, proto_name in enumerate(results.keys(), start=1):
        color = results[proto_name]["color"]
        for j in range(len(table_data[0])):
            table[(i, j)].set_facecolor(color)
            table[(i, j)].set_text_props(color="white", weight="bold")
            table[(i, j)].set_alpha(0.3)

    # Footnote
    ax.text(0.5, 0.02,
            "AmpAI < 10% = near-normal gait  │  AmpAI < 25% = community ambulation  │  "
            "Literature: Rejc 2017 (SCS); Khedr 2005 (TMS)",
            ha="center", va="bottom", fontsize=7, color="#a0a8c0",
            transform=ax.transAxes, style="italic")

    fig.suptitle(
        "Phase 4D: Multi-level Rehabilitation Protocols — TMS vs SCS vs Combined vs Sequential\n"
        "Comparing cortical (top-down) and spinal (bottom-up) stimulation approaches over 12 weeks",
        fontsize=12, fontweight="bold", color=LIGHT, y=0.97)

    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"[Plot] Saved → {out_path}")
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    np.random.seed(42)
    print("=" * 70)
    print("  TVB Phase 4D — Multi-level Rehabilitation Comparison")
    print("=" * 70)

    weeks = np.arange(0, 13, dtype=float)

    print(f"\n[1/3] Configuring multi-level CPG simulator...")
    print(f"  Baseline:  c_L={c_stroke:.3f} (stroke) vs c_healthy={c_healthy:.3f}")
    print(f"  Protocols:")
    for proto_name, spec in PROTOCOLS.items():
        print(f"    {proto_name:25s}  {spec['label']}")

    print(f"\n[2/3] Running four rehabilitation protocols × {len(weeks)} weeks each "
          f"({len(PROTOCOLS)*len(weeks)} CPG sims)...")
    results = run_multimodal_trajectories(weeks)

    print(f"\n  ── COMPARISON: Week to near-normal gait (AmpAI < 10%) ──")
    milestones = {}
    for proto_name, data in results.items():
        norm_wk = next((weeks[i] for i, v in enumerate(data["amp_ai"]) if v < 0.10),
                       None)
        milestones[proto_name] = norm_wk
        if norm_wk:
            print(f"    {proto_name:30s}  week {norm_wk:.0f}")
        else:
            print(f"    {proto_name:30s}  never reached (> 12 wks)")

    # Synergy analysis
    tms_only_wk = milestones.get("A: TMS only")
    scs_only_wk = milestones.get("B: SCS only")
    combined_wk = milestones.get("C: TMS + SCS combined")
    sequential_wk = milestones.get("D: Sequential (TMS→SCS)")

    if combined_wk and tms_only_wk and scs_only_wk:
        synergy = (tms_only_wk + scs_only_wk) - combined_wk
        print(f"\n  Synergy analysis (TMS + SCS combined):")
        print(f"    TMS alone → wk {tms_only_wk:.0f}")
        print(f"    SCS alone → wk {scs_only_wk:.0f}")
        print(f"    Combined  → wk {combined_wk:.0f}  (synergy = {synergy:.1f} wks earlier)")

    print(f"\n[3/3] Generating comparison figure...")
    fig_path = OUTPUT_DIR / "phase4d_multimodal_rehab.png"
    make_comparison_figure(results, weeks, fig_path)

    # Save metrics
    out_h5 = OUTPUT_DIR / "phase4d_recovery_metrics.h5"
    save_results_h5(results, weeks, out_h5)

    print(f"\n{'='*70}")
    print("  Phase 4D complete.")
    print(f"  Figure:  {fig_path}")
    print(f"  Metrics: {out_h5}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
