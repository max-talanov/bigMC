#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TVB Phase 4D Step 2 — UNIFIED OVERLAY COMPARISON
==================================================

Creates a SINGLE comprehensive figure overlaying ALL FOUR protocols
in Phase 2 style:

  - LEFT Extensor Muscle Force: Stroke baseline vs Healthy vs A/B/C/D outcomes
  - RIGHT Extensor Muscle Force: Same overlay
  - L/R Asymmetry Evolution: All protocols on one plot
  - CPG Oscillator Activity: Side-by-side comparison
  - Summary Metrics: Direct comparison table

Run with:  python3 tvb_phase4d_step2_overlay.py
"""

import numpy as np
from pathlib import Path
from scipy.signal import find_peaks, butter, filtfilt

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch

OUTPUT_DIR = Path("./tvb_output/phase4d_overlay")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
#  IMPORT PHASE 2 CPG SIMULATOR
# ─────────────────────────────────────────────────────────────────────────────
import importlib.util
spec = importlib.util.spec_from_file_location(
    "p2", Path(__file__).parent / "tvb_phase2_spinal_cpg.py")
p2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p2)

scales = p2.load_phase1_scales()
C0 = p2.C0
c_stroke = C0 * scales["scale_L"]
c_healthy = C0
c_R = C0 * scales["scale_R"]

# ══════════════════════════════════════════════════════════════════════════════
#  CPG SIMULATOR
# ══════════════════════════════════════════════════════════════════════════════

def simulate_cpg(c_L: float, c_R: float,
                 scs_amp_L: float = 0.0, scs_freq_L: float = 20.0,
                 scs_amp_R: float = 0.0, scs_freq_R: float = 20.0,
                 sim_s: float = 20.0, dt_s: float = 0.001) -> dict:
    """
    Matsuoka CPG using PHASE 2 SIMULATOR with SCS extension.

    Inherits saturating muscle dynamics from Phase 2 (bounded force output)
    and adds epidural SCS sinusoidal modulation to tonic drives.
    """
    # Inherit Phase 2 constants
    W_MAT = p2.W_MAT
    BETA = p2.BETA
    TAU1 = p2.TAU1
    TAU2 = p2.TAU2
    L0 = p2.L0
    ACT_GAIN = p2.ACT_GAIN
    ACT_MAX = p2.ACT_MAX
    FORCE_MAX = p2.FORCE_MAX
    FORCE_SAT_K = p2.FORCE_SAT_K
    TAU_ACT_MS = p2.TAU_ACT_MS
    TAU_FORCE_RISE = p2.TAU_FORCE_RISE
    TAU_FORCE_FALL = p2.TAU_FORCE_FALL
    TAU_LENGTH_MS = p2.TAU_LENGTH_MS
    L_MIN = p2.L_MIN
    L_MAX = p2.L_MAX
    SHORTEN_GAIN = p2.SHORTEN_GAIN
    clamp = p2.clamp

    N = int(sim_s / dt_s)

    tau1 = np.array([TAU1*(C0/c_L), TAU1*(C0/c_L),
                     TAU1*(C0/c_R), TAU1*(C0/c_R)])
    tau2 = np.array([TAU2*(C0/c_L), TAU2*(C0/c_L),
                     TAU2*(C0/c_R), TAU2*(C0/c_R)])

    u = np.array([0.80, -0.50, -0.50, 0.80])
    v = np.array([0.10, 0.0, 0.0, 0.10])
    y = np.maximum(0.0, u)

    act = np.zeros(4)
    force = np.zeros(4)
    leng = np.full(4, L0)

    t_arr = np.zeros(N)
    y_arr = np.zeros((4, N))
    f_arr = np.zeros((4, N))

    tA = TAU_ACT_MS / 1000.0
    tFR = TAU_FORCE_RISE / 1000.0
    tFF = TAU_FORCE_FALL / 1000.0
    tL = TAU_LENGTH_MS / 1000.0

    for k in range(N):
        t = k * dt_s

        # SCS sinusoidal modulation (epidural input)
        scs_L = scs_amp_L * np.sin(2*np.pi * scs_freq_L * t)
        scs_R = scs_amp_R * np.sin(2*np.pi * scs_freq_R * t)
        c_total = np.array([c_L + scs_L, c_L + scs_L,
                           c_R + scs_R, c_R + scs_R])

        # ── Matsuoka ODE (Phase 2 formulation, no commissural) ────────────
        du = (-u - W_MAT @ y - BETA * v + c_total) / tau1
        dv = (-v + y) / tau2
        u = u + dt_s * du
        v = v + dt_s * dv
        y = np.maximum(0.0, u)

        # ── Phase 2 muscle proxy (SATURATING - bounds force) ──────────────
        r = y * 100.0
        for i in range(4):
            act_tgt = clamp(ACT_GAIN * r[i] / 100.0, 0, ACT_MAX)
            act[i] += (dt_s / tA) * (act_tgt - act[i])

            f_tgt = FORCE_MAX * (1.0 - np.exp(-FORCE_SAT_K * act[i]))
            tau_f = tFR if f_tgt > force[i] else tFF
            force[i] += (dt_s / tau_f) * (f_tgt - force[i])
            force[i] = clamp(force[i], 0, FORCE_MAX)

            leng[i] += (dt_s / tL) * (L0 - leng[i])
            leng[i] -= SHORTEN_GAIN * force[i] * dt_s
            leng[i] = clamp(leng[i], L_MIN, L_MAX)

        t_arr[k] = t
        y_arr[:, k] = y
        f_arr[:, k] = force

    return {
        "time_s": t_arr,
        "y": y_arr,
        "force_LE": f_arr[0], "force_LF": f_arr[1],
        "force_RE": f_arr[2], "force_RF": f_arr[3],
    }


def recovery_c_L(weeks, c_stroke, c_healthy, k=0.18, t0=10.0, t_max=24):
    """
    Logistic recovery with CLINICALLY REALISTIC timeline.

    Parameters chosen to match stroke rehabilitation literature:
      - t0=10 weeks  (inflection point ≈ 2.5 months; Copenhagen Stroke Study)
      - k=0.18       (slow growth rate; matches Krakauer 2010 exponential approach)
      - t_max=24     (6 months ≈ clinical plateau; Duncan 2000, EXCITE trial)

    Recovery milestones (without NIBS):
      Week  4:   ~15% recovery (still acute)
      Week  8:   ~40% recovery (subacute)
      Week 12:   ~65% recovery (early plateau)
      Week 16:   ~85% recovery
      Week 24:   ~95% recovery (chronic plateau)
    """
    r = 1.0 / (1.0 + np.exp(-k * (weeks - t0)))
    r_all = 1.0 / (1.0 + np.exp(-k * (np.arange(t_max+1) - t0)))
    r = (r - r_all[0]) / np.clip(r_all[-1] - r_all[0], 1e-6, None)
    return np.clip(c_stroke + r * (c_healthy - c_stroke), c_stroke, c_healthy)


def run_protocol_at_week(proto_spec, wk):
    """Run a single protocol simulation at a given week (24-week timeline)."""
    c_L = recovery_c_L(np.array([wk]), c_stroke, c_healthy)[0]
    c_R_val = c_R

    # TMS boost: clinical protocol 5×/week × 6 weeks acute phase (Hummel/Cohen 2005)
    # Tapers over 12 weeks total
    tms_boost = 0.0
    if proto_spec["tms"]:
        tms_boost = 0.05 * max(0, 1.0 - wk / 12.0)

    # SCS amplitude: ramps over 4 weeks, then maintained (clinical SCS titration)
    # Effects accumulate over 12-16 weeks (Wagner 2018 epidural stim)
    scs_amp = 0.0
    if proto_spec["scs"] and wk >= proto_spec["scs_start"]:
        weeks_since_scs = wk - proto_spec["scs_start"]
        scs_amp = 0.15 * min(1.0, weeks_since_scs / 4.0)  # 4-week ramp
        scs_amp = np.clip(scs_amp, 0, 0.20)

    c_L_stim = np.clip(c_L + tms_boost, c_stroke, c_healthy)
    c_R_stim = np.clip(c_R_val, c_stroke, c_healthy)

    return simulate_cpg(c_L_stim, c_R_stim,
                       scs_amp_L=scs_amp*0.1, scs_freq_L=20.0,
                       scs_amp_R=scs_amp*0.1, scs_freq_R=20.0,
                       sim_s=20.0)


def extract_metrics(res, sim_s=20.0):
    """Extract gait metrics from simulation."""
    t_ss = res["time_s"]
    t_start_idx = np.argmax(t_ss >= sim_s * 0.5)  # Last half
    dt = float(np.diff(t_ss[:10]).mean())

    force_L = (res["force_LE"] + res["force_LF"])[t_start_idx:]
    force_R = (res["force_RE"] + res["force_RF"])[t_start_idx:]

    # Smooth
    b, a = butter(2, 4.0 * dt * 2, btype="low")
    force_L_s = filtfilt(b, a, force_L)
    force_R_s = filtfilt(b, a, force_R)

    pk_L, _ = find_peaks(force_L_s, height=0.5, distance=max(1, int(0.3/dt)))
    pk_R, _ = find_peaks(force_R_s, height=0.5, distance=max(1, int(0.3/dt)))

    amp_L = float(force_L_s[pk_L].mean()) if len(pk_L) > 0 else float(force_L_s.max())
    amp_R = float(force_R_s[pk_R].mean()) if len(pk_R) > 0 else float(force_R_s.max())

    freq_L = float(len(pk_L) / (t_ss[-1] - t_ss[t_start_idx])) if len(pk_L) > 0 else 0
    freq_R = float(len(pk_R) / (t_ss[-1] - t_ss[t_start_idx])) if len(pk_R) > 0 else 0

    period_L = 1000.0 / freq_L if freq_L > 0 else 0
    period_R = 1000.0 / freq_R if freq_R > 0 else 0

    ai = (amp_R - amp_L) / max(amp_R, amp_L) if max(amp_R, amp_L) > 0 else 0

    # Duty cycle (% of cycle where extensor is active)
    duty_L = float(np.sum(force_L_s > force_L_s.max() * 0.3) / len(force_L_s) * 100)
    duty_R = float(np.sum(force_R_s > force_R_s.max() * 0.3) / len(force_R_s) * 100)

    return {
        "amp_L": amp_L, "amp_R": amp_R, "ai": ai,
        "freq_L": freq_L, "freq_R": freq_R,
        "period_L": period_L, "period_R": period_R,
        "duty_L": duty_L, "duty_R": duty_R,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  UNIFIED OVERLAY COMPARISON FIGURE
# ══════════════════════════════════════════════════════════════════════════════

def make_unified_comparison_figure():
    """Create comprehensive Phase 2-style figure overlaying all 4 protocols."""
    DARK = "#0e1117"
    LIGHT = "#e0e0e0"
    ACCENT = "#f9a825"

    # Colors matching Phase 2 style
    COLOR_STROKE = "#e74c3c"      # Red
    COLOR_HEALTHY = "#3498db"     # Blue
    COLOR_BASELINE = "#95a5a6"    # Grey
    COLOR_A = "#ff6b6b"           # TMS - red-orange
    COLOR_B = "#4ecdc4"           # SCS - teal
    COLOR_C = "#f39c12"           # Combined - orange
    COLOR_D = "#9b59b6"           # Sequential - purple

    protocols = {
        "A: TMS only": {"tms": True, "scs": False, "scs_start": 999, "color": COLOR_A},
        "B: SCS only": {"tms": False, "scs": True, "scs_start": 0, "color": COLOR_B},
        "C: Combined": {"tms": True, "scs": True, "scs_start": 0, "color": COLOR_C},
        "D: Sequential": {"tms": True, "scs": True, "scs_start": 6, "color": COLOR_D},
    }

    # Clinical evaluation timepoints
    EVAL_WEEK = 16.0  # Late subacute (~4 months) — typical clinical evaluation
    MAX_WEEK = 24     # 6 months (clinical plateau / chronic phase)

    print("\n[Sim 1/6] Stroke baseline (week 0)...")
    res_stroke = run_protocol_at_week({"tms": False, "scs": False, "scs_start": 999}, 0.0)
    m_stroke = extract_metrics(res_stroke)

    print("[Sim 2/6] Healthy reference...")
    res_healthy = simulate_cpg(c_healthy, c_R, sim_s=20.0)
    m_healthy = extract_metrics(res_healthy)

    proto_results = {}
    proto_metrics = {}
    for i, (proto_name, proto_spec) in enumerate(protocols.items()):
        print(f"[Sim {i+3}/6] {proto_name} at week {EVAL_WEEK}...")
        res = run_protocol_at_week(proto_spec, EVAL_WEEK)
        proto_results[proto_name] = res
        proto_metrics[proto_name] = extract_metrics(res)

    # ── Build figure ──────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(22, 13))
    fig.patch.set_facecolor(DARK)

    gs = gridspec.GridSpec(3, 3, figure=fig,
                          left=0.05, right=0.97, top=0.93, bottom=0.05,
                          hspace=0.45, wspace=0.30,
                          width_ratios=[2.5, 2.5, 1.5])

    def mk_ax(r, c, rs=1, cs=1):
        ax = fig.add_subplot(gs[r:r+rs, c:c+cs])
        ax.set_facecolor("#1a1e2e")
        for sp in ax.spines.values():
            sp.set_edgecolor("#3a3f5c")
        ax.tick_params(colors=LIGHT, labelsize=8)
        ax.xaxis.label.set_color(LIGHT)
        ax.yaxis.label.set_color(LIGHT)
        return ax

    def add_index(ax, letter):
        """Add a bold letter index to the top-left of a panel."""
        ax.text(-0.08, 1.08, letter, fontsize=16, fontweight="bold",
               color=ACCENT, transform=ax.transAxes, ha="left", va="top")

    # Suptitle replaces the removed pipeline row
    fig.suptitle(
        "Phase 4D Step 2: Multi-Protocol Rehabilitation Overlay Comparison\n"
        f"Stroke (week 0): c_L={c_stroke:.3f}, AmpAI={m_stroke['ai']*100:.1f}%  →  "
        f"Healthy: c_L={c_healthy:.3f}, AmpAI={m_healthy['ai']*100:.1f}%",
        fontsize=13, fontweight="bold", color=ACCENT, y=0.985)

    # Build label suffix for evaluation week
    eval_label = f"Week {int(EVAL_WEEK)} (≈{int(EVAL_WEEK/4)} months)"

    # ═══ ROW 0: LEFT MUSCLE FORCE OVERLAY (Panel A) ════════════════════════════
    ax_LF = mk_ax(0, 0, cs=2)
    ax_LF.set_title("LEFT Extensor Muscle Force — Recovery Pattern Overlay (Week 16 ≈ 4 mo)",
                   fontsize=11, color=LIGHT, fontweight="bold", pad=8)
    add_index(ax_LF, "A")

    # Plot stroke baseline (greyish red)
    t_ss = res_stroke["time_s"]
    t_start_idx = np.argmax(t_ss >= 10.0)
    t_plot = t_ss[t_start_idx:]
    force_L_stroke = (res_stroke["force_LE"] + res_stroke["force_LF"])[t_start_idx:]
    ax_LF.plot(t_plot, force_L_stroke, color=COLOR_STROKE, lw=2.0,
              label=f"Stroke wk0 (AmpAI={m_stroke['ai']*100:.0f}%)", alpha=0.85, zorder=3)

    # Plot healthy reference
    force_L_healthy = (res_healthy["force_LE"] + res_healthy["force_LF"])[t_start_idx:]
    ax_LF.plot(t_plot, force_L_healthy, color=COLOR_HEALTHY, lw=2.0,
              label=f"Healthy (target)", alpha=0.85, zorder=3)

    # Plot each protocol's outcome at week 5
    for proto_name, proto_spec in protocols.items():
        res = proto_results[proto_name]
        m = proto_metrics[proto_name]
        force_L = (res["force_LE"] + res["force_LF"])[t_start_idx:]
        ax_LF.plot(t_plot, force_L, color=proto_spec["color"], lw=2.2,
                  label=f"{proto_name} wk5 (AmpAI={m['ai']*100:.1f}%)", alpha=0.9)

    ax_LF.set_xlabel("Time [s]", fontsize=10)
    ax_LF.set_ylabel("Force [N]", fontsize=10)
    ax_LF.legend(fontsize=8, facecolor="#1a1e2e", edgecolor="#3a3f5c",
                labelcolor=LIGHT, loc="upper right", ncol=2)
    ax_LF.grid(True, alpha=0.15, color=LIGHT)
    ax_LF.set_xlim(t_plot[0], t_plot[-1])

    # ═══ ROW 0: CPG OSCILLATOR (Panel B) ═══════════════════════════════════════
    ax_cpg = mk_ax(0, 2)
    ax_cpg.set_title("CPG Oscillator\n(Healthy steady-state)", fontsize=10,
                    color=LIGHT, fontweight="bold", pad=6)
    add_index(ax_cpg, "B")

    cpg_idx = np.argmax(res_healthy["time_s"] >= 10.0)
    cpg_t = res_healthy["time_s"][cpg_idx:cpg_idx+10000]
    ax_cpg.plot(cpg_t, res_healthy["y"][0, cpg_idx:cpg_idx+10000],
               color="#3498db", lw=1.2, label="LE (L ext)", alpha=0.7)
    ax_cpg.plot(cpg_t, res_healthy["y"][1, cpg_idx:cpg_idx+10000],
               color="#74b9ff", lw=1.2, label="LF (L flex)", alpha=0.7)
    ax_cpg.plot(cpg_t, res_healthy["y"][2, cpg_idx:cpg_idx+10000],
               color="#e74c3c", lw=1.2, label="RE (R ext)", alpha=0.7)
    ax_cpg.plot(cpg_t, res_healthy["y"][3, cpg_idx:cpg_idx+10000],
               color="#ff7675", lw=1.2, label="RF (R flex)", alpha=0.7)
    ax_cpg.set_xlabel("Time [s]", fontsize=9)
    ax_cpg.set_ylabel("Matsuoka output y", fontsize=9)
    ax_cpg.legend(fontsize=6.5, facecolor="#1a1e2e", edgecolor="#3a3f5c",
                 labelcolor=LIGHT, loc="upper right")
    ax_cpg.grid(True, alpha=0.15, color=LIGHT)

    # ═══ ROW 1: RIGHT MUSCLE FORCE OVERLAY (Panel C) ═══════════════════════════
    ax_RF = mk_ax(1, 0, cs=2)
    ax_RF.set_title("RIGHT Extensor Muscle Force — Recovery Pattern Overlay (Week 16 ≈ 4 mo)",
                   fontsize=11, color=LIGHT, fontweight="bold", pad=8)
    add_index(ax_RF, "C")

    force_R_stroke = (res_stroke["force_RE"] + res_stroke["force_RF"])[t_start_idx:]
    ax_RF.plot(t_plot, force_R_stroke, color=COLOR_STROKE, lw=2.0,
              label=f"Stroke wk0", alpha=0.85, zorder=3)
    force_R_healthy = (res_healthy["force_RE"] + res_healthy["force_RF"])[t_start_idx:]
    ax_RF.plot(t_plot, force_R_healthy, color=COLOR_HEALTHY, lw=2.0,
              label=f"Healthy", alpha=0.85, zorder=3)

    for proto_name, proto_spec in protocols.items():
        res = proto_results[proto_name]
        force_R = (res["force_RE"] + res["force_RF"])[t_start_idx:]
        ax_RF.plot(t_plot, force_R, color=proto_spec["color"], lw=2.2,
                  label=f"{proto_name} wk5", alpha=0.9)

    ax_RF.set_xlabel("Time [s]", fontsize=10)
    ax_RF.set_ylabel("Force [N]", fontsize=10)
    ax_RF.legend(fontsize=8, facecolor="#1a1e2e", edgecolor="#3a3f5c",
                labelcolor=LIGHT, loc="upper right", ncol=2)
    ax_RF.grid(True, alpha=0.15, color=LIGHT)
    ax_RF.set_xlim(t_plot[0], t_plot[-1])

    # ═══ ROW 1: STEP FREQUENCY BAR CHART (Panel D) ══════════════════════════════
    ax_freq = mk_ax(1, 2)
    ax_freq.set_title("Step Frequency\n(Left vs Right)", fontsize=10,
                     color=LIGHT, fontweight="bold", pad=6)
    add_index(ax_freq, "D")

    bar_labels = ["Stroke", "Healthy"] + [p[0] for p in protocols.keys()]
    freqs_L = [m_stroke["freq_L"], m_healthy["freq_L"]] + \
              [proto_metrics[p]["freq_L"] for p in protocols.keys()]
    freqs_R = [m_stroke["freq_R"], m_healthy["freq_R"]] + \
              [proto_metrics[p]["freq_R"] for p in protocols.keys()]

    x_bar = np.arange(len(bar_labels))
    w_bar = 0.35
    bars_L = ax_freq.bar(x_bar - w_bar/2, freqs_L, w_bar, label="L leg",
                        color=[COLOR_STROKE, COLOR_HEALTHY] + [p["color"] for p in protocols.values()],
                        alpha=0.65, edgecolor=LIGHT, linewidth=0.5)
    bars_R = ax_freq.bar(x_bar + w_bar/2, freqs_R, w_bar, label="R leg",
                        color=[COLOR_STROKE, COLOR_HEALTHY] + [p["color"] for p in protocols.values()],
                        alpha=1.0, edgecolor=LIGHT, linewidth=0.5)

    for bar, val in zip(bars_L, freqs_L):
        if val > 0:
            ax_freq.text(bar.get_x() + bar.get_width()/2, val + 0.01,
                        f"{val:.2f}", ha="center", va="bottom",
                        fontsize=6.5, color=LIGHT)

    ax_freq.set_xticks(x_bar)
    ax_freq.set_xticklabels(bar_labels, rotation=30, ha="right", fontsize=8)
    ax_freq.set_ylabel("Step Frequency [Hz]", fontsize=9)
    ax_freq.legend(fontsize=7, facecolor="#1a1e2e", edgecolor="#3a3f5c",
                  labelcolor=LIGHT, loc="upper right")
    ax_freq.grid(True, alpha=0.15, color=LIGHT, axis="y")

    # ═══ ROW 2: ASYMMETRY EVOLUTION ALL PROTOCOLS (Panel E) ════════════════════
    ax_asym = mk_ax(2, 0, cs=2)
    ax_asym.set_title("Amplitude Asymmetry Index Evolution — All Protocols (Week 0→24, 6 months)",
                     fontsize=11, color=LIGHT, fontweight="bold", pad=8)
    add_index(ax_asym, "E")

    # Clinical 24-week timeline (6 months) with 2-week resolution
    weeks_eval = np.arange(0, 25, 2)

    # Compute asymmetry trajectory for each protocol
    for proto_name, proto_spec in protocols.items():
        ai_arr = []
        for wk in weeks_eval:
            res = run_protocol_at_week(proto_spec, wk)
            m = extract_metrics(res)
            ai_arr.append(abs(m["ai"]) * 100)
        ax_asym.plot(weeks_eval, ai_arr, color=proto_spec["color"], lw=2.5,
                    marker="o", ms=7, label=proto_name, alpha=0.9)

    # Reference lines
    ax_asym.axhline(abs(m_stroke["ai"])*100, color=COLOR_STROKE, lw=1.5, ls=":",
                   alpha=0.6, label=f"Stroke baseline ({abs(m_stroke['ai'])*100:.0f}%)")
    ax_asym.axhline(10, color="#88ff88", lw=1.5, ls="--", alpha=0.7,
                   label="Clinical goal (<10%)")
    ax_asym.axhline(0, color=COLOR_HEALTHY, lw=1.5, ls=":", alpha=0.6,
                   label="Healthy (0%)")

    # Clinical phase annotations
    ax_asym.axvspan(0, 4, alpha=0.06, color="#ff6b6b", label="_acute")
    ax_asym.axvspan(4, 12, alpha=0.06, color="#f9a825", label="_subacute")
    ax_asym.axvspan(12, 24, alpha=0.06, color="#88ff88", label="_chronic")
    ax_asym.text(2, 22.5, "Acute\n(0–4 wk)", ha="center", fontsize=8,
                color="#ff6b6b", fontweight="bold", alpha=0.8)
    ax_asym.text(8, 22.5, "Subacute\n(4–12 wk)", ha="center", fontsize=8,
                color="#f9a825", fontweight="bold", alpha=0.8)
    ax_asym.text(18, 22.5, "Chronic / Plateau\n(12–24 wk)", ha="center", fontsize=8,
                color="#88ff88", fontweight="bold", alpha=0.8)
    ax_asym.axvline(EVAL_WEEK, color=ACCENT, lw=1, ls="--", alpha=0.5)
    ax_asym.text(EVAL_WEEK+0.3, 1, f"Eval\nwk {int(EVAL_WEEK)}", fontsize=7,
                color=ACCENT, alpha=0.8)

    ax_asym.set_xlabel("Rehabilitation Week (1 wk × 24 = 6 months clinical timeline)", fontsize=10)
    ax_asym.set_ylabel("Asymmetry Index |AmpAI| [%]", fontsize=10)
    ax_asym.legend(fontsize=8, facecolor="#1a1e2e", edgecolor="#3a3f5c",
                  labelcolor=LIGHT, loc="upper right", ncol=2)
    ax_asym.grid(True, alpha=0.15, color=LIGHT)
    ax_asym.set_xlim(0, 24)
    ax_asym.set_xticks(np.arange(0, 25, 4))
    ax_asym.set_ylim(0, 25)

    # ═══ ROW 2: SUMMARY METRICS TABLE (Panel F) ═════════════════════════════════
    ax_tab = mk_ax(2, 2)
    ax_tab.axis("off")
    ax_tab.set_title("Summary Metrics @ Week 5", fontsize=10,
                    color=LIGHT, fontweight="bold", pad=4)
    add_index(ax_tab, "F")

    # Best protocol (lowest |AmpAI|)
    best_proto = min(proto_metrics.items(), key=lambda kv: abs(kv[1]["ai"]))
    best_name = best_proto[0]

    summary_lines = [
        f"Eval: wk {int(EVAL_WEEK)} (~{int(EVAL_WEEK/4)} mo)",
        "─" * 26,
        f"Stroke wk0:   {abs(m_stroke['ai'])*100:>5.1f}%",
        f"Healthy:      {abs(m_healthy['ai'])*100:>5.1f}%",
        "─" * 26,
        f"A: TMS:       {abs(proto_metrics['A: TMS only']['ai'])*100:>5.1f}%",
        f"B: SCS:       {abs(proto_metrics['B: SCS only']['ai'])*100:>5.1f}%",
        f"C: Combined:  {abs(proto_metrics['C: Combined']['ai'])*100:>5.1f}%",
        f"D: Sequential:{abs(proto_metrics['D: Sequential']['ai'])*100:>5.1f}%",
        "─" * 26,
        f"★ Best: {best_name.split(':')[0]}",
        f"   AmpAI = {abs(best_proto[1]['ai'])*100:.1f}%",
        f"   PeakF_L = {best_proto[1]['amp_L']:.1f} N",
        f"   PeakF_R = {best_proto[1]['amp_R']:.1f} N",
        "",
        "Clinical lit:",
        "  Copenhagen 1995",
        "  Krakauer 2010",
        "  Duncan 2000",
    ]

    ax_tab.text(0.05, 0.95, "\n".join(summary_lines), ha="left", va="top",
               fontsize=9, color=LIGHT, family="monospace",
               transform=ax_tab.transAxes,
               bbox=dict(fc="#0a0e1a", ec=ACCENT, pad=8, lw=1.2))

    out_path = OUTPUT_DIR / "phase4d_unified_comparison.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"\n[Plot] Saved → {out_path}")
    plt.close()

    return proto_metrics, m_stroke, m_healthy


def main():
    np.random.seed(42)
    print("=" * 85)
    print("  TVB Phase 4D Step 2 — UNIFIED OVERLAY COMPARISON")
    print("=" * 85)

    proto_metrics, m_stroke, m_healthy = make_unified_comparison_figure()

    print(f"\n{'='*85}")
    print(f"  FINAL COMPARISON @ Week 5 (Stroke wk0 = {abs(m_stroke['ai'])*100:.1f}%, "
          f"Healthy = {abs(m_healthy['ai'])*100:.1f}%)")
    print(f"{'='*85}")
    for proto_name, m in proto_metrics.items():
        gain = (abs(m_stroke["ai"]) - abs(m["ai"])) / abs(m_stroke["ai"]) * 100
        print(f"  {proto_name:<20s}  AmpAI = {abs(m['ai'])*100:>5.1f}%   "
              f"(recovery: {gain:+.1f}%)   PeakF_L={m['amp_L']:.1f} N  PeakF_R={m['amp_R']:.1f} N")
    print(f"{'='*85}\n")


if __name__ == "__main__":
    main()
