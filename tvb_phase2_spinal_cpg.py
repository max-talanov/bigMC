#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TVB Motor Cortex + Spinal CPG — Phase 2 (Matsuoka Rate Model)
=============================================================

Connects the TVB Phase 1 cortical output to a bilateral spinal CPG and
demonstrates how a unilateral M1 stroke disrupts locomotor symmetry.

CPG model: Matsuoka (1985) half-centre oscillator
--------------------------------------------------
The Matsuoka oscillator is the **rate-model theoretical basis** of the
tinyCPG spiking NEST network. It replaces the NEST Izhikevich populations
with mean-field firing-rate neurons while reproducing the same network
topology (local E/F reciprocal inhibition, commissural L↔R inhibition).

    τ₁ · du_i/dt  =  −u_i  −  Σⱼ wᵢⱼ·yⱼ  −  β·v_i  +  c_i(t)
    τ₂ · dv_i/dt  =  −v_i  +  y_i
    y_i           =  max(0, u_i)          # rectified linear output

Neurons:  LE (Left Extensor)   LF (Left Flexor)
          RE (Right Extensor)  RF (Right Flexor)

Inhibitory connections:
  LE ↔ LF  (left  half-centre, reciprocal, weight w_local)
  RE ↔ RF  (right half-centre, reciprocal, weight w_local)
  LE ↔ RE  (commissural, anti-phase L/R, weight w_comm)
  LF ↔ RF  (commissural, anti-phase L/R, weight w_comm)

Cortical–spinal interface
--------------------------
TVB Phase 1 provides mean CST drives (Hz) for left and right hemispheres.
These scale the tonic background input c_i(t) per side:

    c_L = c₀ × (cst_drive_L / cst_drive_healthy_L)
    c_R = c₀ × (cst_drive_R / cst_drive_healthy_R)

A stroke-silenced left M1 reduces c_L by ~35.6 %, which reduces the
left-leg oscillation amplitude and shortens its step period — matching
the hemiparetic gait pattern seen clinically.

Muscle proxy (identical to tinyCPG)
-------------------------------------
The Matsuoka outputs y_i are converted to muscle force via the same
first-order activation / saturating-force / length dynamics used in
cpg_2legs.py, preserving direct comparison with the NEST output.

Outputs (tvb_output/phase2/):
  matsuoka_<condition>_<ts>.h5
  phase2_comparison.png

Run with:  python3 tvb_phase2_spinal_cpg.py
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
from matplotlib.ticker import MaxNLocator
from scipy.signal import find_peaks, butter, filtfilt
from scipy.ndimage import uniform_filter1d

OUTPUT_DIR = Path("./tvb_output/phase2")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
#  MATSUOKA OSCILLATOR PARAMETERS  (tuned for ~1 Hz human walking)
# ══════════════════════════════════════════════════════════════════════════════
TAU1   = 0.08   # s — membrane time constant  (τ₂=6τ₁ → ~1 Hz walking)
TAU2   = 0.48   # s — adaptation time constant  (= 6 × τ₁, Matsuoka 1985)
BETA   = 2.5    # adaptation gain
W_LOC  = 2.5    # ipsilateral E↔F inhibitory weight
W_COM  = 0.3    # commissural E↔E and F↔F inhibitory weight
# Oscillation condition: W_LOC + W_COM < 1 + β   (2.8 < 3.5 ✓)
# W_COM reduced from 1.0→0.3 to prevent right-leg over-compensation:
# when left CST drive falls, the contralateral half-centre should
# speed up by <20% (clinical data), not +57% as seen with W_COM=0.8.
C0     = 1.5    # nominal tonic input (healthy, both sides)

# Neurons: 0=LE, 1=LF, 2=RE, 3=RF
# Inhibitory weight matrix  W[i,j] = inhibition from j onto i
def build_W():
    W = np.zeros((4, 4))
    # local half-centres
    W[0, 1] = W[1, 0] = W_LOC   # LE↔LF
    W[2, 3] = W[3, 2] = W_LOC   # RE↔RF
    # commissural (enforce L/R alternation)
    W[0, 2] = W[2, 0] = W_COM   # LE↔RE
    W[1, 3] = W[3, 1] = W_COM   # LF↔RF
    return W

W_MAT = build_W()

# ══════════════════════════════════════════════════════════════════════════════
#  MUSCLE PROXY PARAMETERS  (identical to tinyCPG cpg_2legs.py)
# ══════════════════════════════════════════════════════════════════════════════
TAU_ACT_MS     = 80.0
ACT_GAIN       = 0.6        # maps Matsuoka output y ∈ [0,~2] → activation target
ACT_MAX        = 1.2
TAU_FORCE_RISE = 140.0      # ms
TAU_FORCE_FALL = 60.0       # ms
FORCE_MAX      = 25.0
FORCE_SAT_K    = 2.5
TAU_LENGTH_MS  = 260.0
L0             = 1.0
L_MIN, L_MAX   = 0.5, 2.0
SHORTEN_GAIN   = 0.010

# ══════════════════════════════════════════════════════════════════════════════
#  MOTONEURON-POOL RECRUITMENT  (bio-plausibility: flaccid paralysis threshold)
# ══════════════════════════════════════════════════════════════════════════════
# Physiology: voluntary/CPG force depends on the fraction of the α-motoneuron
# pool that the descending drive can recruit above firing threshold (Henneman
# size principle).  Below a critical drive, too few motor units are recruited
# → negligible force = the flaccid paralysis of acute hemiparesis.
#
# We model recruitment as a sigmoid of the side's descending drive c:
#     R(c) = 1 / (1 + exp(-K_RECRUIT · (c − C_CRIT)))
# applied as a multiplicative gain on the motoneuron activation target.
#
# Tuning (C_CRIT=0.55, K_RECRUIT=12) is deliberately chosen so that the
# previously-validated stroke drive (c≈0.97, "mild") is essentially
# unaffected (R≈0.99), while moderate/severe/complete lesions (lower drive)
# cross the threshold into flaccidity:
#     c=1.50 (healthy)   → R≈1.00   full force
#     c=0.97 (mild)      → R≈0.99   ~unchanged (back-compatible)
#     c=0.60 (moderate)  → R≈0.65
#     c=0.45 (severe)    → R≈0.23
#     c=0.30 (complete)  → R≈0.03   flaccid (force ≈ 0)
C_CRIT        = 0.55    # critical drive for motoneuron recruitment
K_RECRUIT     = 12.0    # steepness of the recruitment sigmoid
RECRUIT_ON    = True    # master switch (False reproduces pre-recruitment model)

# ── Clinical stroke severity → acute descending-drive scale (fraction of C0) ──
# Acute deficit reflects more than the chronic CST drop: spinal shock,
# diaschisis and peri-lesional oedema transiently suppress descending output.
# Recovery (Phase 3/4) lets the drive climb back up through the recruitment
# threshold, restoring force over weeks.
STROKE_SEVERITY = {
    "healthy":  1.00,   # c_L = 1.50  → full force
    "mild":     0.65,   # c_L ≈ 0.97  → ~preserved (matches Phase-1 TVB deficit)
    "moderate": 0.40,   # c_L ≈ 0.60  → ~65% force, clear weakness
    "severe":   0.30,   # c_L ≈ 0.45  → ~23% force, near-flaccid
    "complete": 0.20,   # c_L ≈ 0.30  → ~3% force, flaccid paralysis
}


def recruitment_gain(c: float,
                     c_crit: float = C_CRIT,
                     k: float = K_RECRUIT) -> float:
    """Sigmoidal motoneuron-pool recruitment as a fraction [0,1] of maximal
    recruitable force, given descending drive c (Henneman size principle)."""
    return float(1.0 / (1.0 + np.exp(-k * (c - c_crit))))

# ══════════════════════════════════════════════════════════════════════════════
#  LOAD PHASE 1 TVB RESULTS
# ══════════════════════════════════════════════════════════════════════════════
def load_phase1_scales() -> dict:
    comp = Path("./tvb_output/comparison")
    hf   = sorted(comp.glob("tvb_healthy_*.h5"))
    sf   = sorted(comp.glob("tvb_stroke_*.h5"))

    if not hf or not sf:
        print("[Phase2] WARNING: no Phase 1 files found; using fallback scales.")
        return dict(healthy_L=1.28, healthy_R=1.32,
                    stroke_L=0.825, stroke_R=1.32,
                    scale_L=0.644, scale_R=1.0)

    with h5py.File(hf[-1]) as f:
        hl = float(f["corticospinal_drive_left_hz"][:].mean())
        hr = float(f["corticospinal_drive_right_hz"][:].mean())
    with h5py.File(sf[-1]) as f:
        sl = float(f["corticospinal_drive_left_hz"][:].mean())
        sr = float(f["corticospinal_drive_right_hz"][:].mean())

    scL = sl / hl if hl else 0.0
    scR = sr / hr if hr else 1.0
    print(f"  Healthy CST  L={hl:.4f}  R={hr:.4f} Hz")
    print(f"  Stroke  CST  L={sl:.4f}  R={sr:.4f} Hz")
    print(f"  Scale   L={scL:.4f}  R={scR:.4f}  "
          f"(left deficit: {(1-scL)*100:.1f}%)")
    return dict(healthy_L=hl, healthy_R=hr, stroke_L=sl, stroke_R=sr,
                scale_L=scL, scale_R=scR)


# ══════════════════════════════════════════════════════════════════════════════
#  MATSUOKA SIMULATOR
# ══════════════════════════════════════════════════════════════════════════════
def clamp(x, lo, hi):
    return float(np.clip(x, lo, hi))


def simulate(c_L: float, c_R: float,
             sim_s: float = 20.0, dt_s: float = 0.001,
             recruit: bool = RECRUIT_ON) -> dict:
    """
    Run the bilateral Matsuoka CPG + muscle proxies.

    Parameters
    ----------
    c_L, c_R : tonic inputs to left / right neurons
    sim_s    : total simulation time [s]
    dt_s     : integration step [s]
    recruit  : if True, apply the motoneuron-pool recruitment gain so that
               low descending drive yields flaccid (≈0) force on that side.

    Returns
    -------
    dict with keys: time_s, y (4×N array), force_e/f for L and R,
    act_e/f, len_e/f, and the input drives c_L/c_R.

    Drive-dependent time constants (Patch 2 — dose-response fix)
    -----------------------------------------------------------
    Higher cortical drive speeds up motoneuron dynamics → shorter τ → faster
    stepping cadence.  We scale τ₁ and τ₂ per neuron pair:

        τ_eff = τ_nominal × (C₀ / c_side)

    So c_healthy=1.5 → τ_eff=τ_nominal; c_baseline=1.05 → τ_eff=1.43×τ_nominal
    → baseline steps ~30% slower, creating the c→frequency dose-response.
    """
    N    = int(sim_s / dt_s)
    c    = np.array([c_L, c_L, c_R, c_R])   # LE LF RE RF

    # Per-neuron time constants: inversely scaled by normalised drive
    # Neurons 0,1 = left;  neurons 2,3 = right
    tau1 = np.array([TAU1*(C0/c_L), TAU1*(C0/c_L),
                     TAU1*(C0/c_R), TAU1*(C0/c_R)])
    tau2 = np.array([TAU2*(C0/c_L), TAU2*(C0/c_L),
                     TAU2*(C0/c_R), TAU2*(C0/c_R)])

    # State — asymmetric seed to break symmetry and enter the limit cycle.
    # Matsuoka theory: oscillation requires initial y_active > c / W_LOC
    # (so that the inactive neuron is truly silenced).  With c_max=1.5 and
    # W_LOC=2.5 the threshold is 0.60, so we start LE/RF at 0.80 > 0.60.
    # Bipedal pattern: LE + RF co-active (left stance / right swing phase).
    u    = np.array([ 0.80, -0.50, -0.50,  0.80])   # LE, LF, RE, RF
    v    = np.array([ 0.10,  0.00,  0.00,  0.10])   # mild adaptation pre-warm
    y    = np.maximum(0.0, u)

    # Muscle state per side: [L_ext, L_flex, R_ext, R_flex]
    act   = np.zeros(4)
    force = np.zeros(4)
    leng  = np.full(4, L0)

    # Output buffers
    t_arr  = np.zeros(N)
    y_arr  = np.zeros((4, N))
    a_arr  = np.zeros((4, N))
    f_arr  = np.zeros((4, N))
    l_arr  = np.zeros((4, N))

    tA  = TAU_ACT_MS    / 1000.0
    tFR = TAU_FORCE_RISE / 1000.0
    tFF = TAU_FORCE_FALL / 1000.0
    tL  = TAU_LENGTH_MS  / 1000.0

    # Per-side motoneuron-pool recruitment gain (constant over the run for a
    # given drive).  Neurons 0,1 = left; 2,3 = right.  recruit=False → all 1.0.
    if recruit:
        gL, gR = recruitment_gain(c_L), recruitment_gain(c_R)
    else:
        gL = gR = 1.0
    rec_gain = np.array([gL, gL, gR, gR])

    for k in range(N):
        t = k * dt_s

        # ── Matsuoka ODE (Euler, per-neuron τ) ───────────────────────────
        du = (-u - W_MAT @ y - BETA * v + c) / tau1
        dv = (-v + y) / tau2
        u  = u + dt_s * du
        v  = v + dt_s * dv
        y  = np.maximum(0.0, u)

        # ── Muscle proxy ─────────────────────────────────────────────────
        # Map Matsuoka output [0, ~2] to a firing-rate proxy [0, ~200 Hz]
        r = y * 100.0
        for i in range(4):
            # Motoneuron-pool recruitment gain (per side): neurons 0,1 = left,
            # 2,3 = right.  Below C_CRIT the limb is flaccid (force ≈ 0).
            rec = rec_gain[i]
            # Activation (low-pass), scaled by recruitable motor-unit fraction
            act_tgt = clamp(ACT_GAIN * r[i] / 100.0 * rec, 0, ACT_MAX)
            act[i]  += (dt_s / tA) * (act_tgt - act[i])

            # Force (saturating, asymmetric rise/decay)
            f_tgt = FORCE_MAX * (1.0 - np.exp(-FORCE_SAT_K * act[i]))
            tau_f = tFR if f_tgt > force[i] else tFF
            force[i] += (dt_s / tau_f) * (f_tgt - force[i])
            force[i]  = clamp(force[i], 0, FORCE_MAX)

            # Muscle length (shortening under force)
            leng[i] += (dt_s / tL) * (L0 - leng[i])
            leng[i] -= SHORTEN_GAIN * force[i] * dt_s
            leng[i]  = clamp(leng[i], L_MIN, L_MAX)

        t_arr[k] = t
        y_arr[:, k] = y
        a_arr[:, k] = act
        f_arr[:, k] = force
        l_arr[:, k] = leng

    return dict(
        time_s=t_arr, y=y_arr,
        c_L=c_L, c_R=c_R,
        # Left extensor=0, Left flexor=1, Right extensor=2, Right flexor=3
        force_LE=f_arr[0], force_LF=f_arr[1],
        force_RE=f_arr[2], force_RF=f_arr[3],
        act_LE=a_arr[0],   act_LF=a_arr[1],
        act_RE=a_arr[2],   act_RF=a_arr[3],
        len_LE=l_arr[0],   len_LF=l_arr[1],
        len_RE=l_arr[2],   len_RF=l_arr[3],
    )


# ══════════════════════════════════════════════════════════════════════════════
#  SAVE / LOAD HDF5
# ══════════════════════════════════════════════════════════════════════════════
def save_h5(label: str, res: dict) -> Path:
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = OUTPUT_DIR / f"matsuoka_{label}_{ts}.h5"
    with h5py.File(path, "w") as h:
        h.attrs["label"] = label
        h.attrs["c_L"]   = float(res["c_L"])
        h.attrs["c_R"]   = float(res["c_R"])
        h.attrs["model"] = "Matsuoka"
        h.attrs["tau1"]  = TAU1;  h.attrs["tau2"] = TAU2
        h.attrs["beta"]  = BETA;  h.attrs["w_loc"] = W_LOC
        h.attrs["w_com"] = W_COM; h.attrs["c0"]    = C0
        for k, v in res.items():
            if isinstance(v, np.ndarray):
                h.create_dataset(k, data=v.astype(np.float32), compression="gzip")
    print(f"  Saved → {path}")
    return path


# ══════════════════════════════════════════════════════════════════════════════
#  STEP-RHYTHM ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
def analyse(time_s, force_e, min_height=0.5):
    """Return freq_hz, duty_cycle, peaks_t from extensor force trace."""
    dt    = float(np.diff(time_s[:10]).mean())
    # Analyse only the steady-state second half
    half  = len(force_e) // 2
    fe    = force_e[half:]
    ts    = time_s[half:]

    if len(fe) < 20:
        return dict(freq_hz=0.0, duty_cycle=0.0, step_ms=0.0,
                    n_peaks=0, peaks_t=np.array([]))

    # Low-pass (4 Hz)
    b, a = butter(2, 4.0 * dt * 2, btype="low")
    fe_s = filtfilt(b, a, fe)

    min_d = max(1, int(0.3 / dt))
    peaks, _ = find_peaks(fe_s, height=min_height, distance=min_d)

    if len(peaks) < 2:
        return dict(freq_hz=0.0, duty_cycle=0.0, step_ms=0.0,
                    n_peaks=len(peaks), peaks_t=np.array([]))

    ipi  = np.diff(ts[peaks]) * 1000.0   # inter-peak intervals [ms]
    freq = 1000.0 / ipi.mean() if ipi.mean() > 0 else 0.0
    duty = float(np.mean(fe_s > fe_s.max() * 0.5))
    return dict(freq_hz=float(freq), duty_cycle=duty,
                step_ms=float(ipi.mean()), n_peaks=int(len(peaks)),
                peaks_t=ts[peaks])


# ══════════════════════════════════════════════════════════════════════════════
#  COMPARISON PLOT
# ══════════════════════════════════════════════════════════════════════════════
def make_plot(results: dict, scales: dict, out_path: Path):
    DARK  = "#0e1117"; LIGHT = "#e0e0e0"; ACCENT = "#f9a825"
    C_H   = "#4bc8ff"  # healthy blue
    C_S   = "#ff4b4b"  # stroke  red
    C_B   = "#888888"  # baseline grey

    CMAP  = {"baseline": C_B, "healthy": C_H, "stroke": C_S}

    fig = plt.figure(figsize=(19, 16))
    fig.patch.set_facecolor(DARK)
    gs = gridspec.GridSpec(4, 3, figure=fig,
                           left=0.07, right=0.97, top=0.93, bottom=0.06,
                           hspace=0.55, wspace=0.38)

    axs = {
        "diag":  fig.add_subplot(gs[0, :]),      # pipeline / network diagram
        "L_fe":  fig.add_subplot(gs[1, :2]),     # Left extensor force
        "R_fe":  fig.add_subplot(gs[2, :2]),     # Right extensor force
        "osc":   fig.add_subplot(gs[1, 2]),      # CPG oscillator output (healthy)
        "freq":  fig.add_subplot(gs[2, 2]),      # step frequency bar
        "asym":  fig.add_subplot(gs[3, :2]),     # L/R asymmetry index over time
        "tbl":   fig.add_subplot(gs[3, 2]),      # metrics table
    }

    for ax in axs.values():
        ax.set_facecolor("#1a1e2e")
        for s in ax.spines.values(): s.set_edgecolor("#3a3f5c")
        ax.tick_params(colors=LIGHT, labelsize=8)
        ax.xaxis.label.set_color(LIGHT); ax.yaxis.label.set_color(LIGHT)

    # ── Panel 0: Pipeline diagram ─────────────────────────────────────────
    ax = axs["diag"]
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    boxes = [
        (0.04, "TVB\nPhase 1\n(cortex)", "#2e3b6e"),
        (0.20, "CST Drive\nL: {hl:.3f} Hz\nR: {hr:.3f} Hz".format(
            hl=scales['healthy_L'], hr=scales['healthy_R']), "#1a4a2e"),
        (0.36, "Amplitude\nscale\n(stroke)", "#6e2e2e"),
        (0.52, "Matsuoka\nCPG\n(4 neurons)", "#2e3b6e"),
        (0.68, "Muscle\nProxy\n(force)", "#1a4a2e"),
        (0.84, "Step\nRhythm\nMetrics", "#2a4a1e"),
    ]
    from matplotlib.patches import FancyBboxPatch
    for x0, label, col in boxes:
        rect = FancyBboxPatch((x0-0.07, 0.12), 0.13, 0.76,
                              boxstyle="round,pad=0.02", lw=1.0,
                              edgecolor="#5a6090", facecolor=col,
                              transform=ax.transAxes, clip_on=False)
        ax.add_patch(rect)
        ax.text(x0, 0.50, label, ha="center", va="center", fontsize=7.5,
                color=LIGHT, transform=ax.transAxes, fontweight="bold",
                linespacing=1.35)
    for x0, x1 in [(0.11,0.20),(0.27,0.36),(0.43,0.52),(0.59,0.68),(0.75,0.84)]:
        ax.annotate("", xy=(x1-0.06, 0.50), xytext=(x0+0.01, 0.50),
                    xycoords="axes fraction", textcoords="axes fraction",
                    arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1.5))

    # Stroke note
    stroke_note = (f"Stroke: left M1 silenced → CST_L = {scales['stroke_L']:.3f} Hz "
                   f"({(1-scales['scale_L'])*100:.1f}% deficit)\n"
                   f"→ c_L = {C0*scales['scale_L']:.3f}  vs  c_R = {C0:.3f}  "
                   f"→ asymmetric oscillation → hemiparetic gait")
    ax.text(0.50, 0.02, stroke_note, ha="center", va="bottom", fontsize=7.5,
            color=ACCENT, transform=ax.transAxes,
            bbox=dict(fc="#1a1e2e", ec=ACCENT, pad=3, lw=0.8))

    # ── Helper: downsample ────────────────────────────────────────────────
    def ds(arr, n=2000):
        s = max(1, len(arr) // n)
        return arr[::s]

    # ── Panel 1: Left extensor force ──────────────────────────────────────
    ax = axs["L_fe"]
    for cond, res in results.items():
        t = ds(res["time_s"]); fe = ds(res["force_LE"])
        ax.plot(t, fe, color=CMAP[cond], lw=1.3, alpha=0.85,
                label=cond.capitalize())
    ax.set_xlabel("Time [s]", fontsize=9)
    ax.set_ylabel("Force [a.u.]", fontsize=9)
    ax.set_title("LEFT Extensor Muscle Force  (Stance phase)", fontsize=10,
                 color=LIGHT, pad=5)
    ax.legend(fontsize=8, facecolor="#1a1e2e", edgecolor="#3a3f5c",
               labelcolor=LIGHT)
    ax.set_xlim(0, results["healthy"]["time_s"][-1])

    # Mark step peaks on healthy trace
    m = analyse(results["healthy"]["time_s"], results["healthy"]["force_LE"])
    for pk in m["peaks_t"]:
        ax.axvline(pk, color=C_H, lw=0.5, alpha=0.35)

    note = ("Baseline (grey) steps slower than Healthy (blue) — c→frequency\n"
            "dose-response via drive-scaled τ.  Stroke (red): same cadence\n"
            "as Healthy but REDUCED force amplitude (hemiparetic deficit).")
    ax.text(0.02, 0.05, note, transform=ax.transAxes, fontsize=7,
            color="#a0a8c0",
            bbox=dict(fc="#1a1e2e", ec="#3a3f5c", pad=2, lw=0.7))

    # ── Panel 2: Right extensor force ─────────────────────────────────────
    ax = axs["R_fe"]
    for cond, res in results.items():
        t = ds(res["time_s"]); fe = ds(res["force_RE"])
        ax.plot(t, fe, color=CMAP[cond], lw=1.3, alpha=0.85,
                label=cond.capitalize())
    ax.set_xlabel("Time [s]", fontsize=9)
    ax.set_ylabel("Force [a.u.]", fontsize=9)
    ax.set_title("RIGHT Extensor Muscle Force  (Stance phase)", fontsize=10,
                 color=LIGHT, pad=5)
    ax.legend(fontsize=8, facecolor="#1a1e2e", edgecolor="#3a3f5c",
               labelcolor=LIGHT)
    ax.set_xlim(0, results["healthy"]["time_s"][-1])
    ax.text(0.02, 0.05,
            "Right leg: cadence unaffected (commissural entrainment).\n"
            "Stroke right amplitude slightly higher — unmasked by\n"
            "weaker contralateral drive (mild compensatory effect).",
            transform=ax.transAxes, fontsize=7, color="#a0a8c0",
            bbox=dict(fc="#1a1e2e", ec="#3a3f5c", pad=2, lw=0.7))

    # ── Panel 3: CPG raw oscillator output (healthy) ──────────────────────
    ax = axs["osc"]
    res_h = results["healthy"]
    # Downsample consistently on the time axis (axis-1 for y)
    _stride = max(1, len(res_h["time_s"]) // 1000)
    t_osc   = res_h["time_s"][::_stride]
    y_osc   = res_h["y"][:, ::_stride]          # shape (4, ~1000)
    # Show only steady-state: last 10 s
    mask = t_osc > (t_osc[-1] - 10.0)
    t_w  = t_osc[mask]
    colors_osc = ["#4bc8ff", "#2255aa", "#ff4b4b", "#aa2222"]
    labels_osc = ["LE (L ext)", "LF (L flex)", "RE (R ext)", "RF (R flex)"]
    for i, (col, lbl) in enumerate(zip(colors_osc, labels_osc)):
        ax.plot(t_w, y_osc[i][mask], color=col, lw=1.2, label=lbl, alpha=0.9)
    ax.set_xlabel("Time [s]", fontsize=9)
    ax.set_ylabel("Matsuoka output y", fontsize=9)
    ax.set_title("CPG Oscillator (healthy,\nsteady state)", fontsize=9,
                 color=LIGHT, pad=5)
    ax.legend(fontsize=6.5, facecolor="#1a1e2e", edgecolor="#3a3f5c",
               labelcolor=LIGHT, ncol=1)
    ax.set_xlim(t_w[0], t_w[-1])

    # ── Panel 4: Step frequency bar chart ────────────────────────────────
    ax = axs["freq"]
    conds = list(results.keys())
    x     = np.arange(len(conds))
    w     = 0.35
    for di, (side_key, side_lbl) in enumerate([("force_LE","Left"),("force_RE","Right")]):
        vals = []
        for cond in conds:
            m = analyse(results[cond]["time_s"], results[cond][side_key])
            vals.append(m["freq_hz"])
        bars = ax.bar(x + (di-0.5)*w, vals, w,
                      color=[CMAP[c] for c in conds],
                      alpha=(1.0 if side_lbl=="Left" else 0.6),
                      edgecolor="#3a3f5c", lw=0.7,
                      label=f"{side_lbl} leg")
        for b, v in zip(bars, vals):
            if v > 0:
                ax.text(b.get_x()+b.get_width()/2, v+0.01,
                        f"{v:.2f}", ha="center", va="bottom",
                        fontsize=6.5, color=LIGHT)
    ax.set_xticks(x)
    ax.set_xticklabels([c.capitalize() for c in conds], fontsize=8)
    ax.set_ylabel("Step Frequency [Hz]", fontsize=9)
    ax.set_title("Step Frequency\n(Left vs Right)", fontsize=9,
                 color=LIGHT, pad=5)
    ax.legend(fontsize=7, facecolor="#1a1e2e", edgecolor="#3a3f5c",
               labelcolor=LIGHT)

    # ── Panel 5: L/R force asymmetry index over time ──────────────────────
    ax = axs["asym"]
    for cond, res in results.items():
        if cond == "baseline": continue
        t   = res["time_s"]
        lfe = res["force_LE"]; rfe = res["force_RE"]
        ai  = (rfe - lfe) / np.maximum(rfe + lfe, 1e-3)
        sm  = uniform_filter1d(ai, size=max(1, int(0.5 / (t[1]-t[0]))))
        t_d = ds(t, 2000); sm_d = ds(sm, 2000)
        ax.plot(t_d, sm_d, color=CMAP[cond], lw=1.4,
                label=f"{cond.capitalize()} "
                      f"(mean={sm[len(sm)//2:].mean():.3f})")
    ax.axhline(0, color=LIGHT, lw=0.6, ls="--", alpha=0.4)
    ax.fill_between(ax.get_xlim(), [0, 0], [1, 1],
                    alpha=0.04, color=C_S, transform=ax.get_xaxis_transform())
    ax.set_xlabel("Time [s]", fontsize=9)
    ax.set_ylabel("Asymmetry Index  (R−L)/(R+L)", fontsize=9)
    ax.set_title("L/R Extensor Force Asymmetry  (+ve = Right dominant)",
                 fontsize=10, color=LIGHT, pad=5)
    ax.legend(fontsize=8, facecolor="#1a1e2e", edgecolor="#3a3f5c",
               labelcolor=LIGHT)
    ax.set_ylim(-1, 1)
    ax.set_xlim(0, results["healthy"]["time_s"][-1])

    # ── Panel 6: Summary metrics table ────────────────────────────────────
    ax = axs["tbl"]
    ax.axis("off")
    ax.set_title("Summary Metrics", fontsize=10, color=LIGHT, pad=6)

    rows = [("Metric", "Healthy", "Stroke")]
    rows.append(("─"*18, "─"*8, "─"*8))

    for side_key, side_lbl in [("LE","Left"), ("RE","Right")]:
        fk = f"force_{side_key}"
        for cond in ("healthy", "stroke"):
            m = analyse(results[cond]["time_s"], results[cond][fk])
            rows.append((f"Freq {side_lbl} [Hz]" if cond=="healthy" else "",
                         f"{m['freq_hz']:.3f}" if cond=="healthy" else "",
                         f"{m['freq_hz']:.3f}" if cond=="stroke"  else ""))
        for cond in ("healthy", "stroke"):
            m = analyse(results[cond]["time_s"], results[cond][fk])
            rows.append((f"Period {side_lbl} [ms]" if cond=="healthy" else "",
                         f"{m['step_ms']:.0f}" if cond=="healthy" else "",
                         f"{m['step_ms']:.0f}" if cond=="stroke"  else ""))

    # Compute Freq & Period together for table rows
    rows = [("Metric", "Healthy", "Stroke"),
            ("─"*18, "─"*8, "─"*8)]
    for side_key, side_lbl in [("LE","Left"), ("RE","Right")]:
        fk  = f"force_{side_key}"
        mH  = analyse(results["healthy"]["time_s"], results["healthy"][fk])
        mS  = analyse(results["stroke"]["time_s"],  results["stroke"][fk])
        rows.append((f"Freq {side_lbl} [Hz]",
                     f"{mH['freq_hz']:.3f}", f"{mS['freq_hz']:.3f}"))
        rows.append((f"Period {side_lbl} [ms]",
                     f"{mH['step_ms']:.0f}" if mH['step_ms']>0 else "—",
                     f"{mS['step_ms']:.0f}" if mS['step_ms']>0 else "—"))
        rows.append((f"Duty {side_lbl}",
                     f"{mH['duty_cycle']*100:.0f}%",
                     f"{mS['duty_cycle']*100:.0f}%"))
        rows.append(("", "", ""))

    # Amplitude & duty asymmetry (primary hemiparetic metrics)
    rows.append(("─"*18, "─"*8, "─"*8))
    def _peak_amp(t, force):
        """Mean peak force amplitude in steady-state half."""
        dt = float(np.diff(t[:10]).mean())
        from scipy.signal import find_peaks, butter, filtfilt
        half = len(force)//2
        b, a = butter(2, 4.0*dt*2, btype="low")
        fs = filtfilt(b, a, force[half:])
        pk, _ = find_peaks(fs, height=0.5, distance=max(1,int(0.3/dt)))
        return float(fs[pk].mean()) if len(pk) else 0.0

    for cond in ("healthy", "stroke"):
        res = results[cond]
        ampL = _peak_amp(res["time_s"], res["force_LE"])
        ampR = _peak_amp(res["time_s"], res["force_RE"])
        amp_ai = (ampR - ampL) / max(ampR, ampL) * 100 if max(ampR,ampL)>0 else 0.0
        rows.append((f"PeakF L [N] ({cond[:5]})",
                     f"{ampL:.1f}" if cond=="healthy" else "",
                     f"{ampL:.1f}" if cond=="stroke"  else ""))
        rows.append((f"PeakF R [N] ({cond[:5]})",
                     f"{ampR:.1f}" if cond=="healthy" else "",
                     f"{ampR:.1f}" if cond=="stroke"  else ""))
        rows.append((f"Amp asym ({cond[:5]})",
                     f"{amp_ai:.1f}%" if cond=="healthy" else "",
                     f"{amp_ai:.1f}%" if cond=="stroke"  else ""))
        rows.append(("", "", ""))

    rows.append(("─"*18, "─"*8, "─"*8))
    rows.append(("CST drive L [Hz]",
                 f"{scales['healthy_L']:.3f}", f"{scales['stroke_L']:.3f}"))
    rows.append(("CPG input c_L",
                 f"{C0:.3f}", f"{C0*scales['scale_L']:.3f}"))

    y0, dy = 0.99, 0.060
    col_x  = [0.01, 0.60, 0.80]
    col_c  = [LIGHT, C_H, C_S]
    for i, row in enumerate(rows):
        for j, (cell, cx, cc) in enumerate(zip(row, col_x, col_c)):
            fw = "bold" if i < 2 else "normal"
            ax.text(cx, y0-i*dy, cell, transform=ax.transAxes,
                    fontsize=7.0, color=cc, fontweight=fw, va="top")

    fig.suptitle(
        "Phase 2: TVB Cortex → Matsuoka Spinal CPG\n"
        "Left-M1 Stroke → Reduced CST Drive → Asymmetric Locomotion",
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
    print("  TVB Phase 2 — Cortex + Matsuoka Spinal CPG")
    print("=" * 62)

    print("\n[1/5] Loading Phase 1 TVB results...")
    scales = load_phase1_scales()

    # Input drives per condition
    c_baseline = (C0 * 0.70, C0 * 0.70)   # lower symmetric drive → show freq–drive relation
    c_healthy  = (C0,        C0)            # full symmetric drive
    c_stroke   = (C0 * scales["scale_L"],  # left reduced by stroke
                  C0 * scales["scale_R"])   # right unchanged

    print(f"\n  Tonic inputs c₀ = {C0}")
    print(f"  Baseline : c_L={c_baseline[0]:.3f}  c_R={c_baseline[1]:.3f}")
    print(f"  Healthy  : c_L={c_healthy[0]:.3f}   c_R={c_healthy[1]:.3f}")
    print(f"  Stroke   : c_L={c_stroke[0]:.3f}  c_R={c_stroke[1]:.3f}  "
          f"(left ↓{(1-scales['scale_L'])*100:.1f}%)")

    SIM_S = 20.0; DT_S = 0.001

    print("\n[2/5] Simulating BASELINE condition...")
    res_base = simulate(*c_baseline, sim_s=SIM_S, dt_s=DT_S)
    f_base   = save_h5("baseline", res_base)

    print("\n[3/5] Simulating HEALTHY condition...")
    res_heal = simulate(*c_healthy,  sim_s=SIM_S, dt_s=DT_S)
    f_heal   = save_h5("healthy",  res_heal)

    print("\n[4/5] Simulating STROKE condition...")
    res_stro = simulate(*c_stroke,   sim_s=SIM_S, dt_s=DT_S)
    f_stro   = save_h5("stroke",   res_stro)

    results  = {"baseline": res_base, "healthy": res_heal, "stroke": res_stro}

    print("\n[5/5] Analysis and plot...")
    print(f"\n  {'Cond':10s} {'Side':6s} {'Freq Hz':>8s} {'Period ms':>10s} "
          f"{'Duty':>7s} {'Peaks':>6s}")
    print(f"  {'─'*50}")
    for cond, res in results.items():
        for sk, sl in [("LE","Left"),("RE","Right")]:
            m = analyse(res["time_s"], res[f"force_{sk}"])
            print(f"  {cond:10s} {sl:6s} {m['freq_hz']:8.3f} "
                  f"{m['step_ms']:10.0f} {m['duty_cycle']*100:6.0f}% "
                  f"{m['n_peaks']:6d}")

    # L/R frequency asymmetry
    print(f"\n  L/R frequency asymmetry (steady state):")
    for cond, res in results.items():
        mL = analyse(res["time_s"], res["force_LE"])
        mR = analyse(res["time_s"], res["force_RE"])
        if mL["freq_hz"] > 0 and mR["freq_hz"] > 0:
            ai = abs(mR["freq_hz"]-mL["freq_hz"])/max(mR["freq_hz"],mL["freq_hz"])*100
        else:
            ai = 0.0
        print(f"    {cond:10s}: {ai:.1f}%  "
              f"(L={mL['freq_hz']:.3f}  R={mR['freq_hz']:.3f} Hz)")

    fig_path = OUTPUT_DIR / "phase2_comparison.png"
    make_plot(results, scales, fig_path)

    print(f"\n{'='*62}")
    print("  Phase 2 complete.")
    for cond, fp in [("baseline",f_base),("healthy",f_heal),("stroke",f_stro)]:
        print(f"  {cond:10s} HDF5 : {fp}")
    print(f"  Figure       : {fig_path}")
    print(f"{'='*62}\n")


if __name__ == "__main__":
    main()
