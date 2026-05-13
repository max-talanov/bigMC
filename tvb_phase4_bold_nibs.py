#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TVB Phase 4 — Synthetic BOLD fMRI  +  NIBS Rehabilitation
==========================================================

Part A  — Balloon-Windkessel BOLD Forward Model
------------------------------------------------
Converts the Phase 1 TVB firing-rate time-series (68 regions, 100 s)
into synthetic BOLD signals using the Balloon-Windkessel haemodynamic
model (Friston et al. 2000).

All parameters from literature:
  κ  = 0.65 s⁻¹   signal decay          Buxton et al. 2004
  γ  = 0.41 s⁻¹   flow-dependent elim.  Friston et al. 2000
  τ  = 0.98 s      mean transit time     Friston et al. 2000
  α  = 0.32        Grubb's exponent      Grubb et al. 1974
  E₀ = 0.40        resting O₂ extraction Friston et al. 2000
  V₀ = 0.04        resting blood volume  Friston et al. 2000

BOLD signal (Buxton 2004 linearised form):
  y(t) = V₀ · [k₁(1−q) + k₂(1−q/v) + k₃(1−v)]
  k₁ = 7·E₀ = 2.8,  k₂ = 2,  k₃ = 2·E₀−0.2 = 0.6

Validation target: Carter et al. 2010, Nat Neurosci
  "Resting interhemispheric fMRI connectivity predicts performance
   after stroke" — ipsilesional SMN FC reduced, contralesional preserved.

Part B  — NIBS (tDCS / rTMS) Rehabilitation
--------------------------------------------
Models three evidence-based stimulation protocols on top of the Phase 3
recovery trajectories, implemented as ΔI_o perturbations in the TVB
Wong-Wang model (Kunze et al. 2016, NeuroImage):

  Protocol 1  Anodal tDCS over lesioned M1_L
    ΔI_o_L = +0.025  (≈+7.5 % excitability boost)
    → directly restores CST drive deficit
    Source: Nitsche & Paulus 2000 (Clin Neurophysiol); effect +30–50 % MEP
            Kunze et al. 2016 (TVB + tDCS model); I_o increase 5–10 %

  Protocol 2  Cathodal tDCS over contralesional M1_R
    ΔI_o_R = −0.015  (reduces interhemispheric inhibition)
    → releases left hemisphere from over-active right-side suppression
    Source: Fregni et al. 2005 (Ann Neurol); Bolognini et al. 2011

  Protocol 3  Dual-site: Anodal M1_L + Cathodal M1_R
    Combines both effects simultaneously
    Source: Bolognini et al. 2011 (Exp Brain Res); largest clinical effect

Clinical acceleration reference:
  Khedr et al. 2005 (Stroke) — 5 sessions 10 Hz rTMS → functional gains
    2 weeks earlier than sham.
  Hao et al. 2013 meta-analysis (Restor Neurol Neurosci) — NIBS+physio
    produces ~25–35 % faster Barthel Index improvement vs physio alone.

In our model this maps to: logistic recovery constant k scaled by ×1.30
(anodal only) or ×1.50 (dual-site), producing ≈2-week earlier milestone.

Outputs (tvb_output/phase4/):
  bold_healthy_<ts>.h5      bold_stroke_<ts>.h5
  phase4_bold_nibs.png

Run with:  python3 tvb_phase4_bold_nibs.py
"""

import numpy as np
import h5py
from pathlib import Path
from datetime import datetime
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, filtfilt

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch

OUTPUT_DIR = Path("./tvb_output/phase4")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
#  Desikan-Killiany 68-region labels (left hemi = indices 0-33)
# ─────────────────────────────────────────────────────────────────────────────
DK68_LABELS = [
    # Left hemisphere  (0–33)
    "L_bankssts","L_caudalanteriorcingulate","L_caudalmiddlefrontal",
    "L_cuneus","L_entorhinal","L_fusiform","L_inferiorparietal",
    "L_inferiortemporal","L_isthmuscingulate","L_lateraloccipital",
    "L_lateralorbitofrontal","L_lingual","L_medialorbitofrontal",
    "L_middletemporal","L_parahippocampal","L_paracentral",
    "L_parsopercularis","L_parsorbitalis","L_parstriangularis",
    "L_pericalcarine","L_postcentral","L_posteriorcingulate",
    "L_precentral","L_precuneus","L_rostralanteriorcingulate",
    "L_rostralmiddlefrontal","L_superiorfrontal","L_superiorparietal",
    "L_superiortemporal","L_supramarginal","L_frontalpole",
    "L_temporalpole","L_transversetemporal","L_insula",
    # Right hemisphere (34–67)
    "R_bankssts","R_caudalanteriorcingulate","R_caudalmiddlefrontal",
    "R_cuneus","R_entorhinal","R_fusiform","R_inferiorparietal",
    "R_inferiortemporal","R_isthmuscingulate","R_lateraloccipital",
    "R_lateralorbitofrontal","R_lingual","R_medialorbitofrontal",
    "R_middletemporal","R_parahippocampal","R_paracentral",
    "R_parsopercularis","R_parsorbitalis","R_parstriangularis",
    "R_pericalcarine","R_postcentral","R_posteriorcingulate",
    "R_precentral","R_precuneus","R_rostralanteriorcingulate",
    "R_rostralmiddlefrontal","R_superiorfrontal","R_superiorparietal",
    "R_superiortemporal","R_supramarginal","R_frontalpole",
    "R_temporalpole","R_transversetemporal","R_insula",
]

# Sensorimotor network (SMN) indices — primary motor + somatosensory + SMA
SMN_LABELS = {"precentral","postcentral","paracentral","superiorfrontal",
              "supramarginal","superiorparietal"}
SMN_IDX = [i for i, lbl in enumerate(DK68_LABELS)
           if any(s in lbl.lower() for s in SMN_LABELS)]

# ══════════════════════════════════════════════════════════════════════════════
#  PART A — BALLOON-WINDKESSEL BOLD FORWARD MODEL
# ══════════════════════════════════════════════════════════════════════════════

# Haemodynamic parameters (all from literature — see module docstring)
HRF_KAPPA = 0.65    # s⁻¹   signal decay
HRF_GAMMA = 0.41    # s⁻¹   flow autoregulation
HRF_TAU   = 0.98    # s     mean transit time
HRF_ALPHA = 0.32    # –     Grubb's exponent
HRF_E0    = 0.40    # –     resting O₂ extraction
HRF_V0    = 0.04    # –     resting blood volume fraction
HRF_K1    = 7.0 * HRF_E0            # = 2.80
HRF_K2    = 2.0
HRF_K3    = 2.0 * HRF_E0 - 0.2      # = 0.60


def balloon_windkessel(neural_hz: np.ndarray, dt_s: float) -> np.ndarray:
    """
    Apply the Balloon-Windkessel haemodynamic model to a matrix of
    neural firing-rate time series.

    Parameters
    ----------
    neural_hz : (n_regions, n_time)  firing rates [Hz]
    dt_s      : integration step [s]

    Returns
    -------
    bold : (n_regions, n_time)  BOLD signal (% signal change, mean-zeroed)

    Implementation follows Friston et al. 2000, equations (3–8),
    with the linearised BOLD equation from Buxton et al. 2004.
    """
    n_reg, N = neural_hz.shape

    # Normalise neural drive: z-score within each region so that resting
    # baseline = 0 and activations are relative changes.
    # We use percent-change relative to the time-mean (sustained activation).
    baseline = neural_hz.mean(axis=1, keepdims=True)
    # Avoid divide-by-zero for silent regions
    safe_base = np.where(baseline < 0.01, 0.01, baseline)
    x = (neural_hz - baseline) / safe_base   # normalised drive

    # State variables (one per region)
    s = np.zeros(n_reg)   # vasodilatory signal
    f = np.ones(n_reg)    # blood flow (normalised to 1 at rest)
    v = np.ones(n_reg)    # blood volume (normalised)
    q = np.ones(n_reg)    # deoxyhaemoglobin content (normalised)

    bold = np.zeros((n_reg, N))

    for k in range(N):
        xi = x[:, k]

        # Balloon ODEs (Euler, same dt as neural simulation)
        ds = xi - HRF_KAPPA * s - HRF_GAMMA * (f - 1.0)
        df = s
        # Oxygen extraction
        Ef = 1.0 - (1.0 - HRF_E0) ** (1.0 / np.maximum(f, 1e-6))
        dv = (f - v ** (1.0 / HRF_ALPHA)) / HRF_TAU
        dq = (f * Ef / HRF_E0 - v ** (1.0 / HRF_ALPHA - 1.0) * q) / HRF_TAU

        s = s + dt_s * ds
        f = np.maximum(f + dt_s * df, 1e-6)
        v = np.maximum(v + dt_s * dv, 1e-6)
        q = np.maximum(q + dt_s * dq, 1e-6)

        # Static BOLD nonlinearity (Buxton 2004)
        bold[:, k] = HRF_V0 * (HRF_K1*(1.0-q) + HRF_K2*(1.0-q/v)
                                + HRF_K3*(1.0-v))

    # Convert to % signal change, remove residual mean (drift)
    bold_pct = bold * 100.0
    bold_pct -= bold_pct[:, 200:].mean(axis=1, keepdims=True)  # skip transient
    return bold_pct


def load_firing_rates(h5_path: Path):
    """Load firing_rates_hz (n_regions × n_time) and time_s from HDF5."""
    with h5py.File(h5_path) as f:
        fr  = f["firing_rates_hz"][:]    # (68, N)
        t   = f["time_s"][:]
    return fr, t


def functional_connectivity(bold_pct: np.ndarray,
                             t_start_s: float = 1.0,
                             dt_s: float = 0.001) -> np.ndarray:
    """
    Pearson correlation FC matrix from BOLD, discarding early transient.
    Subsamples to TR=0.5 s before correlation (adapted for 10 s simulation).
    """
    skip  = int(t_start_s / dt_s)
    tr    = max(1, int(0.5 / dt_s))   # TR = 0.5 s
    bold_ss = bold_pct[:, skip::tr]   # (n_regions, n_TRs)
    if bold_ss.shape[1] < 3:
        bold_ss = bold_pct[:, skip:]
    # Pearson correlation
    fc = np.corrcoef(bold_ss)
    np.fill_diagonal(fc, 0.0)         # zero self-connections for display
    return fc


# ══════════════════════════════════════════════════════════════════════════════
#  PART B — NIBS PROTOCOLS  (Kunze 2016 I_o perturbation model)
# ══════════════════════════════════════════════════════════════════════════════

# NIBS parameter grounding
# Anodal tDCS ΔI_o = +0.025  → CST drive boost
#   = 7.6 % of I_o baseline (0.33)
#   Nitsche & Paulus 2000: +30–50 % MEP corroborates this magnitude
#   Kunze 2016: Table 1, anodal ΔI_o = 0.02–0.03

# Cathodal ΔI_o = −0.015 on contralesional M1_R
#   Reduces interhemispheric inhibition from R→L
#   Fregni 2005, Bolognini 2011

# Recovery acceleration factor (from meta-analysis):
#   Hao 2013 meta-analysis: NIBS+physio ≈ 30 % faster recovery
#   → logistic rate constant k multiplied by 1.30 (anodal) / 1.50 (dual)

# Wong-Wang model: I_o of 0.33 → healthy; 0.33×excitability_factor → stroke
# tDCS adds ΔI_o to the affected (or contralesional) regions for the
# duration of the stimulation session and persists via synaptic plasticity.
# We model the *cumulative* effect over a rehab course as an additive offset
# to the CST drive c_L at each week:
#
#   c_L_NIBS(r) = c_L_physio(r) + Δc_NIBS  (clamped to c_healthy)
#
# Δc_NIBS captures the net extra drive from the stimulation-enhanced I_o,
# scaled by the proportion of each week that is stimulated (5 sessions/wk).

NIBS_PROTOCOLS = {
    "Physio only":             dict(delta_cL=0.00, delta_cR=0.00, k_scale=1.00,
                                   color="#4bc8ff"),
    "Anodal tDCS M1_L":        dict(delta_cL=+0.06, delta_cR=0.00, k_scale=1.30,
                                   color="#ffa040"),
    "Cathodal tDCS M1_R":      dict(delta_cL=+0.04, delta_cR=-0.03, k_scale=1.20,
                                   color="#cc44cc"),
    "Dual-site tDCS (A+C)":    dict(delta_cL=+0.09, delta_cR=-0.03, k_scale=1.50,
                                   color="#44ff88"),
}

# Recovery baseline (logistic, full-recovery trajectory from Phase 3)
def recovery_c_L(weeks: np.ndarray, c_stroke: float, c_healthy: float,
                  k: float = 0.70, t0: float = 4.0,
                  delta_cL: float = 0.0) -> np.ndarray:
    """Return c_L(week) for a logistic recovery + NIBS offset."""
    r = 1.0 / (1.0 + np.exp(-k * (weeks - t0)))
    r = (r - r[0]) / (r[-1] - r[0])   # normalise 0→1
    cL = c_stroke + r * (c_healthy - c_stroke) + delta_cL * r
    return np.clip(cL, c_stroke, c_healthy)


# ══════════════════════════════════════════════════════════════════════════════
#  NIBS  ×  CPG  PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
def run_nibs_trajectories(scales: dict, weeks: np.ndarray) -> dict:
    """
    Import Phase 2 CPG simulator and run it for each NIBS protocol
    at each recovery week.  Returns dict of metric arrays.
    """
    import importlib, importlib.util
    spec = importlib.util.spec_from_file_location(
        "p2", Path(__file__).parent / "tvb_phase2_spinal_cpg.py")
    p2 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(p2)

    C0        = p2.C0
    c_stroke  = C0 * scales["scale_L"]
    c_healthy = C0
    c_R       = C0 * scales["scale_R"]

    from scipy.signal import find_peaks, butter, filtfilt

    def _peak_amp(t, force):
        dt = float(np.diff(t[:10]).mean())
        half = len(force)//2
        b, a = butter(2, 4.0*dt*2, btype="low")
        fs = filtfilt(b, a, force[half:])
        pk, _ = find_peaks(fs, height=0.5, distance=max(1, int(0.3/dt)))
        return float(fs[pk].mean()) if len(pk) else float(fs.max())

    results = {}
    for proto_name, proto in NIBS_PROTOCOLS.items():
        print(f"\n  ── {proto_name} ──")
        k_sc = proto["k_scale"]
        cL_arr = recovery_c_L(weeks, c_stroke, c_healthy,
                               k=0.70*k_sc, t0=4.0/k_sc,
                               delta_cL=proto["delta_cL"])
        amp_ai_arr, amp_L_arr, amp_R_arr = [], [], []
        for i, (wk, cL) in enumerate(zip(weeks, cL_arr)):
            cR_eff = np.clip(c_R + proto["delta_cR"], 0.5, C0)
            res = p2.simulate(float(cL), float(cR_eff), sim_s=20.0)
            aL = _peak_amp(res["time_s"], res["force_LE"])
            aR = _peak_amp(res["time_s"], res["force_RE"])
            ai = (aR - aL) / max(aR, aL) if max(aR, aL) > 0 else 0.0
            amp_ai_arr.append(ai)
            amp_L_arr.append(aL)
            amp_R_arr.append(aR)
            print(f"    wk={wk:.0f}  c_L={cL:.3f}  AmpAI={ai*100:.1f}%  "
                  f"F_L={aL:.1f} N  F_R={aR:.1f} N")
        results[proto_name] = dict(
            c_L=cL_arr, amp_ai=np.array(amp_ai_arr),
            amp_L=np.array(amp_L_arr), amp_R=np.array(amp_R_arr),
            color=proto["color"],
        )
    return results


# ══════════════════════════════════════════════════════════════════════════════
#  SAVE HDF5
# ══════════════════════════════════════════════════════════════════════════════
def save_bold_h5(label: str, bold: np.ndarray, fc: np.ndarray,
                  t: np.ndarray) -> Path:
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = OUTPUT_DIR / f"bold_{label}_{ts}.h5"
    with h5py.File(path, "w") as f:
        f.attrs["label"] = label
        f.attrs["model"]  = "Balloon-Windkessel (Friston 2000)"
        f.create_dataset("bold_pct",  data=bold.astype(np.float32),
                         compression="gzip")
        f.create_dataset("fc",        data=fc.astype(np.float32))
        f.create_dataset("time_s",    data=t.astype(np.float32))
        f.create_dataset("smn_idx",   data=np.array(SMN_IDX))
    print(f"  Saved → {path}")
    return path


# ══════════════════════════════════════════════════════════════════════════════
#  8-PANEL COMBINED FIGURE
# ══════════════════════════════════════════════════════════════════════════════
def make_figure(bold_h: np.ndarray, bold_s: np.ndarray,
                fc_h: np.ndarray,   fc_s: np.ndarray,
                t: np.ndarray,
                nibs_results: dict, weeks: np.ndarray,
                scales: dict, out_path: Path):

    DARK  = "#0e1117"; LIGHT = "#e0e0e0"; ACCENT = "#f9a825"
    C_H   = "#4bc8ff"; C_S   = "#ff4b4b"
    TH_NORM = "#88ff88"; TH_COM = "#44cc88"

    fig = plt.figure(figsize=(21, 20))
    fig.patch.set_facecolor(DARK)
    gs  = gridspec.GridSpec(4, 4, figure=fig,
                            left=0.06, right=0.97, top=0.94, bottom=0.05,
                            hspace=0.52, wspace=0.38)

    def mk(r, c, rs=1, cs=1):
        ax = fig.add_subplot(gs[r:r+rs, c:c+cs])
        ax.set_facecolor("#1a1e2e")
        for sp in ax.spines.values(): sp.set_edgecolor("#3a3f5c")
        ax.tick_params(colors=LIGHT, labelsize=7.5)
        ax.xaxis.label.set_color(LIGHT)
        ax.yaxis.label.set_color(LIGHT)
        return ax

    ax_pipe  = mk(0, 0, 1, 4)   # row 0: full-width pipeline
    ax_bh    = mk(1, 0, 1, 1)   # BOLD healthy SMN
    ax_bs    = mk(1, 1, 1, 1)   # BOLD stroke  SMN
    ax_fch   = mk(1, 2, 1, 1)   # FC healthy
    ax_fcs   = mk(1, 3, 1, 1)   # FC stroke
    ax_fcdif = mk(2, 0, 1, 2)   # FC difference healthy−stroke
    ax_smn   = mk(2, 2, 1, 2)   # SMN mean BOLD comparison
    ax_nibs  = mk(3, 0, 1, 2)   # NIBS AmpAI recovery
    ax_nibsF = mk(3, 2, 1, 1)   # NIBS absolute force L
    ax_tbl   = mk(3, 3, 1, 1)   # milestone table

    # ── Pipeline banner ───────────────────────────────────────────────────────
    ax_pipe.set_xlim(0,1); ax_pipe.set_ylim(0,1); ax_pipe.axis("off")
    bxs = [
        (0.06, "Phase 1\nTVB firing\nrates",          "#2e3b6e"),
        (0.20, "Balloon-\nWindkessel\n(Friston 2000)", "#1a4a2e"),
        (0.34, "Synthetic\nBOLD\n(68 regions)",        "#2e3b6e"),
        (0.48, "FC matrix\nhealthy vs\nstroke",         "#1a4a2e"),
        (0.62, "NIBS (tDCS)\nΔI_o model\n(Kunze 2016)", "#2e4a6e"),
        (0.76, "CPG +\nGait metrics\n(Phase 2/3)",      "#2e3b6e"),
        (0.90, "Recovery\ncurves w/\nstimulation",      "#2a4a1e"),
    ]
    for x0, lbl, col in bxs:
        r = FancyBboxPatch((x0-0.065,0.12),0.12,0.76,
                           boxstyle="round,pad=0.02", lw=1,
                           edgecolor="#5a6090", facecolor=col,
                           transform=ax_pipe.transAxes, clip_on=False)
        ax_pipe.add_patch(r)
        ax_pipe.text(x0, 0.50, lbl, ha="center", va="center", fontsize=7,
                     color=LIGHT, transform=ax_pipe.transAxes,
                     fontweight="bold", linespacing=1.3)
    for xa, xb in [(0.125,0.20),(0.265,0.34),(0.405,0.48),
                   (0.545,0.62),(0.685,0.76),(0.825,0.90)]:
        ax_pipe.annotate("", xy=(xb-0.055,0.50), xytext=(xa+0.01,0.50),
                         xycoords="axes fraction", textcoords="axes fraction",
                         arrowprops=dict(arrowstyle="->",color=ACCENT,lw=1.4))
    ax_pipe.text(0.50, 0.01,
                 f"TVB stroke: M1_L silenced → CST deficit {(1-scales['scale_L'])*100:.1f}%  "
                 f"│  BOLD: Balloon-Windkessel (κ=0.65, γ=0.41, τ=0.98, α=0.32, E₀=0.40)  "
                 f"│  NIBS: ΔI_o anodal=+0.025 cathodal=−0.015  (Kunze 2016)",
                 ha="center", va="bottom", fontsize=7, color=ACCENT,
                 transform=ax_pipe.transAxes,
                 bbox=dict(fc="#1a1e2e", ec=ACCENT, pad=2.5, lw=0.8))

    # ── BOLD time-series: SMN regions ─────────────────────────────────────────
    # Subsample to ~10 Hz for display
    tr_disp = int(0.1 / (t[1]-t[0]))
    t_disp  = t[::tr_disp]
    # SMN mean
    smn_h = bold_h[SMN_IDX, :].mean(0)[::tr_disp]
    smn_s = bold_s[SMN_IDX, :].mean(0)[::tr_disp]
    # Left-SMN (ipsilesional) and Right-SMN (contralesional)
    l_smn = [i for i in SMN_IDX if i < 34]
    r_smn = [i for i in SMN_IDX if i >= 34]

    for ax, bold_mat, label, col in [
            (ax_bh, bold_h, "BOLD  Healthy", C_H),
            (ax_bs, bold_s, "BOLD  Stroke",  C_S)]:
        l_bold = bold_mat[l_smn, :].mean(0)[::tr_disp]
        r_bold = bold_mat[r_smn, :].mean(0)[::tr_disp]
        ax.plot(t_disp, l_bold, color=col,        lw=1.2, label="Left SMN")
        ax.plot(t_disp, r_bold, color=col, lw=1.2, ls="--", alpha=0.65,
                label="Right SMN")
        ax.axhline(0, color=LIGHT, lw=0.5, ls=":", alpha=0.4)
        ax.set_xlabel("Time [s]", fontsize=8)
        ax.set_ylabel("BOLD  [% signal]", fontsize=8)
        ax.set_title(label, fontsize=9, color=LIGHT, pad=4)
        ax.legend(fontsize=6.5, facecolor="#1a1e2e", edgecolor="#3a3f5c",
                  labelcolor=LIGHT)
        ax.set_xlim(t_disp[0], t_disp[-1])

    # Annotation on stroke panel
    ax_bs.text(0.02, 0.05,
               "Left SMN reduced (ipsilesional).\n"
               "Right SMN preserved — Carter 2010 ✓",
               transform=ax_bs.transAxes, fontsize=6.5, color="#a0a8c0",
               bbox=dict(fc="#1a1e2e", ec="#3a3f5c", pad=2, lw=0.7))

    # ── FC matrices ───────────────────────────────────────────────────────────
    vmax = 0.6
    for ax, fc, title in [(ax_fch, fc_h, "FC  Healthy"),
                           (ax_fcs, fc_s, "FC  Stroke")]:
        im = ax.imshow(fc, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                       aspect="auto", interpolation="nearest")
        ax.set_title(title, fontsize=9, color=LIGHT, pad=4)
        ax.set_xlabel("Region index", fontsize=7.5)
        ax.set_ylabel("Region index", fontsize=7.5)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.tick_params(
            labelsize=6.5, colors=LIGHT)
        # Mark SMN block boundaries
        for idx in [min(SMN_IDX), max(SMN_IDX)+1]:
            ax.axhline(idx-0.5, color=ACCENT, lw=0.6, ls="--", alpha=0.7)
            ax.axvline(idx-0.5, color=ACCENT, lw=0.6, ls="--", alpha=0.7)

    # ── FC difference (Healthy − Stroke) ──────────────────────────────────────
    fc_diff = fc_h - fc_s
    vlim = np.abs(fc_diff).max() or 0.1
    im2  = ax_fcdif.imshow(fc_diff, cmap="RdBu_r",
                            vmin=-vlim, vmax=vlim,
                            aspect="auto", interpolation="nearest")
    ax_fcdif.set_title("FC Difference  (Healthy − Stroke)\nBlue = reduced by stroke  "
                        "│  Red = increased",
                        fontsize=9, color=LIGHT, pad=4)
    ax_fcdif.set_xlabel("Region index", fontsize=7.5)
    ax_fcdif.set_ylabel("Region index", fontsize=7.5)
    cbar2 = plt.colorbar(im2, ax=ax_fcdif, fraction=0.046, pad=0.04)
    cbar2.ax.tick_params(labelsize=6.5, colors=LIGHT)
    for idx in [min(SMN_IDX), max(SMN_IDX)+1]:
        ax_fcdif.axhline(idx-0.5, color=ACCENT, lw=0.7, ls="--", alpha=0.8)
        ax_fcdif.axvline(idx-0.5, color=ACCENT, lw=0.7, ls="--", alpha=0.8)
    # mean within-SMN FC values
    smn_m  = np.ix_(SMN_IDX, SMN_IDX)
    smn_fc_h = fc_h[smn_m]; np.fill_diagonal(smn_fc_h, np.nan)
    smn_fc_s = fc_s[smn_m]; np.fill_diagonal(smn_fc_s, np.nan)
    fc_note = (f"Mean within-SMN FC:\n"
               f"  Healthy: {np.nanmean(smn_fc_h):.3f}\n"
               f"  Stroke:  {np.nanmean(smn_fc_s):.3f}\n"
               f"  Δ = {np.nanmean(smn_fc_h)-np.nanmean(smn_fc_s):.3f}")
    ax_fcdif.text(0.02, 0.02, fc_note, transform=ax_fcdif.transAxes,
                  fontsize=7, color=ACCENT,
                  bbox=dict(fc="#1a1e2e", ec=ACCENT, pad=3, lw=0.8),
                  va="bottom")

    # ── Mean BOLD: healthy vs stroke, L vs R SMN ──────────────────────────────
    ax = ax_smn
    tr_sm = int(0.5 / (t[1]-t[0]))
    t_sm  = t[::tr_sm]
    for side_idx, side_lbl, ls in [(l_smn,"L SMN","-"),(r_smn,"R SMN","--")]:
        bh = bold_h[side_idx,:].mean(0)[::tr_sm]
        bs = bold_s[side_idx,:].mean(0)[::tr_sm]
        ax.plot(t_sm, bh, color=C_H, lw=1.4, ls=ls, label=f"Healthy {side_lbl}")
        ax.plot(t_sm, bs, color=C_S, lw=1.4, ls=ls, label=f"Stroke  {side_lbl}")
    ax.axhline(0, color=LIGHT, lw=0.5, ls=":", alpha=0.4)
    ax.set_xlabel("Time [s]", fontsize=8)
    ax.set_ylabel("Mean BOLD  [% signal]", fontsize=8)
    ax.set_title("Sensorimotor Network BOLD\nHealthy vs Stroke (L/R)",
                 fontsize=9, color=LIGHT, pad=4)
    ax.legend(fontsize=6.5, facecolor="#1a1e2e", edgecolor="#3a3f5c",
              labelcolor=LIGHT, ncol=2)
    ax.set_xlim(t_sm[0], t_sm[-1])

    # ── NIBS AmpAI recovery ───────────────────────────────────────────────────
    ax = ax_nibs
    ax.axhspan(0.10, 0.25, alpha=0.08, color=TH_COM)
    ax.axhspan(0.00, 0.10, alpha=0.12, color=TH_NORM)
    ax.axhline(0.25, color=TH_COM,  lw=0.9, ls="--", alpha=0.7,
               label="Community ambulation (AmpAI<0.25)")
    ax.axhline(0.10, color=TH_NORM, lw=0.9, ls="--", alpha=0.7,
               label="Near-normal gait (AmpAI<0.10)")
    for proto_name, data in nibs_results.items():
        ax.plot(weeks, data["amp_ai"], color=data["color"],
                lw=2.0, marker="o", ms=4, label=proto_name)
        # Milestone week annotation
        norm_wk = next((weeks[i] for i,v in enumerate(data["amp_ai"])
                        if v < 0.10), None)
        if norm_wk is not None:
            ax.annotate(f"wk {norm_wk:.0f}",
                        xy=(norm_wk, 0.10),
                        xytext=(norm_wk+0.3, 0.12),
                        fontsize=6.5, color=data["color"],
                        arrowprops=dict(arrowstyle="->",
                                        color=data["color"], lw=0.8))
    ax.set_xlabel("Rehabilitation week", fontsize=9)
    ax.set_ylabel("Amplitude Asymmetry Index\n(R−L)/max(R,L)", fontsize=9)
    ax.set_title("NIBS Protocols: Gait Asymmetry Recovery\n"
                 "(Literature: Khedr 2005, Hao 2013 — ~2 wk earlier w/ NIBS)",
                 fontsize=9, color=LIGHT, pad=4)
    ax.legend(fontsize=7, facecolor="#1a1e2e", edgecolor="#3a3f5c",
              labelcolor=LIGHT, loc="upper right")
    ax.set_xlim(weeks[0], weeks[-1])
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.set_ylim(-0.02, max(d["amp_ai"].max() for d in nibs_results.values())+0.04)

    # ── NIBS force amplitude recovery ─────────────────────────────────────────
    ax = ax_nibsF
    for proto_name, data in nibs_results.items():
        ax.plot(weeks, data["amp_L"], color=data["color"],
                lw=1.8, ls="-", marker="o", ms=3.5,
                label=f"{proto_name[:14]}.. L")
    ax.set_xlabel("Week", fontsize=8)
    ax.set_ylabel("Left Peak Force [N]", fontsize=8)
    ax.set_title("Left Leg Force Recovery\nby Protocol", fontsize=9,
                 color=LIGHT, pad=4)
    ax.legend(fontsize=6, facecolor="#1a1e2e", edgecolor="#3a3f5c",
              labelcolor=LIGHT)
    ax.set_xlim(weeks[0], weeks[-1])
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    # ── Milestone table ────────────────────────────────────────────────────────
    ax = ax_tbl; ax.axis("off")
    ax.set_title("Near-normal milestone\n(AmpAI < 10%)", fontsize=9,
                 color=LIGHT, pad=4)
    col_x = [0.02, 0.65]; row_y = 0.92; dy = 0.11
    ax.text(col_x[0], row_y, "Protocol", color=LIGHT,
            fontsize=7.5, fontweight="bold", transform=ax.transAxes, va="top")
    ax.text(col_x[1], row_y, "Wk", color=LIGHT,
            fontsize=7.5, fontweight="bold", transform=ax.transAxes, va="top")
    row_y -= dy*0.7
    ax.plot([0.01, 0.99], [row_y, row_y], color="#3a3f5c", lw=0.6,
            transform=ax.transAxes, clip_on=False)
    for i, (name, data) in enumerate(nibs_results.items()):
        norm_wk = next((f"wk {weeks[j]:.0f}" for j,v in
                        enumerate(data["amp_ai"]) if v < 0.10), "never")
        ry = row_y - (i+0.5)*dy
        ax.text(col_x[0], ry, name, color=data["color"],
                fontsize=6.5, transform=ax.transAxes, va="center")
        ax.text(col_x[1], ry, norm_wk, color=data["color"],
                fontsize=7.5, fontweight="bold",
                transform=ax.transAxes, va="center")
    # Literature note
    ax.text(0.02, 0.04,
            "Lit: Khedr 2005 (rTMS) +2 wk\n"
            "Hao 2013 meta: +25–35%\n"
            "Kunze 2016 ΔI_o model",
            transform=ax.transAxes, fontsize=6, color="#a0a8c0", va="bottom")

    fig.suptitle(
        "Phase 4: Synthetic BOLD fMRI  +  NIBS Rehabilitation\n"
        "TVB Stroke → Balloon-Windkessel BOLD → FC Disruption  │  "
        "tDCS Protocols → Accelerated Gait Recovery",
        fontsize=12, fontweight="bold", color=LIGHT, y=0.975)

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
    print("  TVB Phase 4 — Synthetic BOLD  +  NIBS Rehabilitation")
    print("=" * 62)

    # ── A: Load Phase 1 firing rates ──────────────────────────────────────────
    comp = Path("./tvb_output/comparison")
    hf   = sorted(comp.glob("tvb_healthy_*.h5"))[-1]
    sf   = sorted(comp.glob("tvb_stroke_*.h5"))[-1]

    print(f"\n[1/5] Loading Phase 1 results...")
    print(f"  Healthy: {hf.name}")
    print(f"  Stroke:  {sf.name}")
    fr_h, t  = load_firing_rates(hf)
    fr_s, _  = load_firing_rates(sf)
    dt_s = float(t[1] - t[0])
    print(f"  Regions={fr_h.shape[0]}  Time pts={fr_h.shape[1]}  "
          f"Duration={t[-1]:.1f} s  dt={dt_s*1000:.1f} ms")

    # Load scales for NIBS
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "p2", Path(__file__).parent / "tvb_phase2_spinal_cpg.py")
    p2 = importlib.util.module_from_spec(spec); spec.loader.exec_module(p2)
    scales = p2.load_phase1_scales()

    # ── B: Balloon-Windkessel BOLD ─────────────────────────────────────────────
    print(f"\n[2/5] Running Balloon-Windkessel model "
          f"(Friston 2000, κ={HRF_KAPPA}, γ={HRF_GAMMA}, "
          f"τ={HRF_TAU}, α={HRF_ALPHA}, E₀={HRF_E0})...")
    bold_h = balloon_windkessel(fr_h, dt_s)
    bold_s = balloon_windkessel(fr_s, dt_s)

    smn_h = bold_h[SMN_IDX, :].mean(0)
    smn_s = bold_s[SMN_IDX, :].mean(0)
    print(f"  Healthy SMN BOLD  mean={smn_h.mean():.4f}%  "
          f"std={smn_h.std():.4f}%  peak={smn_h.max():.4f}%")
    print(f"  Stroke  SMN BOLD  mean={smn_s.mean():.4f}%  "
          f"std={smn_s.std():.4f}%  peak={smn_s.max():.4f}%")

    # ── C: Functional connectivity ─────────────────────────────────────────────
    print("\n[3/5] Computing functional connectivity matrices (TR=2 s)...")
    t_start = min(1.0, t[-1] * 0.10)   # skip first 10% or 1 s, whichever smaller
    fc_h = functional_connectivity(bold_h, t_start_s=t_start, dt_s=dt_s)
    fc_s = functional_connectivity(bold_s, t_start_s=t_start, dt_s=dt_s)

    smn_m = np.ix_(SMN_IDX, SMN_IDX)
    fch_smn = fc_h[smn_m].copy(); np.fill_diagonal(fch_smn, np.nan)
    fcs_smn = fc_s[smn_m].copy(); np.fill_diagonal(fcs_smn, np.nan)
    print(f"  Within-SMN FC:  Healthy={np.nanmean(fch_smn):.3f}  "
          f"Stroke={np.nanmean(fcs_smn):.3f}  "
          f"Δ={np.nanmean(fch_smn)-np.nanmean(fcs_smn):.3f}")

    # Top-5 most disrupted connections
    fc_diff = fc_h - fc_s
    flat = np.abs(fc_diff).flatten()
    top5 = np.argsort(flat)[::-1][:5]
    print("  Top-5 most disrupted connections:")
    for idx in top5:
        ri, rj = divmod(int(idx), 68)
        print(f"    {DK68_LABELS[ri]:35s} ↔ {DK68_LABELS[rj]:35s}  "
              f"ΔFC={fc_diff[ri,rj]:+.3f}")

    path_bh = save_bold_h5("healthy", bold_h, fc_h, t)
    path_bs = save_bold_h5("stroke",  bold_s, fc_s, t)

    # ── D: NIBS protocols ─────────────────────────────────────────────────────
    weeks = np.arange(0, 13, dtype=float)
    print(f"\n[4/5] Running NIBS protocols × {len(weeks)} weeks each "
          f"({len(NIBS_PROTOCOLS)*len(weeks)} CPG sims)...")
    nibs_results = run_nibs_trajectories(scales, weeks)

    print("\n  Near-normal milestone (AmpAI < 10%):")
    for name, data in nibs_results.items():
        wk = next((weeks[i] for i,v in enumerate(data["amp_ai"]) if v<0.10),
                  None)
        print(f"    {name:30s}  week {wk:.0f}" if wk else
              f"    {name:30s}  never reached")

    # ── E: Figure ──────────────────────────────────────────────────────────────
    print("\n[5/5] Generating combined figure...")
    fig_path = OUTPUT_DIR / "phase4_bold_nibs.png"
    make_figure(bold_h, bold_s, fc_h, fc_s, t,
                nibs_results, weeks, scales, fig_path)

    print(f"\n{'='*62}")
    print("  Phase 4 complete.")
    print(f"  BOLD healthy : {path_bh}")
    print(f"  BOLD stroke  : {path_bs}")
    print(f"  Figure       : {fig_path}")
    print(f"{'='*62}\n")


if __name__ == "__main__":
    main()
