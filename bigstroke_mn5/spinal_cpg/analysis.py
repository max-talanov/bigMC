"""
Gait analysis for the spinal CPG output.

Extracts:
  - Per-population firing rates
  - Motoneuron pool firing-rate envelopes (proxy for muscle force)
  - Left-right alternation phase (target: ~180° anti-phase in healthy walk)
  - Amplitude asymmetry index (AmpAI) from MN output
"""
from __future__ import annotations
from pathlib import Path
import numpy as np

from . import params as P
from .recording import get_spike_data


# ════════════════════════════════════════════════════════════════════════════
#  Metrics
# ════════════════════════════════════════════════════════════════════════════

def compute_firing_rates(recorders, sizes, t_start_ms=None, t_end_ms=None):
    if t_start_ms is None: t_start_ms = P.WARMUP_MS
    if t_end_ms is None: t_end_ms = P.SIM_TIME_MS
    duration_s = (t_end_ms - t_start_ms) / 1000.0
    rates = {}
    for name, sr in recorders.items():
        d = get_spike_data(sr)
        mask = (d["times"] >= t_start_ms) & (d["times"] < t_end_ms)
        n_spikes = int(mask.sum())
        rates[name] = n_spikes / max(sizes[name], 1) / max(duration_s, 1e-6)
    return rates


def mn_rate_envelope(recorders, sizes, side: str, muscle: str,
                     t_start_ms=None, t_end_ms=None, bin_ms=20.0):
    """
    Build a continuous firing-rate envelope (Hz, population mean) for one
    motoneuron pool, by binning spikes into bin_ms windows.
    Returns (t_centers_ms, rate_hz_array).
    """
    if t_start_ms is None: t_start_ms = P.WARMUP_MS
    if t_end_ms is None: t_end_ms = P.SIM_TIME_MS

    full_name = P.pop_full_name(side, muscle)
    d = get_spike_data(recorders[full_name])
    n_neurons = sizes[full_name]

    edges = np.arange(t_start_ms, t_end_ms + bin_ms, bin_ms)
    counts, _ = np.histogram(d["times"], bins=edges)
    centers = (edges[:-1] + edges[1:]) / 2.0
    rate = counts / max(n_neurons, 1) / (bin_ms / 1000.0)
    return centers, rate


def compute_gait_metrics(recorders, sizes):
    """
    Headline gait metrics: per-side MN_F and MN_E rates, amplitude asymmetry.
    """
    t_L_F, env_L_F = mn_rate_envelope(recorders, sizes, "L", "MN_F")
    t_L_E, env_L_E = mn_rate_envelope(recorders, sizes, "L", "MN_E")
    t_R_F, env_R_F = mn_rate_envelope(recorders, sizes, "R", "MN_F")
    t_R_E, env_R_E = mn_rate_envelope(recorders, sizes, "R", "MN_E")

    mn_L_total = env_L_F + env_L_E
    mn_R_total = env_R_F + env_R_E

    amp_L = float(mn_L_total.mean())
    amp_R = float(mn_R_total.mean())
    AmpAI = (amp_R - amp_L) / max(amp_R, amp_L) if max(amp_R, amp_L) > 0 else 0.0

    return {
        "envelopes": {
            "L_F": (t_L_F, env_L_F),
            "L_E": (t_L_E, env_L_E),
            "R_F": (t_R_F, env_R_F),
            "R_E": (t_R_E, env_R_E),
        },
        "mn_L_mean_hz": amp_L,
        "mn_R_mean_hz": amp_R,
        "AmpAI": float(AmpAI),
    }


# ════════════════════════════════════════════════════════════════════════════
#  Plots
# ════════════════════════════════════════════════════════════════════════════

def plot_mn_envelopes(metrics, out_path: Path, title: str = "Spinal CPG MN output"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    DARK = "#0e1117"; LIGHT = "#e0e0e0"; ACCENT = "#f9a825"

    envs = metrics["envelopes"]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    fig.patch.set_facecolor(DARK)

    for ax in (ax1, ax2):
        ax.set_facecolor("#1a1e2e")
        for sp in ax.spines.values():
            sp.set_edgecolor("#3a3f5c")
        ax.tick_params(colors=LIGHT, labelsize=9)
        ax.xaxis.label.set_color(LIGHT)
        ax.yaxis.label.set_color(LIGHT)

    # Top: flexor MN rates
    t, env = envs["L_F"]
    ax1.plot(t, env, color="#ff6b6b", lw=2, label="L_MN_F (left flexor)", alpha=0.9)
    t, env = envs["R_F"]
    ax1.plot(t, env, color="#4ecdc4", lw=2, label="R_MN_F (right flexor)", alpha=0.9)
    ax1.set_ylabel("Flexor MN rate [Hz]", fontsize=10)
    ax1.set_title(title, color=ACCENT, fontsize=12, fontweight="bold", pad=8)
    ax1.legend(fontsize=9, facecolor="#1a1e2e", edgecolor="#3a3f5c",
              labelcolor=LIGHT, loc="upper right")
    ax1.grid(True, alpha=0.15, color=LIGHT)

    # Bottom: extensor MN rates
    t, env = envs["L_E"]
    ax2.plot(t, env, color="#ff6b6b", lw=2, ls="--", label="L_MN_E (left extensor)",
             alpha=0.9)
    t, env = envs["R_E"]
    ax2.plot(t, env, color="#4ecdc4", lw=2, ls="--", label="R_MN_E (right extensor)",
             alpha=0.9)
    ax2.set_xlabel("Time [ms]", fontsize=10)
    ax2.set_ylabel("Extensor MN rate [Hz]", fontsize=10)
    ax2.legend(fontsize=9, facecolor="#1a1e2e", edgecolor="#3a3f5c",
              labelcolor=LIGHT, loc="upper right")
    ax2.grid(True, alpha=0.15, color=LIGHT)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"[Plot] Saved MN envelopes → {out_path}")


def plot_raster(recorders, sizes, out_path: Path,
                t_window_ms=None, max_per_pop=120, title="Spinal CPG raster"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    DARK = "#0e1117"; LIGHT = "#e0e0e0"; ACCENT = "#f9a825"

    fig, ax = plt.subplots(figsize=(14, 10))
    fig.patch.set_facecolor(DARK)
    ax.set_facecolor("#1a1e2e")

    pal_L = ["#ff6b6b", "#ff8e72", "#ffa478", "#ffc093",
             "#e74c3c", "#c0392b", "#d35400", "#e67e22"]
    pal_R = ["#4ecdc4", "#5dade2", "#3498db", "#2980b9",
             "#1abc9c", "#16a085", "#9b59b6", "#8e44ad"]

    y_off = 0; yticks = []; yticklabels = []
    for side, palette in (("L", pal_L), ("R", pal_R)):
        for i, pop in enumerate(P.POP_NAMES):
            full_name = P.pop_full_name(side, pop)
            d = get_spike_data(recorders[full_name])
            if t_window_ms:
                mask = ((d["times"] >= t_window_ms[0]) &
                        (d["times"] <  t_window_ms[1]))
                t = d["times"][mask]; s = d["senders"][mask]
            else:
                t = d["times"]; s = d["senders"]

            uniq = np.unique(s)
            keep = uniq[:max_per_pop] if len(uniq) > max_per_pop else uniq
            mask2 = np.isin(s, keep)
            t_p = t[mask2]; s_p = s[mask2]
            if len(t_p):
                _, s_rel = np.unique(s_p, return_inverse=True)
                ax.plot(t_p, s_rel + y_off, '.',
                       color=palette[i], markersize=1.5, alpha=0.85,
                       rasterized=True)
            yticks.append(y_off + max_per_pop / 2)
            yticklabels.append(full_name)
            y_off += max_per_pop + 12

    for sp in ax.spines.values():
        sp.set_edgecolor("#3a3f5c")
    ax.tick_params(colors=LIGHT, labelsize=9)
    ax.set_yticks(yticks); ax.set_yticklabels(yticklabels)
    ax.set_xlabel("Time [ms]", color=LIGHT, fontsize=11)
    ax.set_title(title, color=ACCENT, fontsize=13, fontweight="bold", pad=10)
    if t_window_ms:
        ax.set_xlim(*t_window_ms)
    ax.grid(True, alpha=0.1, color=LIGHT)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"[Plot] Saved raster → {out_path}")


def save_summary_csv(rates: dict, metrics: dict, out_path: Path):
    import csv
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        for k in sorted(rates):
            w.writerow([f"rate_{k}", f"{rates[k]:.3f}"])
        w.writerow(["mn_L_mean_hz", f"{metrics['mn_L_mean_hz']:.3f}"])
        w.writerow(["mn_R_mean_hz", f"{metrics['mn_R_mean_hz']:.3f}"])
        w.writerow(["AmpAI", f"{metrics['AmpAI']:.3f}"])
    print(f"[CSV] Saved spinal summary → {out_path}")
