"""
Basic analysis: firing rates, ISI CV, raster plots.

Designed to be fast and lightweight, so it can run inside the simulation
job to produce QC plots without a separate post-processing step.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np

from . import params as P
from .recording import get_spike_data


# ════════════════════════════════════════════════════════════════════════════
#  Numerical metrics
# ════════════════════════════════════════════════════════════════════════════

def compute_firing_rates(recorders: dict, sizes: dict,
                          t_start_ms: float = None,
                          t_end_ms: float = None) -> dict:
    """
    Mean firing rate (Hz) per population, computed over [t_start_ms, t_end_ms].

    Returns {full_name: rate_hz}.
    """
    if t_start_ms is None:
        t_start_ms = P.WARMUP_MS
    if t_end_ms is None:
        t_end_ms = P.SIM_TIME_MS

    duration_s = (t_end_ms - t_start_ms) / 1000.0
    rates = {}
    for name, sr in recorders.items():
        data = get_spike_data(sr)
        in_window = (data["times"] >= t_start_ms) & (data["times"] < t_end_ms)
        n_spikes = int(in_window.sum())
        n_neurons = sizes[name]
        rate = n_spikes / max(n_neurons, 1) / max(duration_s, 1e-6)
        rates[name] = float(rate)
    return rates


def compute_isi_cv(recorders: dict, t_start_ms: float = None,
                    t_end_ms: float = None) -> dict:
    """
    Coefficient of variation of inter-spike intervals, per population.
    CV close to 1.0 indicates Poisson-like spiking.
    """
    if t_start_ms is None:
        t_start_ms = P.WARMUP_MS
    if t_end_ms is None:
        t_end_ms = P.SIM_TIME_MS

    cvs = {}
    for name, sr in recorders.items():
        data = get_spike_data(sr)
        mask = (data["times"] >= t_start_ms) & (data["times"] < t_end_ms)
        senders = data["senders"][mask]
        times = data["times"][mask]
        cv_list = []
        for nid in np.unique(senders):
            spk_t = np.sort(times[senders == nid])
            if len(spk_t) < 3:
                continue
            isi = np.diff(spk_t)
            cv_list.append(np.std(isi) / np.mean(isi))
        cvs[name] = float(np.mean(cv_list)) if cv_list else float("nan")
    return cvs


# ════════════════════════════════════════════════════════════════════════════
#  Plots
# ════════════════════════════════════════════════════════════════════════════

def plot_raster(recorders: dict, sizes: dict, out_path: Path,
                t_window_ms: tuple = None, max_per_pop: int = 200,
                title: str = "M1 microcircuit raster"):
    """
    Multi-population raster plot.  Each population gets a colored band.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    DARK = "#0e1117"
    LIGHT = "#e0e0e0"

    fig, ax = plt.subplots(figsize=(14, 10))
    fig.patch.set_facecolor(DARK)
    ax.set_facecolor("#1a1e2e")

    # Color palette: left = warm, right = cool
    colors_L = ["#ff6b6b", "#ff8e72", "#ffa478", "#ffc093",
                "#e74c3c", "#c0392b", "#f39c12", "#e67e22"]
    colors_R = ["#4ecdc4", "#5dade2", "#3498db", "#2980b9",
                "#1abc9c", "#16a085", "#9b59b6", "#8e44ad"]

    y_offset = 0
    yticks = []
    yticklabels = []

    for hemi, palette in (("L", colors_L), ("R", colors_R)):
        for i, pop in enumerate(P.POP_NAMES):
            full_name = P.pop_full_name(hemi, pop)
            data = get_spike_data(recorders[full_name])
            if t_window_ms:
                mask = ((data["times"] >= t_window_ms[0]) &
                        (data["times"] <  t_window_ms[1]))
                t = data["times"][mask]
                s = data["senders"][mask]
            else:
                t = data["times"]
                s = data["senders"]

            # subsample senders for plotting clarity
            unique_s = np.unique(s)
            keep_s = unique_s[:max_per_pop] if len(unique_s) > max_per_pop else unique_s
            keep_mask = np.isin(s, keep_s)
            t_plot = t[keep_mask]
            s_plot = s[keep_mask]

            if len(t_plot):
                # remap sender IDs to a continuous index for plotting
                _, s_rel = np.unique(s_plot, return_inverse=True)
                ax.plot(t_plot, s_rel + y_offset, '.',
                       color=palette[i], markersize=1.5, alpha=0.85,
                       rasterized=True)
            yticks.append(y_offset + max_per_pop / 2)
            yticklabels.append(full_name)
            y_offset += max_per_pop + 20

    for sp in ax.spines.values():
        sp.set_edgecolor("#3a3f5c")
    ax.tick_params(colors=LIGHT, labelsize=9)
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)
    ax.set_xlabel("Time [ms]", color=LIGHT, fontsize=11)
    ax.set_ylabel("Population", color=LIGHT, fontsize=11)
    ax.set_title(title, color="#f9a825", fontsize=13, fontweight="bold", pad=10)
    if t_window_ms:
        ax.set_xlim(*t_window_ms)
    ax.grid(True, alpha=0.1, color=LIGHT)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"[Plot] Saved raster → {out_path}")


def save_rates_csv(rates: dict, cvs: dict, sizes: dict, out_path: Path):
    """
    Write a CSV summary of per-population firing rates and ISI CVs.
    """
    import csv
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["population", "n_neurons", "rate_hz", "isi_cv", "target_hz", "match"])
        targets = {
            "L23e": 0.74, "L23i": 2.65, "L4e": 4.40, "L4i": 5.40,
            "L5e":  6.78, "L5i":  8.31, "L6e": 1.10, "L6i": 7.61,
        }
        for full_name in sorted(rates):
            pop = full_name.split("_", 1)[1]
            target = targets[pop]
            rate = rates[full_name]
            cv = cvs.get(full_name, float("nan"))
            n = sizes[full_name]
            match = "✓" if abs(rate - target) / max(target, 0.1) < 0.5 else "x"
            w.writerow([full_name, n, f"{rate:.3f}", f"{cv:.3f}",
                       f"{target:.3f}", match])
    print(f"[CSV]  Saved firing rates → {out_path}")
