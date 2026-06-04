"""
Progressive-degeneration simulation driver (Stage A: time-phased).

Runs the M1 microcircuit through a sequence of "epochs".  Each epoch:
  1. Simulate a short window of spiking (activity sample)
  2. Measure per-population firing rates in that window
  3. Advance the biological degeneration clock and update neuron fates

Produces two key outputs:
  • Survival curve   : surviving neuron count by vulnerability class vs hours
  • Rate evolution   : mean E vs I firing rate vs hours (shows disinhibition
                       as inhibitory neurons die before — or alongside — E)

Usage:
    python -m bigstroke_mn5.m1_microcircuit.simulate_degeneration \
        --scale 0.05 --lesion 0.4 --epochs 24 --label degen
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path
import numpy as np

from . import params as P
from . import network as N
from . import stimulus as S
from . import recording as R
from . import degeneration as D


def _epoch_rates(recorders, sizes, t0_ms, t1_ms):
    """Mean firing rate (Hz) per population within [t0_ms, t1_ms)."""
    from .recording import get_spike_data
    dur_s = (t1_ms - t0_ms) / 1000.0
    rates = {}
    for name, sr in recorders.items():
        d = get_spike_data(sr)
        m = (d["times"] >= t0_ms) & (d["times"] < t1_ms)
        rates[name] = int(m.sum()) / max(sizes[name], 1) / max(dur_s, 1e-6)
    return rates


def _ei_means(rates):
    """Collapse per-population rates into mean excitatory & inhibitory rate
    over the LEFT hemisphere (the lesioned side)."""
    e_vals, i_vals = [], []
    for name, r in rates.items():
        hemi, pop = name.split("_", 1)
        if hemi != "L":
            continue
        (e_vals if P.is_excitatory(pop) else i_vals).append(r)
    return (float(np.mean(e_vals)) if e_vals else 0.0,
            float(np.mean(i_vals)) if i_vals else 0.0)


def run(scale=None, lesion=None, epochs=None, hours_per_epoch=None,
        epoch_sim_ms=None, output_dir=None, label="degen",
        seed=42, n_threads=1):
    scale          = scale if scale is not None else 0.05
    lesion         = lesion if lesion is not None else P.DEGEN_LESION_FRACTION
    epochs         = epochs if epochs is not None else P.DEGEN_EPOCHS
    hours_per_epoch = hours_per_epoch if hours_per_epoch is not None else P.DEGEN_HOURS_PER_EPOCH
    epoch_sim_ms   = epoch_sim_ms if epoch_sim_ms is not None else P.DEGEN_EPOCH_SIM_MS

    if output_dir is None:
        output_dir = Path(__file__).parent.parent / "output"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print(f"  BigStroke-MN5 — Progressive Degeneration  ({label})")
    print("=" * 72)
    print(f"  scale            = {scale}")
    print(f"  lesion fraction  = {lesion}  (spatial infarct, all L populations)")
    print(f"  epochs           = {epochs} × {hours_per_epoch} h = "
          f"{epochs*hours_per_epoch:.0f} biological hours")
    print(f"  epoch sim window = {epoch_sim_ms} ms")
    print("=" * 72)

    try:
        import nest
        print(f"[NEST] version {nest.__version__}")
    except ImportError:
        print("[ERROR] NEST not installed. See bigstroke_mn5/README.md.")
        sys.exit(1)

    t0 = time.time()

    # ── Build healthy network ───────────────────────────────────────────
    print("\n[Build] Constructing healthy network ...")
    net = N.build_network(scale=scale, stroke_fraction=0.0,
                          seed=seed, n_threads=n_threads)
    pops = net["populations"]
    sizes = net["sizes"]
    print(f"[Build] {sum(sizes.values()):,} neurons.")

    print("[Stim ] Background Poisson ...")
    S.attach_background(pops)
    print("[Rec  ] Spike recorders ...")
    recs = R.attach_spike_recorders(pops)

    # ── Classify vulnerability + initialise lesion ──────────────────────
    print("\n[Degen] Classifying cell-type vulnerability ...")
    vuln = D.classify_vulnerability(pops, seed=seed)
    n_e = sum(1 for v in vuln.values() if v == "excitatory")
    n_pv = sum(1 for v in vuln.values() if v == "pv_inhibitory")
    n_buf = sum(1 for v in vuln.values() if v == "buffered_inhibitory")
    print(f"        excitatory={n_e}, pv_inhibitory={n_pv}, buffered_inhibitory={n_buf}")

    print(f"[Degen] Initialising spatial infarct (lesion={lesion}) ...")
    state = D.init_lesion(pops, vuln, lesion_fraction=lesion, seed=seed)
    n_core = len(state["core_ids"])
    n_peri = len(state["penumbra"])
    print(f"        Core (silenced now): {n_core}   Penumbra (at risk): {n_peri}")

    # ── Epoch loop (operator splitting) ─────────────────────────────────
    print(f"\n[Sim  ] Running {epochs} epochs ...")
    rate_hist = []
    sim_clock_ms = 0.0
    for ep in range(epochs):
        t_start = sim_clock_ms
        nest.Simulate(epoch_sim_ms)
        sim_clock_ms += epoch_sim_ms
        # discard first 20% of window as transient after fate change
        warm = t_start + 0.2 * epoch_sim_ms
        rates = _epoch_rates(recs, sizes, warm, sim_clock_ms)
        e_mean, i_mean = _ei_means(rates)

        info = D.step(state, hours_advance=hours_per_epoch)
        hour = state["clock_hours"]
        rate_hist.append({"hour": hour, "e_mean": e_mean, "i_mean": i_mean})

        if ep % max(1, epochs // 8) == 0 or ep == epochs - 1:
            print(f"    epoch {ep:>2d}  t={hour:>4.0f}h   "
                  f"E={e_mean:>5.2f} Hz  I={i_mean:>5.2f} Hz   "
                  f"+{info['newly_dead']} dead, +{info['newly_recovered']} recovered")

    state["rate_history"] = rate_hist
    t_sim = time.time() - t0

    # ── Report ──────────────────────────────────────────────────────────
    print(f"\n[Anal ] Final survival (by vulnerability class):")
    print(D.survival_summary(state))

    # ── Plots ───────────────────────────────────────────────────────────
    surv_path = output_dir / f"degen_survival_{label}.png"
    rate_path = output_dir / f"degen_rates_{label}.png"
    _plot_survival(state, surv_path, label)
    _plot_rate_evolution(state, rate_path, label)

    print(f"\n{'='*72}")
    print(f"  DONE in {t_sim:.1f}s")
    print(f"  Survival curve : {surv_path}")
    print(f"  Rate evolution : {rate_path}")
    print("=" * 72)
    return state


# ════════════════════════════════════════════════════════════════════════════
#  Plots
# ════════════════════════════════════════════════════════════════════════════

def _plot_survival(state, out_path, label):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    DARK = "#0e1117"; LIGHT = "#e0e0e0"; ACCENT = "#f9a825"
    colors = {"excitatory": "#ff6b6b",
              "pv_inhibitory": "#f39c12",
              "buffered_inhibitory": "#4ecdc4"}
    nice = {"excitatory": "Excitatory (pyramidal)",
            "pv_inhibitory": "PV⁺ fast-spiking IN",
            "buffered_inhibitory": "Buffered IN (CB⁺/CR⁺)"}

    hours = [h["hour"] for h in state["history"]]

    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor(DARK)
    ax.set_facecolor("#1a1e2e")

    for c in ("excitatory", "pv_inhibitory", "buffered_inhibitory"):
        alive = []
        for h in state["history"]:
            a = h["counts"][c]["alive"]
            d = h["counts"][c]["dead"]
            tot = a + d
            alive.append(100.0 * a / tot if tot else 100.0)
        ax.plot(hours, alive, lw=2.5, marker="o", ms=4,
                color=colors[c], label=nice[c], alpha=0.9)

    for sp in ax.spines.values():
        sp.set_edgecolor("#3a3f5c")
    ax.tick_params(colors=LIGHT, labelsize=9)
    ax.set_xlabel("Biological time since stroke onset [hours]",
                  color=LIGHT, fontsize=11)
    ax.set_ylabel("Surviving neurons in lesion [%]", color=LIGHT, fontsize=11)
    ax.set_title(f"Progressive cell death by vulnerability class — {label}\n"
                 f"(selective neuronal vulnerability: E & PV⁺ die, buffered IN survive)",
                 color=ACCENT, fontsize=12, fontweight="bold", pad=10)
    ax.legend(fontsize=10, facecolor="#1a1e2e", edgecolor="#3a3f5c",
              labelcolor=LIGHT, loc="lower left")
    ax.grid(True, alpha=0.15, color=LIGHT)
    ax.set_ylim(-3, 103)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"[Plot] Saved survival curve → {out_path}")


def _plot_rate_evolution(state, out_path, label):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    DARK = "#0e1117"; LIGHT = "#e0e0e0"; ACCENT = "#f9a825"
    rh = state["rate_history"]
    hours = [r["hour"] for r in rh]
    e = [r["e_mean"] for r in rh]
    i = [r["i_mean"] for r in rh]

    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor(DARK)
    ax.set_facecolor("#1a1e2e")

    ax.plot(hours, e, lw=2.5, marker="o", ms=4, color="#ff6b6b",
            label="Left-hemisphere excitatory mean", alpha=0.9)
    ax.plot(hours, i, lw=2.5, marker="s", ms=4, color="#4ecdc4",
            label="Left-hemisphere inhibitory mean", alpha=0.9)

    for sp in ax.spines.values():
        sp.set_edgecolor("#3a3f5c")
    ax.tick_params(colors=LIGHT, labelsize=9)
    ax.set_xlabel("Biological time since stroke onset [hours]",
                  color=LIGHT, fontsize=11)
    ax.set_ylabel("Mean firing rate [Hz]", color=LIGHT, fontsize=11)
    ax.set_title(f"Network rate evolution during degeneration — {label}\n"
                 f"(watch for transient disinhibition as inhibitory cells die)",
                 color=ACCENT, fontsize=12, fontweight="bold", pad=10)
    ax.legend(fontsize=10, facecolor="#1a1e2e", edgecolor="#3a3f5c",
              labelcolor=LIGHT, loc="upper right")
    ax.grid(True, alpha=0.15, color=LIGHT)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"[Plot] Saved rate evolution → {out_path}")


def _cli():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scale", type=float, default=0.05)
    ap.add_argument("--lesion", type=float, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--hours-per-epoch", type=float, default=None)
    ap.add_argument("--epoch-sim-ms", type=float, default=None)
    ap.add_argument("--output-dir", type=str, default=None)
    ap.add_argument("--label", type=str, default="degen")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--threads", type=int, default=1)
    args = ap.parse_args()
    run(scale=args.scale, lesion=args.lesion, epochs=args.epochs,
        hours_per_epoch=args.hours_per_epoch, epoch_sim_ms=args.epoch_sim_ms,
        output_dir=args.output_dir, label=args.label,
        seed=args.seed, n_threads=args.threads)


if __name__ == "__main__":
    _cli()
