"""
Main simulation driver — orchestrates network build, stimulus,
recording, simulation, and analysis.

Usage:
    python -m bigstroke_mn5.m1_microcircuit.simulate \
        --scale 0.01 \
        --stroke-fraction 0.0 \
        --sim-time-ms 1000 \
        --output-dir bigstroke_mn5/output
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

from . import params as P
from . import network as N
from . import stimulus as S
from . import recording as R
from . import analysis as A


def run(scale: float = None,
        stroke_fraction: float = None,
        sim_time_ms: float = None,
        output_dir: Path = None,
        label: str = "local",
        seed: int = 42,
        n_threads: int = 1):
    """Run one full simulation and produce QC outputs."""
    scale = scale if scale is not None else P.NETWORK_SCALE
    stroke_fraction = (stroke_fraction if stroke_fraction is not None
                       else P.STROKE_FRACTION_L5e_LEFT)
    sim_time_ms = sim_time_ms if sim_time_ms is not None else P.SIM_TIME_MS

    if output_dir is None:
        output_dir = Path(__file__).parent.parent / "output"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print(f"  BigStroke-MN5 — M1 Microcircuit  ({label})")
    print("=" * 72)
    print(f"  scale            = {scale}")
    print(f"  stroke_fraction  = {stroke_fraction}")
    print(f"  sim_time_ms      = {sim_time_ms}")
    print(f"  seed             = {seed}")
    print(f"  n_threads        = {n_threads}")
    print(f"  output_dir       = {output_dir}")
    print("=" * 72)

    # ── Check NEST is available ─────────────────────────────────────────
    try:
        import nest  # noqa: F401
        print(f"[NEST] version {nest.__version__}")
    except ImportError:
        print("[ERROR] NEST is not installed.")
        print("        Install via:  pip install nest-simulator")
        print("                or:   conda install -c conda-forge nest-simulator")
        print("        See bigstroke_mn5/README.md for full instructions.")
        sys.exit(1)

    t0 = time.time()

    # ── Build the network ───────────────────────────────────────────────
    print("\n[Build] Constructing network ...")
    net = N.build_network(scale=scale, stroke_fraction=stroke_fraction,
                          seed=seed, n_threads=n_threads)
    populations = net["populations"]
    sizes = net["sizes"]
    n_total = sum(sizes.values())
    print(f"[Build] {n_total:,} neurons across {len(populations)} populations.")
    if stroke_fraction > 0:
        n_L5e = sizes["L_L5e"]
        print(f"[Build] Stroke model (graded lesion in left L5e, n={n_L5e}):")
        print(f"         Core     (fully silenced) : {net['n_core']:>4d} neurons "
              f"({net['n_core']/n_L5e*100:.0f}%)")
        print(f"         Penumbra (impaired, active): {net['n_penumbra']:>4d} neurons "
              f"({net['n_penumbra']/n_L5e*100:.0f}%)")
        print(f"         Healthy  (unaffected)      : "
              f"{n_L5e - net['n_core'] - net['n_penumbra']:>4d} neurons "
              f"({(n_L5e - net['n_core'] - net['n_penumbra'])/n_L5e*100:.0f}%)")
        print(f"         → Diaschisis: callosal projections from core naturally "
              f"silent (wired but unfired)")

    t_build = time.time() - t0

    # ── Attach stimulus and recorders ───────────────────────────────────
    print("\n[Stim ] Attaching background Poisson generators ...")
    S.attach_background(populations)

    print("[Rec  ] Attaching spike recorders ...")
    recorders = R.attach_spike_recorders(populations)

    # ── Simulate ────────────────────────────────────────────────────────
    print(f"\n[Sim  ] Running {sim_time_ms} ms of biological time ...")
    import nest
    t_sim_start = time.time()
    nest.Simulate(sim_time_ms)
    t_sim = time.time() - t_sim_start

    rtf = t_sim / (sim_time_ms / 1000.0)
    print(f"[Sim  ] Wall time {t_sim:.1f} s  →  real-time factor = {rtf:.1f}x")

    # ── Analyse ─────────────────────────────────────────────────────────
    print("\n[Anal ] Computing firing rates and ISI CVs ...")
    rates = A.compute_firing_rates(recorders, sizes)
    cvs = A.compute_isi_cv(recorders)

    print(f"\n  Population firing rates (Hz):")
    print(f"  {'Population':>10s}  {'rate':>7s}  {'CV':>5s}  {'target':>7s}")
    for full_name in sorted(rates):
        pop = full_name.split("_", 1)[1]
        targets = {
            "L23e": 0.74, "L23i": 2.65, "L4e": 4.40, "L4i": 5.40,
            "L5e":  6.78, "L5i":  8.31, "L6e": 1.10, "L6i": 7.61,
        }
        print(f"  {full_name:>10s}  {rates[full_name]:>7.2f}  "
              f"{cvs[full_name]:>5.2f}  {targets[pop]:>7.2f}")

    # ── Save outputs ────────────────────────────────────────────────────
    raster_path = output_dir / f"raster_{label}.png"
    csv_path = output_dir / f"firing_rates_{label}.csv"

    print(f"\n[Out  ] Saving raster plot ...")
    A.plot_raster(recorders, sizes, raster_path,
                  t_window_ms=(P.WARMUP_MS, min(P.WARMUP_MS + 500, sim_time_ms)),
                  title=f"M1 microcircuit ({label}, scale={scale}, "
                        f"stroke={stroke_fraction})")
    print(f"[Out  ] Saving firing-rate CSV ...")
    A.save_rates_csv(rates, cvs, sizes, csv_path)

    t_total = time.time() - t0
    print(f"\n{'='*72}")
    print(f"  DONE in {t_total:.1f} s "
          f"(build {t_build:.1f}s + sim {t_sim:.1f}s + analysis {t_total-t_build-t_sim:.1f}s)")
    print(f"  Raster: {raster_path}")
    print(f"  CSV:    {csv_path}")
    print("=" * 72)

    return {
        "rates": rates,
        "cvs": cvs,
        "sizes": sizes,
        "wall_time_s": t_total,
        "real_time_factor": rtf,
    }


def _cli():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scale", type=float, default=None,
                    help="Network scale (0.01 to 1.0)")
    ap.add_argument("--stroke-fraction", type=float, default=None,
                    help="Fraction of left-M1 L5e to silence (0.0 = healthy)")
    ap.add_argument("--sim-time-ms", type=float, default=None)
    ap.add_argument("--output-dir", type=str, default=None)
    ap.add_argument("--label", type=str, default="local")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--threads", type=int, default=1)
    args = ap.parse_args()

    run(scale=args.scale,
        stroke_fraction=args.stroke_fraction,
        sim_time_ms=args.sim_time_ms,
        output_dir=args.output_dir,
        label=args.label,
        seed=args.seed,
        n_threads=args.threads)


if __name__ == "__main__":
    _cli()
