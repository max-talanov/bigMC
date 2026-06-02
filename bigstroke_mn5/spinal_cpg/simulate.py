"""
Spinal CPG simulation driver.

Usage:
    python -m bigstroke_mn5.spinal_cpg.simulate \
        --scale 0.05 --stroke 0.36 --sim-time-ms 4000 --label local_stroke
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


def run(scale=None, stroke=None, sim_time_ms=None,
        output_dir=None, label="local", seed=42, n_threads=1):
    scale = scale if scale is not None else P.NETWORK_SCALE
    stroke = stroke if stroke is not None else P.STROKE_FRACTION_L
    sim_time_ms = sim_time_ms if sim_time_ms is not None else P.SIM_TIME_MS

    if output_dir is None:
        output_dir = Path(__file__).parent.parent / "output"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print(f"  BigStroke-MN5 — Spinal CPG  ({label})")
    print("=" * 72)
    print(f"  scale            = {scale}")
    print(f"  stroke reduction = {stroke}")
    print(f"  sim_time_ms      = {sim_time_ms}")
    print(f"  output_dir       = {output_dir}")
    print("=" * 72)

    try:
        import nest
        print(f"[NEST] version {nest.__version__}")
    except ImportError:
        print("[ERROR] NEST is not installed. See bigstroke_mn5/README.md.")
        sys.exit(1)

    t0 = time.time()

    print("\n[Build] Constructing bilateral spinal CPG ...")
    net = N.build_network(scale=scale, stroke_fraction=stroke,
                          seed=seed, n_threads=n_threads)
    pops = net["populations"]
    sizes = net["sizes"]
    print(f"[Build] {sum(sizes.values()):,} neurons across {len(pops)} populations.")
    print(f"[Build] Descending drive: L={net['drive_L_hz']:.0f} Hz  R={net['drive_R_hz']:.0f} Hz")
    if stroke > 0:
        f_core = stroke * P.CORE_FRACTION
        f_peri = stroke * (1.0 - P.CORE_FRACTION)
        print(f"[Build] Graded stroke (left M1 L5e → V2a drive):")
        print(f"         Total lesion:   {stroke*100:.1f}% of L M1 L5e")
        print(f"         Core:           {f_core*100:.1f}%  → 0% drive  (silenced)")
        print(f"         Penumbra:       {f_peri*100:.1f}%  → "
              f"{P.PENUMBRA_DRIVE_FRACTION*100:.0f}% drive (impaired)")
        print(f"         Effective drive reduction: "
              f"{net['drive_reduction_L']*100:.1f}%  "
              f"(drive {net['drive_L_hz']:.0f} Hz)")

    print("\n[Stim ] Background Poisson per population ...")
    S.attach_background(pops)

    print("[Stim ] Descending CST/M1 drive to V2a ...")
    S.attach_descending_drive(pops, net["drive_L_hz"], net["drive_R_hz"])

    print("[Rec  ] Spike recorders per population ...")
    recs = R.attach_spike_recorders(pops)

    print(f"\n[Sim  ] Simulating {sim_time_ms} ms ...")
    import nest
    t_sim_start = time.time()
    nest.Simulate(sim_time_ms)
    t_sim = time.time() - t_sim_start
    rtf = t_sim / (sim_time_ms / 1000.0)
    print(f"[Sim  ] Wall {t_sim:.1f}s, RTF {rtf:.1f}x")

    print("\n[Anal ] Firing rates per population ...")
    rates = A.compute_firing_rates(recs, sizes)
    for pop in P.POP_NAMES:
        l_full = P.pop_full_name("L", pop)
        r_full = P.pop_full_name("R", pop)
        print(f"  {pop:>5s}  L={rates[l_full]:>6.2f} Hz   R={rates[r_full]:>6.2f} Hz")

    print("\n[Anal ] Gait metrics from MN pools ...")
    metrics = A.compute_gait_metrics(recs, sizes)
    print(f"  MN_L mean rate  = {metrics['mn_L_mean_hz']:.2f} Hz")
    print(f"  MN_R mean rate  = {metrics['mn_R_mean_hz']:.2f} Hz")
    print(f"  AmpAI (gait asymmetry) = {metrics['AmpAI']*100:+.1f}%")

    # Outputs
    raster_path = output_dir / f"spinal_raster_{label}.png"
    mn_path = output_dir / f"spinal_mn_envelopes_{label}.png"
    csv_path = output_dir / f"spinal_summary_{label}.csv"

    A.plot_raster(recs, sizes, raster_path,
                  t_window_ms=(P.WARMUP_MS, min(P.WARMUP_MS + 2000, sim_time_ms)),
                  title=f"Spinal CPG ({label}, scale={scale}, stroke={stroke})")
    A.plot_mn_envelopes(metrics, mn_path,
                        title=f"Motoneuron pool rates — {label} "
                              f"(AmpAI={metrics['AmpAI']*100:+.1f}%)")
    A.save_summary_csv(rates, metrics, csv_path)

    t_total = time.time() - t0
    print(f"\n{'='*72}")
    print(f"  DONE in {t_total:.1f}s  (build {t_total-t_sim:.1f}s + sim {t_sim:.1f}s)")
    print(f"  Raster:       {raster_path}")
    print(f"  MN envelopes: {mn_path}")
    print(f"  CSV:          {csv_path}")
    print("=" * 72)
    return {"rates": rates, "metrics": metrics, "wall_s": t_total, "rtf": rtf}


def _cli():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scale", type=float, default=None)
    ap.add_argument("--stroke", type=float, default=None,
                    help="Fraction reduction of left descending drive (0.0=healthy, 0.36=Phase 1)")
    ap.add_argument("--sim-time-ms", type=float, default=None)
    ap.add_argument("--output-dir", type=str, default=None)
    ap.add_argument("--label", type=str, default="local")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--threads", type=int, default=1)
    args = ap.parse_args()
    run(scale=args.scale, stroke=args.stroke,
        sim_time_ms=args.sim_time_ms, output_dir=args.output_dir,
        label=args.label, seed=args.seed, n_threads=args.threads)


if __name__ == "__main__":
    _cli()
