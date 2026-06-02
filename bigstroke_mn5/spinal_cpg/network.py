"""
Build the bilateral spinal locomotor CPG in NEST.

Architecture: 8 populations × 2 sides = 16 populations.
Connectivity: ipsilateral (CONN_IPSI) + commissural (CONN_CONTRA).
"""
from __future__ import annotations
import numpy as np

from . import params as P


def _nest():
    import nest
    return nest


def build_network(scale=None, stroke_drive_reduction_L=None,
                  seed=42, n_threads=1):
    """
    Build the bilateral spinal CPG.

    Parameters
    ----------
    scale : float or None
        Override NETWORK_SCALE.
    stroke_drive_reduction_L : float or None
        Fraction by which to reduce descending drive on the LEFT side
        (0.0 = healthy; 0.36 = matches Phase 1 CST deficit).
    seed, n_threads : NEST kernel params.

    Returns
    -------
    dict {populations, sizes, scale, drive_L_hz, drive_R_hz}
    """
    nest = _nest()

    if scale is None:
        scale = P.NETWORK_SCALE
    if stroke_drive_reduction_L is None:
        stroke_drive_reduction_L = P.STROKE_DRIVE_REDUCTION_L

    nest.ResetKernel()
    nest.SetKernelStatus({
        "resolution":        P.DT_MS,
        "rng_seed":          seed,
        "local_num_threads": n_threads,
        "print_time":        False,
    })

    sizes = P.get_pop_sizes(scale)

    # ── Create populations ──────────────────────────────────────────────
    populations = {}
    for side in P.SIDES:
        for pop in P.POP_NAMES:
            full_name = P.pop_full_name(side, pop)
            n = sizes[pop]
            params_use = P.MN_PARAMS if pop in P.MOTONEURONS else P.NEURON_PARAMS
            nodes = nest.Create(P.NEURON_MODEL, n, params=params_use)
            # Randomise V_m for diversity
            v_init = np.random.uniform(
                low=params_use["E_L"],
                high=params_use["V_th"],
                size=n,
            )
            nodes.set(V_m=list(v_init))
            populations[full_name] = nodes

    # ── Ipsilateral connections ─────────────────────────────────────────
    full_sizes = P.N_FULL
    for side in P.SIDES:
        for tgt_pop in P.POP_NAMES:
            t_idx = P.population_index(tgt_pop)
            tgt = populations[P.pop_full_name(side, tgt_pop)]
            for src_pop in P.POP_NAMES:
                s_idx = P.population_index(src_pop)
                p = P.CONN_IPSI[t_idx, s_idx]
                if p <= 0:
                    continue
                src = populations[P.pop_full_name(side, src_pop)]
                _connect(nest, src, tgt, p,
                         src_pop, tgt_pop,
                         delay_ms=P.DELAY_E_MS if P.is_excitatory(src_pop) else P.DELAY_I_MS,
                         is_contra=False,
                         full_sizes=full_sizes)

    # ── Commissural connections (cross-midline) ─────────────────────────
    for src_side, tgt_side in (("L", "R"), ("R", "L")):
        for tgt_pop in P.POP_NAMES:
            t_idx = P.population_index(tgt_pop)
            tgt = populations[P.pop_full_name(tgt_side, tgt_pop)]
            for src_pop in P.POP_NAMES:
                s_idx = P.population_index(src_pop)
                p = P.CONN_CONTRA[t_idx, s_idx]
                if p <= 0:
                    continue
                src = populations[P.pop_full_name(src_side, src_pop)]
                _connect(nest, src, tgt, p,
                         src_pop, tgt_pop,
                         delay_ms=P.DELAY_CONTRA_MS,
                         is_contra=True,
                         full_sizes=full_sizes)

    # ── Compute descending (CST/M1) drive per side ──────────────────────
    drive_L = P.DESCENDING_DRIVE_L_HZ * (1.0 - stroke_drive_reduction_L)
    drive_R = P.DESCENDING_DRIVE_R_HZ

    return {
        "populations": populations,
        "sizes": {n: len(nc) for n, nc in populations.items()},
        "scale": scale,
        "drive_L_hz": drive_L,
        "drive_R_hz": drive_R,
    }


def _connect(nest, src, tgt, p, src_pop, tgt_pop,
             delay_ms, is_contra, full_sizes):
    """Internal: create one population-to-population connection block."""
    N_src_full = full_sizes[src_pop]
    N_tgt_full = full_sizes[tgt_pop]

    if P.PRESERVE_INDEGREE_AT_SCALE and p < 1.0:
        n_total_full = int(round(
            np.log(1.0 - p) /
            np.log(1.0 - 1.0 / (N_src_full * N_tgt_full))
        ))
        K_indegree = max(1, int(round(n_total_full / N_tgt_full)))
    else:
        K_indegree = max(1, int(round(p * len(src))))

    # Weight
    if P.is_excitatory(src_pop):
        w_mean = _psc_from_psp(P.PSP_E)
    else:
        w_mean = P.G_INH_REL_E * _psc_from_psp(P.PSP_E)

    w_sd = P.PSP_REL_SD * abs(w_mean)
    d_sd = P.DELAY_REL_SD * delay_ms

    syn_spec = {
        "synapse_model": "static_synapse",
        "weight": nest.random.normal(mean=w_mean, std=w_sd),
        "delay":  nest.math.redraw(
            nest.random.normal(mean=delay_ms, std=d_sd),
            min=P.DT_MS, max=30.0,
        ),
    }
    conn_spec = {
        "rule": "fixed_indegree",
        "indegree": K_indegree,
        "allow_autapses": False,
        "allow_multapses": True,
    }
    nest.Connect(src, tgt, conn_spec, syn_spec)


def _psc_from_psp(psp_mv: float) -> float:
    """PSP→PSC conversion for iaf_psc_exp (closed-form)."""
    tau_m = P.NEURON_PARAMS["tau_m"]
    tau_s = P.NEURON_PARAMS["tau_syn_ex"]
    C_m   = P.NEURON_PARAMS["C_m"]
    t_max = (tau_m * tau_s) / (tau_m - tau_s) * np.log(tau_m / tau_s)
    psc = psp_mv * C_m * (1.0 / tau_s - 1.0 / tau_m) / (
        np.exp(-t_max / tau_m) - np.exp(-t_max / tau_s)
    )
    return float(psc)
