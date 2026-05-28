"""
Build the bilateral M1 microcircuit in NEST.

Hierarchical structure:
  16 populations = 8 populations × 2 hemispheres
  Intra-hemispheric: full Potjans-Diesmann connectivity
  Inter-hemispheric: callosal projections (L23e/L5e ↔ L23e/L5e)
  Stroke: silence a fraction of L5e in the left hemisphere

Returns a dict of {full_name: nest.NodeCollection} for downstream use.
"""
from __future__ import annotations
import numpy as np

from . import params as P


def _nest():
    """Lazy import so that the rest of the package is importable without NEST."""
    import nest
    return nest


def build_network(scale=None, stroke_fraction=None, seed=42, n_threads=1):
    """
    Build the full bilateral M1 microcircuit and return a layout dict.

    Parameters
    ----------
    scale : float or None
        Override the default NETWORK_SCALE.
    stroke_fraction : float or None
        Override STROKE_FRACTION_L5e_LEFT (0.0 = healthy, 0.5 = severe).
    seed : int
        Master RNG seed for NEST.
    n_threads : int
        Local OpenMP threads per MPI process.

    Returns
    -------
    dict
        {
          "populations": {full_name: NodeCollection, ...},
          "sizes":       {full_name: int, ...},
          "stroke_silenced": NodeCollection or None,
          "scale": float,
        }
    """
    nest = _nest()

    if scale is None:
        scale = P.NETWORK_SCALE
    if stroke_fraction is None:
        stroke_fraction = P.STROKE_FRACTION_L5e_LEFT

    # ── Kernel reset & global config ────────────────────────────────────
    nest.ResetKernel()
    nest.SetKernelStatus({
        "resolution":      P.DT_MS,
        "rng_seed":        seed,
        "local_num_threads": n_threads,
        "print_time":      False,
    })

    sizes = P.get_pop_sizes(scale)

    # ── Create populations ──────────────────────────────────────────────
    populations = {}
    for hemi in P.HEMISPHERES:
        for pop in P.POP_NAMES:
            full_name = P.pop_full_name(hemi, pop)
            n = sizes[pop]
            nodes = nest.Create(P.NEURON_MODEL, n, params=P.NEURON_PARAMS)
            # Initialise membrane voltages uniformly within (E_L, V_th)
            v_init = np.random.uniform(
                low=P.NEURON_PARAMS["E_L"],
                high=P.NEURON_PARAMS["V_th"],
                size=n,
            )
            nodes.set(V_m=list(v_init))
            populations[full_name] = nodes

    # ── Apply stroke (silence a fraction of L5e in left hemisphere) ─────
    stroke_silenced = None
    if stroke_fraction > 0:
        L_L5e = populations["L_L5e"]
        n_silence = int(round(len(L_L5e) * stroke_fraction))
        if n_silence > 0:
            # Pick first n_silence (deterministic for reproducibility)
            silenced = L_L5e[:n_silence]
            # Silence by raising threshold beyond reach
            silenced.set({"V_th": 1e6, "V_reset": -65.0})
            stroke_silenced = silenced

    # ── Connect populations (intra-hemispheric Potjans-Diesmann) ────────
    for hemi in P.HEMISPHERES:
        for tgt_pop in P.POP_NAMES:
            tgt = populations[P.pop_full_name(hemi, tgt_pop)]
            t_idx = P.population_index(tgt_pop)
            for src_pop in P.POP_NAMES:
                src = populations[P.pop_full_name(hemi, src_pop)]
                s_idx = P.population_index(src_pop)
                p = P.CONN_PROB[t_idx, s_idx]
                if p <= 0:
                    continue

                # Number of input synapses per target neuron
                # K = ln(1 - p) / ln(1 - 1/(N_src * N_tgt)) * N_src * N_tgt (full formula)
                # Simplified Potjans formula:
                n_syn = max(1, int(round(
                    np.log(1.0 - p) /
                    np.log(1.0 - 1.0 / (len(src) * len(tgt)))
                ))) if p < 1.0 else len(src) * len(tgt)

                # Weight: positive for E, negative for I
                if P.is_excitatory(src_pop):
                    w_mean = _psc_from_psp(P.PSP_E)
                    delay_mean = P.DELAY_E_MS
                else:
                    w_mean = P.G_INH_REL_E * _psc_from_psp(P.PSP_E)
                    delay_mean = P.DELAY_I_MS

                w_sd = P.PSP_REL_SD * abs(w_mean)
                d_sd = P.DELAY_REL_SD * delay_mean

                syn_spec = {
                    "synapse_model": "static_synapse",
                    "weight": nest.random.normal(mean=w_mean, std=w_sd),
                    "delay":  nest.math.redraw(
                        nest.random.normal(mean=delay_mean, std=d_sd),
                        min=P.DT_MS, max=20.0,
                    ),
                }

                conn_spec = {
                    "rule": "fixed_total_number",
                    "N": n_syn,
                    "allow_autapses": False,
                    "allow_multapses": True,
                }

                nest.Connect(src, tgt, conn_spec, syn_spec)

    # ── Callosal (inter-hemispheric) projections ────────────────────────
    for src_hemi, tgt_hemi in (("L", "R"), ("R", "L")):
        for src_pop in P.CALLOSAL_SOURCES:
            for tgt_pop in P.CALLOSAL_TARGETS:
                src = populations[P.pop_full_name(src_hemi, src_pop)]
                tgt = populations[P.pop_full_name(tgt_hemi, tgt_pop)]
                s_idx = P.population_index(src_pop)
                t_idx = P.population_index(tgt_pop)
                p = P.CONN_PROB[t_idx, s_idx] * P.CALLOSAL_PROB_SCALE
                if p <= 0:
                    continue

                n_syn = max(1, int(round(p * len(src) * len(tgt) / 10)))
                w_mean = _psc_from_psp(P.PSP_E)
                w_sd = P.PSP_REL_SD * abs(w_mean)
                syn_spec = {
                    "synapse_model": "static_synapse",
                    "weight": nest.random.normal(mean=w_mean, std=w_sd),
                    "delay": P.CALLOSAL_DELAY_MS,
                }
                conn_spec = {
                    "rule": "fixed_total_number",
                    "N": n_syn,
                    "allow_autapses": False,
                    "allow_multapses": True,
                }
                nest.Connect(src, tgt, conn_spec, syn_spec)

    return {
        "populations": populations,
        "sizes": {n: len(nc) for n, nc in populations.items()},
        "stroke_silenced": stroke_silenced,
        "scale": scale,
    }


def _psc_from_psp(psp_mv: float) -> float:
    """
    Convert mean PSP amplitude (mV) to PSC weight (pA) for iaf_psc_exp.

    Closed-form derivation in Potjans-Diesmann supplementary.  For
    tau_m = tau_syn, the PSP amplitude is:
        PSP_amp = (PSC / C_m) * (tau_m * tau_syn) / (tau_syn - tau_m) * (...)
    Here we use a simplified factor that works for the default
    Potjans-Diesmann parameter set (tau_m=10ms, tau_syn_ex=0.5ms).
    """
    tau_m = P.NEURON_PARAMS["tau_m"]
    tau_s = P.NEURON_PARAMS["tau_syn_ex"]
    C_m = P.NEURON_PARAMS["C_m"]

    # Approximate analytical conversion
    t_max = (tau_m * tau_s) / (tau_m - tau_s) * np.log(tau_m / tau_s)
    psc = psp_mv * C_m * (1.0 / tau_s - 1.0 / tau_m) / (
        np.exp(-t_max / tau_m) - np.exp(-t_max / tau_s)
    )
    return float(psc)
