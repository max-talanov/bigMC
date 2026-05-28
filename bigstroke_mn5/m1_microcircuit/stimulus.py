"""
Background Poisson input + (later) NIBS injection.

Stage 1: background Poisson generator per population to drive baseline firing.
Stage 4: tDCS (constant V_m offset) and rTMS (suprathreshold pulses) hooks.
"""
from __future__ import annotations
from . import params as P


def _nest():
    import nest
    return nest


def attach_background(populations: dict) -> dict:
    """
    Attach one Poisson generator per population, modeled as a single
    presynaptic source with rate = ν_bg × K_ext (number of external synapses).

    Returns a dict {full_name: poisson_node}.
    """
    nest = _nest()
    poissons = {}

    for full_name, nodes in populations.items():
        # full_name e.g. 'L_L23e' → pop = 'L23e'
        pop = full_name.split("_", 1)[1]
        k_ext = P.K_EXT[pop]
        rate = P.BG_RATE_HZ * k_ext

        pg = nest.Create("poisson_generator", params={"rate": rate})
        # Connect with the same weight as an excitatory synapse
        from .network import _psc_from_psp
        w = _psc_from_psp(P.PSP_E)
        nest.Connect(
            pg, nodes,
            syn_spec={"weight": w, "delay": P.DT_MS},
        )
        poissons[full_name] = pg

    return poissons


def apply_tdcs(populations: dict, hemi: str, polarity: str = "anodal",
               magnitude_mv: float = 0.5):
    """
    Apply (Stage 4) sub-threshold tDCS to L5 pyramidal somata.

    Anodal: depolarizing (positive V_m offset).
    Cathodal: hyperpolarizing.

    Currently a stub; full implementation depends on NEST's I_e current
    or a custom NESTML neuron with an external field input.
    """
    nest = _nest()
    sign = +1 if polarity == "anodal" else -1
    target = populations[P.pop_full_name(hemi, "L5e")]
    # Convert mV offset → equivalent constant input current (pA)
    R_m = P.NEURON_PARAMS["tau_m"] / P.NEURON_PARAMS["C_m"] * 1000  # MOhm
    i_e_pa = sign * magnitude_mv / R_m * 1000  # mV / MOhm → nA → pA × 1000
    target.set({"I_e": float(i_e_pa)})


def apply_rtms(populations: dict, hemi: str, freq_hz: float = 10.0,
               n_pulses: int = 1500, pulse_amp_mv: float = 25.0):
    """
    Apply rTMS as suprathreshold depolarization bursts to L5 pyramidals.

    Stub — full implementation uses a spike_generator that injects PSCs
    large enough to drive immediate spiking.
    """
    raise NotImplementedError(
        "rTMS not yet implemented; will be added in Stage 4 alongside STDP."
    )
