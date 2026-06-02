"""
External and descending inputs for the spinal CPG.

Two streams:
  1. Diffuse background Poisson per population (afferent / brainstem noise)
  2. Descending (CST/M1) drive targeting V2a per side, with stroke asymmetry
"""
from __future__ import annotations
from . import params as P
from .network import _psc_from_psp


def _nest():
    import nest
    return nest


def attach_background(populations: dict) -> dict:
    """Poisson background per population.  Returns {full_name: pg}."""
    nest = _nest()
    pgs = {}
    w = _psc_from_psp(P.PSP_E)
    for full_name, nodes in populations.items():
        pop = full_name.split("_", 1)[1]
        k_ext = P.K_EXT[pop]
        rate = P.BG_RATE_HZ * k_ext
        pg = nest.Create("poisson_generator", params={"rate": rate})
        nest.Connect(pg, nodes, syn_spec={"weight": w, "delay": P.DT_MS})
        pgs[full_name] = pg
    return pgs


def attach_descending_drive(populations: dict, drive_L_hz: float,
                            drive_R_hz: float) -> dict:
    """
    Tonic descending drive from CST/M1 to V2a per side.

    drive_*_hz is the *total* Poisson rate (not per-synapse); use a single
    Poisson generator per side and broadcast to all V2a neurons.
    """
    nest = _nest()
    descending = {}
    w = _psc_from_psp(P.PSP_E) * 1.5  # stronger than recurrent (suprathreshold)

    for side, rate in (("L", drive_L_hz), ("R", drive_R_hz)):
        if rate <= 0:
            continue
        target = populations[P.pop_full_name(side, P.DESCENDING_TARGET)]
        pg = nest.Create("poisson_generator", params={"rate": rate})
        nest.Connect(pg, target, syn_spec={"weight": w, "delay": P.DT_MS})
        descending[side] = pg
    return descending
