"""Spike recorders for the spinal CPG."""
from __future__ import annotations
from . import params as P


def _nest():
    import nest
    return nest


def attach_spike_recorders(populations: dict) -> dict:
    """One spike_recorder per population. Returns {full_name: sr}."""
    nest = _nest()
    recs = {}
    for full_name, nodes in populations.items():
        sr = nest.Create("spike_recorder", params={"record_to": "memory"})
        nest.Connect(nodes, sr)
        recs[full_name] = sr
    return recs


def get_spike_data(recorder):
    import numpy as np
    events = recorder.get("events")
    return {
        "times":   np.asarray(events["times"]),
        "senders": np.asarray(events["senders"]),
    }
