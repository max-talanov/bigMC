"""
Spike recorders per population.

For Stage 1, a single spike_recorder per population is sufficient.
Voltage meters (sample subset for raster plots) are optional.
"""
from __future__ import annotations
from . import params as P


def _nest():
    import nest
    return nest


def attach_spike_recorders(populations: dict) -> dict:
    """
    Attach one spike_recorder per population.
    Returns {full_name: spike_recorder_node}.
    """
    nest = _nest()
    recorders = {}
    for full_name, nodes in populations.items():
        sr = nest.Create("spike_recorder", params={
            "record_to": "memory",
        })
        nest.Connect(nodes, sr)
        recorders[full_name] = sr
    return recorders


def attach_voltage_meters(populations: dict, n_per_pop: int = 5) -> dict:
    """
    Attach voltmeters to a small sample of neurons per population
    for raster / voltage trace plots.
    """
    nest = _nest()
    voltmeters = {}
    for full_name, nodes in populations.items():
        n_sample = min(n_per_pop, len(nodes))
        vm = nest.Create("voltmeter", params={
            "interval": P.DT_MS,
            "record_to": "memory",
        })
        nest.Connect(vm, nodes[:n_sample])
        voltmeters[full_name] = vm
    return voltmeters


def get_spike_data(recorder) -> dict:
    """Extract spike times and senders from a recorder. Returns
    {'times': np.ndarray, 'senders': np.ndarray}."""
    import numpy as np
    events = recorder.get("events")
    return {
        "times":   np.asarray(events["times"]),
        "senders": np.asarray(events["senders"]),
    }
