"""
Multi-timescale stroke degeneration engine.

Operator splitting across timescales:
  • Fast  (ms)    : NEST spike simulation, run in short "epochs"
  • Slow  (hours) : between epochs, advance a biological degeneration clock
                    and convert penumbra neurons to core (silenced) following
                    a cell-type-specific schedule.

STAGE A (this module): time-phased death.
  Each penumbra neuron is assigned a death time at lesion onset, drawn from a
  distribution whose mean depends on its vulnerability class:

      excitatory          → DEATH_TIME_HOURS['excitatory']   (fast)
      pv_inhibitory       → DEATH_TIME_HOURS['pv_inhibitory'] (fast-ish)
      buffered_inhibitory → ∞  (survives; recovers after RECOVERY_TIME_HOURS)

  As the slow clock advances, neurons whose death time has passed are silenced.
  Buffered interneurons instead recover toward healthy parameters.

STAGE B (planned): the death time will be modulated in real time by each
  neuron's firing rate (activity-dependent excitotoxicity), so that the
  loss of inhibition accelerates excitatory death (disinhibition feedback).
  This module is structured so Stage B only needs to update death times
  inside `step()` from measured rates.

Selective neuronal vulnerability references:
  Hofmeijer & van Putten (2012) Stroke 43:607-615
  Lo, Dalkara, Moskowitz (2003) Nat Rev Neurosci 4:399-415
  Kreuzberg et al. (2010)  — PV+ interneuron vulnerability
  Povysheva et al. (2019)  — interneuron Ca2+ buffering / resistance
"""
from __future__ import annotations
import numpy as np

from . import params as P


def _nest():
    import nest
    return nest


# ════════════════════════════════════════════════════════════════════════════
#  Vulnerability classification
# ════════════════════════════════════════════════════════════════════════════

def classify_vulnerability(populations, seed=42):
    """
    Assign each neuron a vulnerability class.

    Excitatory populations  → all 'excitatory'.
    Inhibitory populations   → a PV_FRACTION subset 'pv_inhibitory',
                               remainder 'buffered_inhibitory'.

    Returns
    -------
    dict {global_id (int): vulnerability_class (str)}
    """
    rng = np.random.default_rng(seed + 7)
    vuln = {}
    for full_name, nodes in populations.items():
        pop = full_name.split("_", 1)[1]
        ids = list(nodes.global_id) if len(nodes) > 1 else [int(nodes.global_id)]
        if P.is_excitatory(pop):
            for nid in ids:
                vuln[int(nid)] = "excitatory"
        else:
            n = len(ids)
            n_pv = int(round(n * P.PV_FRACTION))
            perm = rng.permutation(n)
            pv_set = set(perm[:n_pv].tolist())
            for k, nid in enumerate(ids):
                vuln[int(nid)] = ("pv_inhibitory" if k in pv_set
                                  else "buffered_inhibitory")
    return vuln


# ════════════════════════════════════════════════════════════════════════════
#  Lesion initialisation (spatial infarct across all LEFT populations)
# ════════════════════════════════════════════════════════════════════════════

def init_lesion(populations, vuln_map,
                lesion_fraction=None, core_fraction=None,
                hemi="L", seed=42):
    """
    Create a spatial infarct: a fraction of every population in `hemi`
    is lesioned.  Within the lesion, core_fraction is silenced immediately;
    the rest become penumbra with assigned death times.

    Returns a `degen_state` dict to be passed to `step()`.
    """
    nest = _nest()
    if lesion_fraction is None:
        lesion_fraction = P.DEGEN_LESION_FRACTION
    if core_fraction is None:
        core_fraction = P.STROKE_CORE_FRACTION

    rng = np.random.default_rng(seed + 11)

    core_ids = []
    penumbra = []   # list of dicts: {id, vuln, death_t, recover_t, status}

    for pop in P.POP_NAMES:
        full_name = P.pop_full_name(hemi, pop)
        nodes = populations[full_name]
        ids = list(nodes.global_id) if len(nodes) > 1 else [int(nodes.global_id)]
        n = len(ids)
        n_lesion = int(round(n * lesion_fraction))
        if n_lesion == 0:
            continue

        perm = rng.permutation(n)
        lesion_local = perm[:n_lesion]
        n_core = int(round(n_lesion * core_fraction))
        core_local = lesion_local[:n_core]
        peri_local = lesion_local[n_core:]

        # Core: silence immediately
        for li in core_local:
            core_ids.append(int(ids[li]))

        # Penumbra: impair + assign death time by vulnerability class
        base_E_L  = P.NEURON_PARAMS["E_L"]
        base_V_th = P.NEURON_PARAMS["V_th"]
        base_tau  = P.NEURON_PARAMS["tau_m"]
        for li in peri_local:
            nid = int(ids[li])
            vclass = vuln_map[nid]
            mean_death = P.DEATH_TIME_HOURS[vclass]
            if np.isinf(mean_death):
                death_t = np.inf
            else:
                sd = P.DEATH_TIME_REL_SD * mean_death
                death_t = max(0.1, float(rng.normal(mean_death, sd)))
            penumbra.append({
                "id":        nid,
                "vuln":      vclass,
                "death_t":   death_t,
                "recover_t": P.RECOVERY_TIME_HOURS,
                "status":    "penumbra",
            })

    # Apply: silence core, impair penumbra
    if core_ids:
        nest.NodeCollection(sorted(core_ids)).set({"V_th": 1e6, "V_reset": -65.0})
    _apply_penumbra_params(nest, [d["id"] for d in penumbra])

    state = {
        "clock_hours":   0.0,
        "core_ids":      core_ids,
        "penumbra":      penumbra,
        "vuln_map":      vuln_map,
        "history":       [],     # filled by record_survival()
        "rate_history":  [],     # filled by simulate driver
    }
    record_survival(state)
    return state


def _apply_penumbra_params(nest, ids):
    if not ids:
        return
    base_E_L  = P.NEURON_PARAMS["E_L"]
    base_V_th = P.NEURON_PARAMS["V_th"]
    base_tau  = P.NEURON_PARAMS["tau_m"]
    nest.NodeCollection(sorted(ids)).set({
        "E_L":   base_E_L  + P.PENUMBRA_E_L_SHIFT_MV,
        "V_th":  base_V_th + P.PENUMBRA_V_TH_SHIFT_MV,
        "tau_m": base_tau  * P.PENUMBRA_TAU_M_FACTOR,
        "V_m":   base_E_L  + P.PENUMBRA_E_L_SHIFT_MV,
    })


def _restore_healthy_params(nest, ids):
    if not ids:
        return
    nest.NodeCollection(sorted(ids)).set({
        "E_L":   P.NEURON_PARAMS["E_L"],
        "V_th":  P.NEURON_PARAMS["V_th"],
        "tau_m": P.NEURON_PARAMS["tau_m"],
    })


# ════════════════════════════════════════════════════════════════════════════
#  Slow-timescale step
# ════════════════════════════════════════════════════════════════════════════

def step(state, hours_advance=None, firing_rates=None):
    """
    Advance the biological degeneration clock by `hours_advance` and update
    penumbra neuron fates.

    Parameters
    ----------
    state         : dict from init_lesion (mutated in place)
    hours_advance : float biological hours to advance (default per-epoch)
    firing_rates  : optional {global_id: rate_hz} — Stage B hook.  Currently
                    unused for fate decisions (time-phased), but recorded.

    Stage B note: when activity-dependent death is enabled, this is where
    death_t would be decremented faster for high-firing neurons:
        d["death_t"] -= k_excito * firing_rates[d["id"]] * hours_advance
    """
    nest = _nest()
    if hours_advance is None:
        hours_advance = P.DEGEN_HOURS_PER_EPOCH

    state["clock_hours"] += hours_advance
    clock = state["clock_hours"]

    newly_dead = []
    newly_recovered = []
    for d in state["penumbra"]:
        if d["status"] != "penumbra":
            continue
        if clock >= d["death_t"]:
            newly_dead.append(d["id"])
            d["status"] = "dead"
        elif d["vuln"] == "buffered_inhibitory" and clock >= d["recover_t"]:
            newly_recovered.append(d["id"])
            d["status"] = "recovered"

    if newly_dead:
        nest.NodeCollection(sorted(newly_dead)).set({"V_th": 1e6, "V_reset": -65.0})
    if newly_recovered:
        _restore_healthy_params(nest, newly_recovered)

    record_survival(state)
    return {"newly_dead": len(newly_dead),
            "newly_recovered": len(newly_recovered)}


# ════════════════════════════════════════════════════════════════════════════
#  Survival bookkeeping
# ════════════════════════════════════════════════════════════════════════════

def record_survival(state):
    """Append current surviving-neuron counts (by class) to history."""
    counts = {"excitatory": {"alive": 0, "dead": 0},
              "pv_inhibitory": {"alive": 0, "dead": 0},
              "buffered_inhibitory": {"alive": 0, "dead": 0}}

    # core neurons count as dead, attributed by their class
    for nid in state["core_ids"]:
        c = state["vuln_map"][nid]
        counts[c]["dead"] += 1

    for d in state["penumbra"]:
        c = d["vuln"]
        if d["status"] == "dead":
            counts[c]["dead"] += 1
        else:  # penumbra or recovered both count as "alive"
            counts[c]["alive"] += 1

    state["history"].append({
        "hour": state["clock_hours"],
        "counts": {k: dict(v) for k, v in counts.items()},
    })


def survival_summary(state):
    """Return a human-readable summary of the final state."""
    last = state["history"][-1]["counts"]
    lines = []
    for c in ("excitatory", "pv_inhibitory", "buffered_inhibitory"):
        alive = last[c]["alive"]
        dead = last[c]["dead"]
        total = alive + dead
        pct = 100.0 * dead / total if total else 0.0
        lines.append(f"    {c:>20s}: {dead:>4d}/{total:<4d} dead ({pct:.0f}%)")
    return "\n".join(lines)
