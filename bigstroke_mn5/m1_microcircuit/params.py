"""
Potjans-Diesmann 2014 cortical microcircuit parameters, adapted for
bilateral M1 with stroke-zone support.

Reference:
  Potjans, T.C. & Diesmann, M. (2014). The cell-type specific cortical
  microcircuit. Cerebral Cortex 24:785–806.
"""
import numpy as np

# ════════════════════════════════════════════════════════════════════════════
#  Top-level controls
# ════════════════════════════════════════════════════════════════════════════

# Network size relative to full Potjans-Diesmann column (per hemisphere).
#   0.01 = laptop quick test  (~770 neurons per hemisphere)
#   0.1  = workstation        (~7,700 per hemi)
#   1.0  = MN5 production     (~77,000 per hemi)
NETWORK_SCALE = 0.01

# Bilateral M1 (left + right hemispheres)
HEMISPHERES = ("L", "R")

# Preserve mean in-degree at full-scale value when NETWORK_SCALE < 1.0.
# When True, recurrent indegrees are kept at full-scale K via multapses, so
# spontaneous firing rates match Potjans-Diesmann 2014 targets at any scale
# (at the cost of higher pairwise correlation at small scales).  See PD2014
# Appendix D and the PyNEST microcircuit reference implementation.
PRESERVE_INDEGREE_AT_SCALE = True

# ── Stroke zone: graded lesion with ischemic core + penumbra ──────────────
#
# Based on Hofmeijer & van Putten (2012) Stroke 43:607–615
# and Lo et al. (2003) Nature Reviews Neuroscience 4:399–415
#
# Total lesion fraction of L5e neurons in the LEFT hemisphere:
#   0.0  = healthy
#   0.3  = moderate stroke (30% of L5e affected)
#   0.5  = severe stroke
#
# Within the lesion, 60% is ischemic core, 40% penumbra (typical ratio).
# Core:     neurons completely silenced (V_th → ∞)
# Penumbra: neurons partially impaired:
#             E_L raised +5 mV (ATP-dependent K+ pump failure)
#             V_th raised +8 mV (lower firing probability)
#             tau_m halved (leak conductance ×2 from membrane damage)
# Diaschisis hook: callosal L5e→L5e projections are naturally reduced once
#   core neurons stop spiking, but contralateral hyper-excitability is NOT
#   yet modelled explicitly (planned Stage 4 extension).

STROKE_FRACTION_L5e_LEFT = 0.0   # default: healthy

STROKE_CORE_FRACTION     = 0.60  # 60% of lesion = ischemic core
STROKE_PENUMBRA_FRACTION = 0.40  # 40% of lesion = penumbra ring

# Penumbra biophysical deltas (from resting healthy state)
PENUMBRA_E_L_SHIFT_MV   = +5.0  # depolarization (ATP pump failure)
PENUMBRA_V_TH_SHIFT_MV  = +8.0  # reduced reliability (membrane damage)
PENUMBRA_TAU_M_FACTOR   =  0.5  # tau_m halved → leak conductance ×2

# ── Progressive degeneration (Stage A: time-phased cell death) ────────────
#
# Multi-timescale model: fast spiking (ms) + slow degeneration (hours).
# A spatial infarct affects ALL left-hemisphere populations (vascular
# territory, not cell-type ablation).  Within the penumbra, neurons die on
# a cell-type-specific schedule reflecting selective neuronal vulnerability:
#
#   Excitatory pyramidal    → high vulnerability (NMDA/AMPA excitotoxicity)
#   PV+ fast-spiking IN      → high vulnerability (high metabolic demand)
#   Buffered IN (CB+/CR+)    → resistant (Ca²⁺ buffering) → may recover
#
# References: Hofmeijer & van Putten 2012; Lo et al. 2003;
#             Kreuzberg et al. 2010 (PV+ vulnerability);
#             Povysheva et al. 2019 (interneuron Ca²⁺ buffering).

# Fraction of each inhibitory population that is PV-like (vulnerable).
# Cortical PV+ cells are ~40% of GABAergic neurons (Rudy et al. 2011).
PV_FRACTION = 0.40

# Penumbra → core conversion time (biological hours) per vulnerability class.
# After this long in penumbra, the neuron is silenced (dies).  inf = survives.
DEATH_TIME_HOURS = {
    "excitatory":          6.0,            # pyramidal: fast excitotoxic death
    "pv_inhibitory":       9.0,            # PV+ fast-spiker: vulnerable, slower
    "buffered_inhibitory": float("inf"),   # calbindin/calretinin: survive
}

# Buffered penumbra interneurons recover toward healthy params after this long.
RECOVERY_TIME_HOURS = 12.0

# Spread in per-neuron death times (relative SD) — biological variability.
DEATH_TIME_REL_SD = 0.25

# Degeneration-run defaults (operator splitting across timescales)
DEGEN_EPOCHS          = 24      # number of slow steps (→ 24 biological hours)
DEGEN_HOURS_PER_EPOCH = 1.0     # biological hours advanced per epoch
DEGEN_EPOCH_SIM_MS    = 400.0   # ms of spiking simulated per epoch (activity sample)
DEGEN_LESION_FRACTION = 0.40    # spatial fraction of L hemisphere in the infarct

# ── Simulation ────────────────────────────────────────────────────────────
SIM_TIME_MS = 1000.0       # total simulation time
DT_MS       = 0.1          # integration timestep
WARMUP_MS   = 100.0        # discard this much initial transient

# ════════════════════════════════════════════════════════════════════════════
#  Population sizes (Potjans-Diesmann 2014, Table 5, full scale)
# ════════════════════════════════════════════════════════════════════════════

POP_NAMES = ["L23e", "L23i", "L4e", "L4i", "L5e", "L5i", "L6e", "L6i"]

# Full-scale neuron counts (one column)
N_FULL = {
    "L23e": 20683,
    "L23i":  5834,
    "L4e":  21915,
    "L4i":   5479,
    "L5e":   4850,
    "L5i":   1065,
    "L6e":  14395,
    "L6i":   2948,
}


def get_pop_sizes(scale=NETWORK_SCALE):
    """Return scaled population sizes (int)."""
    return {p: max(1, int(round(N_FULL[p] * scale))) for p in POP_NAMES}


# ════════════════════════════════════════════════════════════════════════════
#  Connection probabilities (Potjans-Diesmann 2014, Table 5)
# ════════════════════════════════════════════════════════════════════════════
# rows = target, cols = source; order = L23e L23i L4e L4i L5e L5i L6e L6i

CONN_PROB = np.array([
    # source:   L23e   L23i   L4e    L4i    L5e    L5i    L6e    L6i
    [0.1009, 0.1689, 0.0437, 0.0818, 0.0323, 0.0000, 0.0076, 0.0000],  # L23e (tgt)
    [0.1346, 0.1371, 0.0316, 0.0515, 0.0755, 0.0000, 0.0042, 0.0000],  # L23i
    [0.0077, 0.0059, 0.0497, 0.1350, 0.0067, 0.0003, 0.0453, 0.0000],  # L4e
    [0.0691, 0.0029, 0.0794, 0.1597, 0.0033, 0.0000, 0.1057, 0.0000],  # L4i
    [0.1004, 0.0622, 0.0505, 0.0057, 0.0831, 0.3726, 0.0204, 0.0000],  # L5e
    [0.0548, 0.0269, 0.0257, 0.0022, 0.0600, 0.3158, 0.0086, 0.0000],  # L5i
    [0.0156, 0.0066, 0.0211, 0.0166, 0.0572, 0.0197, 0.0396, 0.2252],  # L6e
    [0.0364, 0.0010, 0.0034, 0.0005, 0.0277, 0.0080, 0.0658, 0.1443],  # L6i
])

assert CONN_PROB.shape == (8, 8)


# ════════════════════════════════════════════════════════════════════════════
#  Neuron model parameters (iaf_psc_exp, Potjans-Diesmann)
# ════════════════════════════════════════════════════════════════════════════

NEURON_MODEL = "iaf_psc_exp"

NEURON_PARAMS = {
    "E_L":       -65.0,    # mV, resting potential
    "V_th":      -50.0,    # mV, spike threshold
    "V_reset":   -65.0,    # mV, reset potential
    "C_m":       250.0,    # pF, membrane capacitance
    "tau_m":      10.0,    # ms, membrane time constant
    "t_ref":       2.0,    # ms, refractory period
    "tau_syn_ex":  0.5,    # ms, excitatory synaptic time constant
    "tau_syn_in":  0.5,    # ms, inhibitory synaptic time constant
}

# ════════════════════════════════════════════════════════════════════════════
#  Synapse parameters
# ════════════════════════════════════════════════════════════════════════════

PSP_E       = 0.15         # mV, mean excitatory PSP amplitude
PSP_REL_SD  = 0.10         # relative SD of PSP amplitudes
G_INH_REL_E = -4.0         # inhibitory weight relative to excitatory (g = -4)

# Synaptic delays (mean ± SD) — uniform across populations for simplicity
DELAY_E_MS  = 1.5
DELAY_I_MS  = 0.75
DELAY_REL_SD = 0.50

# ════════════════════════════════════════════════════════════════════════════
#  Background input
# ════════════════════════════════════════════════════════════════════════════
# Each population receives a Poisson background of rate ν_bg × K_ext, where
# K_ext is population-specific (number of external synapses per neuron).

BG_RATE_HZ = 8.0  # background rate per external synapse

K_EXT = {
    # source:   external synapses per neuron
    "L23e": 1600, "L23i": 1500,
    "L4e":  2100, "L4i":  1900,
    "L5e":  2000, "L5i":  1900,
    "L6e":  2900, "L6i":  2100,
}

# ════════════════════════════════════════════════════════════════════════════
#  Cross-hemisphere callosal projections (simplified)
# ════════════════════════════════════════════════════════════════════════════
# Callosal fibers originate primarily from L2/3e and L5e of one hemisphere
# and target homotopic L2/3e and L5e of the contralateral side.
# Probability is reduced relative to intra-hemispheric (≈10%).

CALLOSAL_PROB_SCALE = 0.10   # relative to ipsilateral connection prob
CALLOSAL_SOURCES    = ("L23e", "L5e")
CALLOSAL_TARGETS    = ("L23e", "L5e")
CALLOSAL_DELAY_MS   = 8.0    # interhemispheric delay (long-range axons)


# ════════════════════════════════════════════════════════════════════════════
#  Convenience getters
# ════════════════════════════════════════════════════════════════════════════

def population_index(name):
    """Return integer index in CONN_PROB matrix."""
    return POP_NAMES.index(name)


def is_excitatory(name):
    return name.endswith("e")


def pop_full_name(hemi, pop):
    """E.g. ('L', 'L23e') -> 'L_L23e'."""
    return f"{hemi}_{pop}"


def vulnerability_of(pop, is_pv=False):
    """
    Return the vulnerability class for a neuron.

    Parameters
    ----------
    pop   : str   population name (e.g. 'L5e', 'L4i')
    is_pv : bool  for inhibitory neurons, whether this one is PV-like

    Returns
    -------
    str   one of 'excitatory', 'pv_inhibitory', 'buffered_inhibitory'
    """
    if is_excitatory(pop):
        return "excitatory"
    return "pv_inhibitory" if is_pv else "buffered_inhibitory"


if __name__ == "__main__":
    # Quick sanity check
    sizes = get_pop_sizes()
    total_per_hemi = sum(sizes.values())
    total_bilateral = total_per_hemi * len(HEMISPHERES)
    print(f"NETWORK_SCALE = {NETWORK_SCALE}")
    print(f"Per hemisphere: {total_per_hemi:,} neurons")
    print(f"Bilateral total: {total_bilateral:,} neurons")
    for p in POP_NAMES:
        print(f"  {p:>5s} : {sizes[p]:>6,} per hemi")
