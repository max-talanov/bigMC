"""
Spinal locomotor CPG parameters: bilateral lumbar circuit with V0-V3
interneurons and motoneuron pools.

References:
  Kiehn, O. (2016). Decoding the organization of spinal circuits that control
    locomotion. Nature Reviews Neuroscience 17:224-238.
  Danner, S.M. et al. (2019). Spinal V3 interneurons and left-right
    coordination in mammalian locomotion. Frontiers in Cellular Neuroscience 13:516.
  Hagglund, M. et al. (2010). Activation of groups of excitatory neurons in
    the mammalian spinal cord generates rhythmic motor activity. Nature Neurosci.
  Ausborn, J. et al. (2019). Computational modeling of spinal locomotor
    circuitry in the age of molecular genetics. Int. J. Mol. Sci. 20:6835.

Population identity (per side, 8 populations):
  V0_D : commissural inhibitory (slow-walk left-right alternation)
  V0_V : commissural excitatory
  V1   : ipsilateral inhibitory (Renshaw + Ia inhibitory; flexor-extensor alt.)
  V2a  : ipsilateral excitatory premotor (rhythm generator core)
  V2b  : ipsilateral inhibitory (flexor-extensor alternation)
  V3   : commissural excitatory (bilateral synchrony at high speed)
  MN_F : alpha-motoneurons innervating flexor muscle group
  MN_E : alpha-motoneurons innervating extensor muscle group
"""
import numpy as np


# ════════════════════════════════════════════════════════════════════════════
#  Top-level controls
# ════════════════════════════════════════════════════════════════════════════

# Scale: 0.01 = laptop, 0.1 = workstation, 1.0 = MN5 production
NETWORK_SCALE = 0.05

# Bilateral
SIDES = ("L", "R")

# Descending (CST/M1) drive in Hz per side
DESCENDING_DRIVE_L_HZ = 4000.0
DESCENDING_DRIVE_R_HZ = 4000.0

# ── Graded stroke: CST drive asymmetry from M1 lesion ────────────────────
# When used standalone (without M1 microcircuit coupled), the spinal CPG
# receives a proxy drive reduction that mirrors the graded lesion:
#
#   STROKE_FRACTION: total fraction of L M1 L5e affected
#   Core    (60% of STROKE_FRACTION): contributes 0% drive (silenced)
#   Penumbra(40% of STROKE_FRACTION): contributes PENUMBRA_DRIVE_FRACTION
#                                     of normal (partially active)
#
# Example: stroke_fraction=0.5, core=0.3, penumbra=0.2
#   effective_reduction = 0.3*1.0 + 0.2*(1-0.5) = 0.3 + 0.1 = 0.4 (40%)
#
STROKE_FRACTION_L        = 0.0    # total lesion (0.0=healthy, 0.5=severe)
CORE_FRACTION            = 0.60   # proportion of lesion that is core
PENUMBRA_DRIVE_FRACTION  = 0.50   # penumbra neurons fire at 50% of normal

# Simulation
SIM_TIME_MS = 4000.0
DT_MS = 0.1
WARMUP_MS = 500.0   # discard for steady-state metrics

PRESERVE_INDEGREE_AT_SCALE = True


# ════════════════════════════════════════════════════════════════════════════
#  Population sizes (full scale, one side; rodent-extrapolated)
# ════════════════════════════════════════════════════════════════════════════

POP_NAMES = ["V0_D", "V0_V", "V1", "V2a", "V2b", "V3", "MN_F", "MN_E"]

N_FULL = {
    "V0_D": 2000,
    "V0_V": 2000,
    "V1":   5000,
    "V2a":  3000,
    "V2b":  2000,
    "V3":   2000,
    "MN_F":  600,    # ~ alpha-MN pool size for one muscle group
    "MN_E":  600,
}


def get_pop_sizes(scale=NETWORK_SCALE):
    return {p: max(1, int(round(N_FULL[p] * scale))) for p in POP_NAMES}


# ════════════════════════════════════════════════════════════════════════════
#  Connection probabilities (one side -> same side; "ipsi")
# ════════════════════════════════════════════════════════════════════════════
# Order: V0_D V0_V V1 V2a V2b V3 MN_F MN_E
# Probabilities derived from Ausborn 2019 + Danner 2019 circuit diagrams,
# adapted to dense (CPG-scale) connectivity (5-15%).

CONN_IPSI = np.array([
#   src:  V0_D  V0_V   V1   V2a   V2b   V3   MN_F MN_E
    [0.00, 0.00, 0.00, 0.08, 0.00, 0.00, 0.00, 0.00],   # V0_D <- only V2a (flexor-phase)
    [0.00, 0.00, 0.00, 0.10, 0.00, 0.00, 0.00, 0.00],   # V0_V <- V2a
    [0.00, 0.00, 0.05, 0.12, 0.00, 0.00, 0.00, 0.00],   # V1   <- V2a + recurrent
    [0.00, 0.00, 0.10, 0.15, 0.00, 0.00, 0.00, 0.00],   # V2a  <- V1 (inhibit) + recurrent
    [0.00, 0.00, 0.00, 0.10, 0.00, 0.00, 0.00, 0.00],   # V2b  <- V2a (flexor->extensor side)
    [0.00, 0.00, 0.00, 0.08, 0.00, 0.00, 0.00, 0.00],   # V3   <- V2a
    [0.00, 0.00, 0.10, 0.20, 0.00, 0.00, 0.00, 0.00],   # MN_F <- V2a (drive) + V1 (inh)
    [0.00, 0.00, 0.00, 0.00, 0.15, 0.00, 0.00, 0.00],   # MN_E <- V2b (inh of antagonist)
])
assert CONN_IPSI.shape == (8, 8)


# Cross-side (commissural)
CONN_CONTRA = np.array([
#   src:  V0_D  V0_V   V1   V2a   V2b   V3   MN_F MN_E
    [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
    [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
    [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
    [0.10, 0.00, 0.00, 0.00, 0.00, 0.08, 0.00, 0.00],   # V2a (contra) <- V0_D (inh), V3 (exc)
    [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
    [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
    [0.05, 0.08, 0.00, 0.00, 0.00, 0.06, 0.00, 0.00],   # MN_F (contra) <- V0 + V3
    [0.05, 0.08, 0.00, 0.00, 0.00, 0.06, 0.00, 0.00],   # MN_E (contra)
])
assert CONN_CONTRA.shape == (8, 8)


# Excitatory / inhibitory identity
EXCITATORY = {"V0_V", "V2a", "V3"}        # excite their targets
INHIBITORY = {"V0_D", "V1", "V2b"}        # inhibit
MOTONEURONS = {"MN_F", "MN_E"}             # excite muscle (output)


def effective_drive_reduction(stroke_fraction=None,
                               core_frac=None, penumbra_drive=None):
    """
    Compute the effective fractional reduction in descending drive from
    a graded lesion (core fully silent + penumbra partially active).

    Parameters
    ----------
    stroke_fraction : float  total fraction of L5e affected (default STROKE_FRACTION_L)
    core_frac       : float  fraction of lesion that is core (default CORE_FRACTION)
    penumbra_drive  : float  penumbra firing as fraction of normal (default PENUMBRA_DRIVE_FRACTION)

    Returns
    -------
    float   effective drive reduction (0.0 = healthy, → 1.0 = all silent)

    Example
    -------
    >>> effective_drive_reduction(0.5)   # moderate stroke
    0.40   # 30% from core + 10% from penumbra
    """
    if stroke_fraction is None: stroke_fraction = STROKE_FRACTION_L
    if core_frac       is None: core_frac       = CORE_FRACTION
    if penumbra_drive  is None: penumbra_drive  = PENUMBRA_DRIVE_FRACTION

    f_core     = stroke_fraction * core_frac
    f_penumbra = stroke_fraction * (1.0 - core_frac)
    reduction  = f_core * 1.0 + f_penumbra * (1.0 - penumbra_drive)
    return float(reduction)


# Backward-compat alias (old --stroke flag)
STROKE_DRIVE_REDUCTION_L = effective_drive_reduction()


def is_excitatory(pop):
    return pop in EXCITATORY or pop in MOTONEURONS


def population_index(name):
    return POP_NAMES.index(name)


def pop_full_name(side, pop):
    """E.g. ('L', 'V2a') -> 'L_V2a'."""
    return f"{side}_{pop}"


# ════════════════════════════════════════════════════════════════════════════
#  Neuron model (iaf_psc_exp, simple but bio-plausible)
# ════════════════════════════════════════════════════════════════════════════

NEURON_MODEL = "iaf_psc_exp"

NEURON_PARAMS = {
    "E_L":       -65.0,
    "V_th":      -50.0,
    "V_reset":   -65.0,
    "C_m":       250.0,
    "tau_m":      15.0,    # spinal INs slightly slower than cortical
    "t_ref":       2.0,
    "tau_syn_ex":  1.0,
    "tau_syn_in":  2.0,
}

# Motoneurons: lower threshold, longer tau (need to spike easily on synaptic input)
MN_PARAMS = dict(NEURON_PARAMS)
MN_PARAMS.update({
    "tau_m":      20.0,
    "tau_syn_ex":  1.5,
})


# ════════════════════════════════════════════════════════════════════════════
#  Synapse parameters
# ════════════════════════════════════════════════════════════════════════════

PSP_E       = 0.5          # mV; stronger than cortex (CPG INs need to drive rhythm)
PSP_REL_SD  = 0.10
G_INH_REL_E = -2.5         # E/I balance similar to Potjans-Diesmann cortex

DELAY_E_MS = 1.0
DELAY_I_MS = 1.0
DELAY_REL_SD = 0.40

# Commissural axons cross the midline; somewhat longer delay
DELAY_CONTRA_MS = 2.5


# ════════════════════════════════════════════════════════════════════════════
#  External input
# ════════════════════════════════════════════════════════════════════════════
# Tonic background to all interneurons (representing diffuse afferent
# excitation / brainstem reticulospinal drive).

BG_RATE_HZ = 1.5       # Hz per external synapse (tuned for 5-10 Hz baseline)
K_EXT = {
    "V0_D": 500,
    "V0_V": 500,
    "V1":   500,
    "V2a":  300,         # lower -- V2a primarily driven by descending input
    "V2b":  500,
    "V3":   400,
    "MN_F": 200,
    "MN_E": 200,
}

# Descending (CST/M1) drive: targets V2a specifically
# Modeled as a separate Poisson stream with stroke-controllable rate
DESCENDING_TARGET = "V2a"


if __name__ == "__main__":
    sizes = get_pop_sizes()
    print(f"NETWORK_SCALE = {NETWORK_SCALE}")
    total_per_side = sum(sizes.values())
    print(f"Per side : {total_per_side:,} neurons")
    print(f"Bilateral: {total_per_side * 2:,} neurons")
    for p in POP_NAMES:
        n = sizes[p]
        eih = "E" if is_excitatory(p) else "I"
        if p in MOTONEURONS:
            eih = "MN"
        print(f"  {p:>5s} ({eih:>2s}) : {n:>5,} per side")
