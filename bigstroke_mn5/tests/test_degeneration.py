"""
Tests for the degeneration params + vulnerability logic (NEST not required).
"""
import numpy as np

from bigstroke_mn5.m1_microcircuit import params as P


def test_pv_fraction_bounds():
    assert 0.0 <= P.PV_FRACTION <= 1.0


def test_death_time_classes_complete():
    """Every vulnerability class has a death time defined."""
    for c in ("excitatory", "pv_inhibitory", "buffered_inhibitory"):
        assert c in P.DEATH_TIME_HOURS


def test_buffered_survives():
    """Buffered interneurons must have infinite death time (survive)."""
    assert np.isinf(P.DEATH_TIME_HOURS["buffered_inhibitory"])


def test_vulnerable_classes_finite():
    """Excitatory and PV+ must have finite death times."""
    assert np.isfinite(P.DEATH_TIME_HOURS["excitatory"])
    assert np.isfinite(P.DEATH_TIME_HOURS["pv_inhibitory"])


def test_excitatory_dies_first():
    """
    Literature: pyramidal excitotoxic death is faster than PV+ metabolic death.
    """
    assert P.DEATH_TIME_HOURS["excitatory"] <= P.DEATH_TIME_HOURS["pv_inhibitory"]


def test_vulnerability_of_excitatory():
    assert P.vulnerability_of("L5e") == "excitatory"
    assert P.vulnerability_of("L23e") == "excitatory"


def test_vulnerability_of_inhibitory_pv():
    assert P.vulnerability_of("L5i", is_pv=True) == "pv_inhibitory"
    assert P.vulnerability_of("L5i", is_pv=False) == "buffered_inhibitory"


def test_degen_defaults_sane():
    assert P.DEGEN_EPOCHS > 0
    assert P.DEGEN_HOURS_PER_EPOCH > 0
    assert P.DEGEN_EPOCH_SIM_MS > 0
    assert 0 < P.DEGEN_LESION_FRACTION <= 1.0


def test_recovery_after_some_death():
    """
    Buffered recovery time should be later than the fastest death,
    so we see death happen before recovery in a typical run.
    """
    assert P.RECOVERY_TIME_HOURS > P.DEATH_TIME_HOURS["excitatory"]
