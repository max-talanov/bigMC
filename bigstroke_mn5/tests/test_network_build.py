"""
Smoke tests that do NOT require NEST to pass.
They exercise the parameter logic only.

Run with:  python -m pytest bigstroke_mn5/tests
"""
import numpy as np

from bigstroke_mn5.m1_microcircuit import params as P


def test_pop_names_count():
    assert len(P.POP_NAMES) == 8


def test_conn_prob_shape():
    assert P.CONN_PROB.shape == (8, 8)


def test_conn_prob_within_bounds():
    assert (P.CONN_PROB >= 0).all()
    assert (P.CONN_PROB <= 1).all()


def test_population_sizes_scale():
    for scale in (0.001, 0.01, 0.1, 1.0):
        sizes = P.get_pop_sizes(scale)
        assert len(sizes) == 8
        for p in P.POP_NAMES:
            assert sizes[p] >= 1


def test_is_excitatory():
    for p in P.POP_NAMES:
        assert P.is_excitatory(p) == p.endswith("e")


def test_pop_full_name():
    assert P.pop_full_name("L", "L23e") == "L_L23e"
    assert P.pop_full_name("R", "L5e") == "R_L5e"


def test_kext_complete():
    """Every population has a defined external-input count."""
    for p in P.POP_NAMES:
        assert p in P.K_EXT, f"K_EXT missing {p}"
        assert P.K_EXT[p] > 0


def test_default_simulation_params_sane():
    assert P.DT_MS > 0 and P.DT_MS < 1.0
    assert P.SIM_TIME_MS >= 100
    assert 0 <= P.STROKE_FRACTION_L5e_LEFT <= 1
