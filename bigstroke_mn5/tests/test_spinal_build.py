"""Smoke tests for the spinal CPG params (NEST not required)."""
import numpy as np

from bigstroke_mn5.spinal_cpg import params as P


def test_pop_names_count():
    assert len(P.POP_NAMES) == 8


def test_conn_matrices_shape():
    assert P.CONN_IPSI.shape == (8, 8)
    assert P.CONN_CONTRA.shape == (8, 8)


def test_conn_matrices_bounds():
    for M in (P.CONN_IPSI, P.CONN_CONTRA):
        assert (M >= 0).all()
        assert (M <= 1).all()


def test_population_sizes_scale():
    for scale in (0.001, 0.05, 0.1, 1.0):
        sizes = P.get_pop_sizes(scale)
        assert len(sizes) == 8
        for p in P.POP_NAMES:
            assert sizes[p] >= 1


def test_excitatory_inhibitory_complete():
    for p in P.POP_NAMES:
        in_E = p in P.EXCITATORY
        in_I = p in P.INHIBITORY
        in_MN = p in P.MOTONEURONS
        # each population belongs to exactly one category
        assert sum([in_E, in_I, in_MN]) == 1, f"{p} classification ambiguous"


def test_is_excitatory():
    assert P.is_excitatory("V2a")   # excitatory IN
    assert P.is_excitatory("MN_F")  # MN excites muscle
    assert not P.is_excitatory("V1")  # inhibitory IN
    assert not P.is_excitatory("V0_D")  # inhibitory commissural


def test_pop_full_name():
    assert P.pop_full_name("L", "V2a") == "L_V2a"
    assert P.pop_full_name("R", "MN_F") == "R_MN_F"


def test_kext_complete():
    for p in P.POP_NAMES:
        assert p in P.K_EXT
        assert P.K_EXT[p] > 0


def test_default_simulation_params_sane():
    assert P.DT_MS > 0 and P.DT_MS < 1.0
    assert P.SIM_TIME_MS > P.WARMUP_MS
    assert 0 <= P.STROKE_DRIVE_REDUCTION_L <= 1


def test_descending_target_valid():
    assert P.DESCENDING_TARGET in P.POP_NAMES
