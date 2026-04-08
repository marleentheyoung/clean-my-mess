"""
test_smoke.py  --  Stage A

Smoke tests that run in < 30 s (after Numba compilation).
Verify imports, parameter structure, grid creation, and Tauchen properties.
"""

import importlib
import numpy as np
import pytest


# ---- Module import tests ----

# All modules that should import without nlopt
IMPORTABLE_MODULES = [
    "LoM_epsilons",
    "buyer_problem_epsilons",
    "buyer_problem_simulation",
    "continuation_value_nolearning",
    "equilibrium",
    "experiments",
    "grid_creation",
    "grids",
    "household_problem_epsilons_nolearning",
    "interp",
    "misc_functions",
    "moments",
    "mortgage_choice_simulation",
    "mortgage_choice_simulation_exc",
    "par_epsilons",
    "plot_creation",
    "proper_welfare_debug",
    "simulate_initial_joint",
    "simulation",
    "solve_epsilons",
    "stayer_problem",
    "stayer_problem_renter",
    "tauchen",
    "utility_epsilons",
]


class TestModuleImports:
    """Verify that all 24 modules import without error."""

    @pytest.mark.parametrize("module_name", IMPORTABLE_MODULES)
    def test_import(self, module_name, compiled_modules):
        """Each module should be importable (Numba compiles on first import)."""
        assert module_name in compiled_modules
        mod = compiled_modules[module_name]
        assert mod is not None


# ---- Parameter dict tests ----

class TestParDict:
    """Verify par_epsilons.par_dict has expected keys and sensible values."""

    def test_par_dict_has_expected_keys(self):
        import par as parfile
        expected_keys = [
            "dBeta", "dSigma", "iNj", "dPhi", "vPi_S_median",
            "iNb", "iBmax", "dRho", "dSigmaeps", "iNumStates",
            "iXin", "r", "r_m", "dDelta", "dPsi", "dKappa_sell",
            "max_ltv", "dGamma", "dNu", "dZeta",
        ]
        for key in expected_keys:
            assert key in parfile.par_dict, f"Missing key: {key}"

    def test_par_dict_scalar_types(self):
        import par as parfile
        d = parfile.par_dict
        # Scalars should be numeric (int or float)
        assert isinstance(d["dBeta"], float)
        assert isinstance(d["dSigma"], (int, float))
        assert isinstance(d["iNj"], int)
        assert isinstance(d["dPhi"], float)

    def test_par_dict_vPi_S_median_shape(self):
        import par as parfile
        vPi_S = parfile.par_dict["vPi_S_median"]
        assert isinstance(vPi_S, np.ndarray)
        # Should have > 50 elements (56 original values, transformed by
        # 1-(1-p)^time_increment at module level)
        assert vPi_S.shape[0] >= 50
        # All probabilities should be in (0, 1)
        assert np.all(vPi_S > 0)
        assert np.all(vPi_S < 1)

    def test_par_dict_iNj_value(self):
        import par as parfile
        assert parfile.par_dict["iNj"] == 30

    def test_par_dict_iNumStates_is_odd(self):
        import par as parfile
        assert parfile.par_dict["iNumStates"] % 2 == 1, "iNumStates must be odd"


# ---- Par jitclass tests ----

class TestParJitclass:
    """Verify the numba jitclass par object."""

    def test_par_has_key_attributes(self, par):
        assert hasattr(par, "dBeta")
        assert hasattr(par, "dSigma")
        assert hasattr(par, "iNj")
        assert hasattr(par, "dPhi")
        assert hasattr(par, "vPi_S_median")

    def test_par_iNj(self, par):
        assert par.iNj == 30

    def test_par_dBeta_positive(self, par):
        assert 0 < par.dBeta < 1


# ---- Grid creation tests ----

class TestGridCreation:
    """Verify grid_creation.create(par) returns grids with expected attributes."""

    def test_grids_has_expected_attributes(self, grids):
        expected_attrs = [
            "vM", "vH", "vL", "vE", "vTime", "vZ", "vPDF_z",
            "vX", "vB", "vG", "vK", "vChi", "vH_renter",
            "vX_sim", "vM_sim", "vL_sim", "vPi_S_median",
            "mPTI", "vTypes", "vPi_E", "vPi_L", "median_inc",
        ]
        for attr in expected_attrs:
            assert hasattr(grids, attr), f"grids missing attribute: {attr}"

    def test_grids_vM_nontrivial(self, grids):
        assert grids.vM.size > 0
        assert not np.all(grids.vM == 0)
        assert np.all(np.isfinite(grids.vM))

    def test_grids_vH_nontrivial(self, grids):
        assert grids.vH.size >= 3
        assert grids.vH[0] > 0
        assert np.all(np.diff(grids.vH) > 0), "vH should be strictly increasing"

    def test_grids_vE_nontrivial(self, grids):
        assert grids.vE.size > 0
        assert np.all(np.isfinite(grids.vE))

    def test_grids_vTime_nontrivial(self, grids):
        assert grids.vTime.size > 0

    def test_grids_vZ_structure(self, grids):
        # vZ[0] == 1 (no damage), rest < 1 (damage fractions)
        assert grids.vZ[0] == 1.0
        assert np.all(grids.vZ[1:] < 1.0)
        assert np.all(grids.vZ > 0)

    def test_grids_vPDF_z_structure(self, grids):
        # vPDF_z[0] == 1 (prob of no damage | no flood)
        # vPDF_z[1:] are conditional damage probs | flood, should sum to 1
        assert grids.vPDF_z[0] == 1.0
        assert np.isclose(np.sum(grids.vPDF_z[1:]), 1.0)

    def test_grids_vG_has_unit_center(self, grids):
        """The amenity preference grid should contain 1.0 at its center."""
        mid = (grids.vG.size - 1) // 2
        assert grids.vG[mid] == 1.0

    def test_grids_vK_has_two_types(self, grids):
        assert grids.vK.size == 2  # realist (0) and optimist (1)

    def test_grids_shapes_consistent_with_par(self, par, grids):
        assert grids.vM.size == par.iNb
        assert grids.vB.size == par.iNb
        assert grids.vX.size == par.iNb
        assert grids.vG.size == par.iXin
        assert grids.vE.size == par.iNumStates


# ---- Tauchen Markov matrix tests ----

class TestTauchenMarkov:
    """Verify the Tauchen Markov transition matrix."""

    def test_markov_rows_sum_to_one(self, mMarkov):
        row_sums = mMarkov.sum(axis=1)
        assert np.allclose(row_sums, 1.0), (
            f"Markov rows do not sum to 1. Max deviation: {np.max(np.abs(row_sums - 1.0))}"
        )

    def test_markov_nonnegative(self, mMarkov):
        assert np.all(mMarkov >= 0), "Markov matrix has negative entries"

    def test_markov_square(self, mMarkov):
        assert mMarkov.shape[0] == mMarkov.shape[1]

    def test_markov_size_matches_vE(self, mMarkov, grids):
        assert mMarkov.shape[0] == grids.vE.size

    def test_markov_no_absorbing_states(self, mMarkov):
        """No row should be all zeros except the diagonal."""
        for i in range(mMarkov.shape[0]):
            # At least two nonzero entries (the state can transition)
            assert np.sum(mMarkov[i, :] > 1e-15) >= 2, (
                f"Row {i} appears to be an absorbing state"
            )
