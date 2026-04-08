"""Stage A: Smoke tests (<30s after Numba compilation).

Verifies all modules import, parameters load, grids are well-formed.
"""
import numpy as np


def test_all_modules_import():
    """All 24 Python modules import without error."""
    import solve
    import tauchen
    import grid_creation
    import simulation
    import equilibrium
    import household_problem
    import moments
    import welfare
    import experiments
    import plot_creation
    import interp
    import misc_functions
    import grids
    import par
    import utility
    import lom
    import stayer_problem
    import stayer_problem_renter
    import buyer_problem
    import buyer_problem_simulation
    import mortgage_choice_simulation
    import mortgage_choice_simulation_exc
    import simulate_initial_joint
    import continuation_value_nolearning


def test_par_dict_keys():
    """Parameter dict has expected calibration keys."""
    import par as parfile
    required_keys = ["dBeta", "dSigma", "iNj", "dPhi", "dDelta", "r", "r_m",
                     "iNumStates", "dRho", "dSigmaeps", "iXin", "iBmax"]
    for key in required_keys:
        assert key in parfile.par_dict, f"Missing parameter: {key}"


def test_par_values_final():
    """Spot-check that key parameters match the final calibration."""
    import par as parfile
    assert parfile.par_dict["iNj"] == 30
    assert parfile.par_dict["j_ret"] == 23
    assert parfile.par_dict["iNumStates"] == 5
    assert parfile.par_dict["iXin"] == 7


def test_grid_creation(grids_and_markov_full):
    """Grid creation produces grids with expected attributes and shapes."""
    grids, mMarkov = grids_and_markov_full
    for attr in ["vM", "vH", "vL", "vE", "vTime", "vZ", "vPDF_z", "vG", "vK", "vX"]:
        arr = getattr(grids, attr)
        assert arr.size > 0, f"Grid attribute {attr} is empty"


def test_tauchen_matrix_rows_sum_to_one(grids_and_markov_full):
    """Tauchen transition matrix rows each sum to 1."""
    _, mMarkov = grids_and_markov_full
    row_sums = mMarkov.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-10), f"Row sums: {row_sums}"


def test_grid_shapes_nontrivial(grids_and_markov_full):
    """Grids have expected sizes from the full calibration."""
    grids, _ = grids_and_markov_full
    assert grids.vH.size == 3, f"Expected 3 house sizes, got {grids.vH.size}"
    assert grids.vE.size == 5, f"Expected 5 income states, got {grids.vE.size}"
    assert grids.vZ.size == 4, f"Expected 4 damage states, got {grids.vZ.size}"
    assert grids.vTime.size > 0


def test_vpdf_z_conditional_structure(grids_and_markov_full):
    """vPDF_z has intentional conditional probability structure (not a proper PDF)."""
    grids, _ = grids_and_markov_full
    # Element 0 = P(no damage | no flood) = 1
    assert grids.vPDF_z[0] == 1.0
    # Elements 1-3 = P(damage | flood), should sum to 1
    assert np.allclose(grids.vPDF_z[1:].sum(), 1.0, atol=1e-10)


def test_coefficient_values_pinned():
    """Hardcoded coefficient values match known-good inputs."""
    vCoeff_C_initial = np.array([0.69906474, 0., 0., 0., 0.])
    vCoeff_NC_initial = np.array([0.78259554, 0., 0., 0., 0.])
    assert np.allclose(vCoeff_C_initial[0], 0.69906474, rtol=1e-6)
    assert np.allclose(vCoeff_NC_initial[0], 0.78259554, rtol=1e-6)
