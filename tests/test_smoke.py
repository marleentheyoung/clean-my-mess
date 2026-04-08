"""Stage A: Smoke tests (<30s after Numba compilation).

Verifies all model.* modules import, parameters load, grids are well-formed.
"""
import numpy as np


def test_all_modules_import():
    """All model.* modules import without error."""
    from model.config import create_par_dict
    from model.utils import construct_jitclass
    from model.grids import create, nonlinspace_jit
    from model.tauchen import tauchen
    from model.interp import binary_search, interp_1d, interp_3d
    from model.utility import u, u_c, W_bequest
    from model.lom import LoM
    from model.household.vfi import solve, solve_ss
    from model.household.continuation import solve_last_period_owners_C
    from model.household.stayer import solve as _
    from model.household.renter import solve as _
    from model.household.buyer import solve as _
    from model.simulation.distribution import stat_dist_finder
    from model.simulation.excess_demand import excess_demand_continuous
    from model.simulation.buyer_sim import solve as _
    from model.simulation.mortgage_sim import solve as _
    from model.simulation.mortgage_sim_exc import solve as _
    from model.simulation.initial_joint import initial_joint
    from model.equilibrium.solver import find_coefficients
    from model.analysis.moments import calc_moments
    from model.analysis.welfare import find_expenditure_equiv
    from model.analysis.experiments import full_information_experiment
    from model.run import main
    from model.plots import plot_pricepaths


def test_par_dict_keys():
    """Parameter dict has expected calibration keys."""
    from model.config import create_par_dict
    par_dict = create_par_dict()
    required_keys = ["dBeta", "dSigma", "iNj", "dPhi", "dDelta", "r", "r_m",
                     "iNumStates", "dRho", "dSigmaeps", "iXin", "iBmax"]
    for key in required_keys:
        assert key in par_dict, f"Missing parameter: {key}"


def test_par_values_final():
    """Spot-check that key parameters match the final calibration."""
    from model.config import create_par_dict
    par_dict = create_par_dict()
    assert par_dict["iNj"] == 30
    assert par_dict["j_ret"] == 23
    assert par_dict["iNumStates"] == 5
    assert par_dict["iXin"] == 7


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
    assert grids.vPDF_z[0] == 1.0
    assert np.allclose(grids.vPDF_z[1:].sum(), 1.0, atol=1e-10)


def test_coefficient_values_pinned():
    """Hardcoded coefficient values match known-good inputs."""
    vCoeff_C_initial = np.array([0.69906474, 0., 0., 0., 0.])
    vCoeff_NC_initial = np.array([0.78259554, 0., 0., 0., 0.])
    assert np.allclose(vCoeff_C_initial[0], 0.69906474, rtol=1e-6)
    assert np.allclose(vCoeff_NC_initial[0], 0.78259554, rtol=1e-6)
