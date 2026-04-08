"""Stage C: Regression tests.

Fast tests pin grid creation and solve_ss on full grids.
The slow find_expenditure_equiv test is marked @pytest.mark.slow — run manually.
"""
import numpy as np
import os
import pytest

SNAPSHOTS_DIR = os.path.join(os.path.dirname(__file__), "snapshots")


def test_grid_creation_deterministic(grids_and_markov_full):
    """Grid creation is deterministic — same inputs always produce same grids."""
    grids, mMarkov = grids_and_markov_full

    snapshot_path = os.path.join(SNAPSHOTS_DIR, "grid_creation.npz")

    grid_data = {
        "vM": np.asarray(grids.vM), "vH": np.asarray(grids.vH),
        "vL": np.asarray(grids.vL), "vE": np.asarray(grids.vE),
        "vX": np.asarray(grids.vX), "vZ": np.asarray(grids.vZ),
        "vPDF_z": np.asarray(grids.vPDF_z), "mMarkov": mMarkov,
    }

    if not os.path.exists(snapshot_path):
        np.savez(snapshot_path, **grid_data)
        return

    ref = np.load(snapshot_path)
    for name, arr in grid_data.items():
        assert np.allclose(arr, ref[name], atol=1e-10, rtol=1e-10), \
            f"Grid {name} drifted from snapshot"


def test_solve_ss_full_grid_finite(par_full, grids_and_markov_full):
    """solve_ss on full grids produces finite value functions."""
    import household_problem_epsilons_nolearning as hp

    grids, mMarkov = grids_and_markov_full
    dCoeff_C = 0.69906474
    dCoeff_NC = 0.78259554

    result = hp.solve_ss(grids, par_full, par_full.iNj, mMarkov,
                         dCoeff_C, dCoeff_NC, initial=True, sceptics=True, welfare=False)

    vt_stay_c, vt_stay_nc, vt_renter, b_stay_c, b_stay_nc, b_renter = result

    for name, arr in [("vt_stay_c", vt_stay_c), ("vt_stay_nc", vt_stay_nc),
                      ("vt_renter", vt_renter)]:
        assert np.all(np.isfinite(arr)), f"{name} contains NaN or Inf on full grids"
        assert np.any(arr != 0), f"{name} is all zeros on full grids"


def test_solve_ss_full_grid_snapshot(par_full, grids_and_markov_full):
    """Pin solve_ss full-grid output against saved snapshot."""
    import household_problem_epsilons_nolearning as hp

    grids, mMarkov = grids_and_markov_full
    dCoeff_C = 0.69906474
    dCoeff_NC = 0.78259554

    result = hp.solve_ss(grids, par_full, par_full.iNj, mMarkov,
                         dCoeff_C, dCoeff_NC, initial=True, sceptics=True, welfare=False)

    vt_stay_c, vt_stay_nc, vt_renter, b_stay_c, b_stay_nc, b_renter = result

    snapshot_path = os.path.join(SNAPSHOTS_DIR, "solve_ss_full.npz")

    if not os.path.exists(snapshot_path):
        np.savez(snapshot_path,
                 vt_stay_c=vt_stay_c, vt_stay_nc=vt_stay_nc, vt_renter=vt_renter,
                 b_stay_c=b_stay_c, b_stay_nc=b_stay_nc, b_renter=b_renter)
        return

    ref = np.load(snapshot_path)
    for name in ["vt_stay_c", "vt_stay_nc", "vt_renter", "b_stay_c", "b_stay_nc", "b_renter"]:
        current = [vt_stay_c, vt_stay_nc, vt_renter, b_stay_c, b_stay_nc, b_renter][
            ["vt_stay_c", "vt_stay_nc", "vt_renter", "b_stay_c", "b_stay_nc", "b_renter"].index(name)]
        assert np.allclose(current, ref[name], atol=1e-10, rtol=1e-10), \
            f"{name} drifted from snapshot on full grids"


@pytest.mark.slow
def test_find_expenditure_equiv(par_full, grids_and_markov_full):
    """Pin welfare equivalents from the live entry point. Takes 10-15 min.

    Run manually with: pytest tests/test_regression.py -m slow
    """
    import welfare as welfare_stats
    from conftest import VCOEFF_C_INITIAL, VCOEFF_NC_INITIAL, VCOEFF_C, VCOEFF_NC

    grids, mMarkov = grids_and_markov_full

    tax_equiv_C, tax_equiv_NC, tax_equiv_renter, tax_equiv_newborns = \
        welfare_stats.find_expenditure_equiv(
            par_full, grids, mMarkov,
            VCOEFF_C_INITIAL, VCOEFF_NC_INITIAL, VCOEFF_C, VCOEFF_NC)

    snapshot_path = os.path.join(SNAPSHOTS_DIR, "welfare_equivalents.npz")

    results = {
        "tax_equiv_C": np.array(tax_equiv_C),
        "tax_equiv_NC": np.array(tax_equiv_NC),
        "tax_equiv_renter": np.array(tax_equiv_renter),
        "tax_equiv_newborns": np.array(tax_equiv_newborns),
    }

    if not os.path.exists(snapshot_path):
        np.savez(snapshot_path, **results)
        return

    ref = np.load(snapshot_path)
    for name, val in results.items():
        assert np.allclose(val, ref[name], rtol=1e-6), \
            f"{name} drifted: got {val}, expected {ref[name]}"
