"""Stage B: VFI and distribution checks (~5 min on reduced grids).

Tests solve_ss convergence and output properties on reduced grids.
"""
import numpy as np
import os

SNAPSHOTS_DIR = os.path.join(os.path.dirname(__file__), "snapshots")


def test_solve_ss_returns_finite(par_reduced, grids_and_markov_reduced):
    """solve_ss produces finite value functions (no NaN/Inf) on reduced grids."""
    import household_problem as hp

    grids, mMarkov = grids_and_markov_reduced
    dCoeff_C = 0.69906474  # vCoeff_C_initial[0]
    dCoeff_NC = 0.78259554  # vCoeff_NC_initial[0]

    result = hp.solve_ss(grids, par_reduced, par_reduced.iNj, mMarkov,
                         dCoeff_C, dCoeff_NC, initial=True, sceptics=True, welfare=False)

    vt_stay_c, vt_stay_nc, vt_renter, b_stay_c, b_stay_nc, b_renter = result

    for name, arr in [("vt_stay_c", vt_stay_c), ("vt_stay_nc", vt_stay_nc),
                      ("vt_renter", vt_renter), ("b_stay_c", b_stay_c),
                      ("b_stay_nc", b_stay_nc), ("b_renter", b_renter)]:
        assert np.all(np.isfinite(arr)), f"{name} contains NaN or Inf"


def test_solve_ss_output_shapes(par_reduced, grids_and_markov_reduced):
    """solve_ss output arrays have expected dimensionality."""
    import household_problem as hp

    grids, mMarkov = grids_and_markov_reduced
    dCoeff_C = 0.69906474
    dCoeff_NC = 0.78259554

    result = hp.solve_ss(grids, par_reduced, par_reduced.iNj, mMarkov,
                         dCoeff_C, dCoeff_NC, initial=True, sceptics=True, welfare=False)

    vt_stay_c, vt_stay_nc, vt_renter, b_stay_c, b_stay_nc, b_renter = result

    # Stay arrays: (1, iNj, k_dim, G, M, H, L, E)
    assert vt_stay_c.ndim == 8, f"Expected 8 dims, got {vt_stay_c.ndim}"
    assert vt_stay_c.shape[0] == 1  # time dim
    assert vt_stay_c.shape[1] == par_reduced.iNj  # age dim

    # Renter arrays: (1, iNj, k_dim, G, X, E)
    assert vt_renter.ndim == 6, f"Expected 6 dims, got {vt_renter.ndim}"
    assert vt_renter.shape[1] == par_reduced.iNj


def test_solve_ss_value_functions_nonzero(par_reduced, grids_and_markov_reduced):
    """Value functions are not trivially zero (solve_ss actually computed something)."""
    import household_problem as hp

    grids, mMarkov = grids_and_markov_reduced
    dCoeff_C = 0.69906474
    dCoeff_NC = 0.78259554

    result = hp.solve_ss(grids, par_reduced, par_reduced.iNj, mMarkov,
                         dCoeff_C, dCoeff_NC, initial=True, sceptics=True, welfare=False)

    vt_stay_c = result[0]
    vt_renter = result[2]

    assert np.any(vt_stay_c != 0), "vt_stay_c is all zeros — solve_ss didn't compute"
    assert np.any(vt_renter != 0), "vt_renter is all zeros — solve_ss didn't compute"


def test_solve_ss_snapshot(par_reduced, grids_and_markov_reduced):
    """Pin solve_ss output against saved snapshot (created on first run)."""
    import household_problem as hp

    grids, mMarkov = grids_and_markov_reduced
    dCoeff_C = 0.69906474
    dCoeff_NC = 0.78259554

    result = hp.solve_ss(grids, par_reduced, par_reduced.iNj, mMarkov,
                         dCoeff_C, dCoeff_NC, initial=True, sceptics=True, welfare=False)

    vt_stay_c, vt_stay_nc, vt_renter, b_stay_c, b_stay_nc, b_renter = result

    snapshot_path = os.path.join(SNAPSHOTS_DIR, "solve_ss_reduced.npz")

    if not os.path.exists(snapshot_path):
        # First run: save snapshot
        np.savez(snapshot_path,
                 vt_stay_c=vt_stay_c, vt_stay_nc=vt_stay_nc, vt_renter=vt_renter,
                 b_stay_c=b_stay_c, b_stay_nc=b_stay_nc, b_renter=b_renter)
        return  # nothing to compare yet

    # Subsequent runs: compare against snapshot
    ref = np.load(snapshot_path)
    for name in ["vt_stay_c", "vt_stay_nc", "vt_renter", "b_stay_c", "b_stay_nc", "b_renter"]:
        assert np.allclose(result[["vt_stay_c", "vt_stay_nc", "vt_renter",
                                   "b_stay_c", "b_stay_nc", "b_renter"].index(name)],
                           ref[name], atol=1e-10, rtol=1e-10), \
            f"{name} drifted from snapshot"
