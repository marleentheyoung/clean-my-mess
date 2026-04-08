"""
test_vfi.py  --  Stage B

VFI and distribution tests on reduced grids (iXin=3, iNumStates=3).
Runtime: ~5 min excluding first Numba compilation.
"""

import hashlib
import os
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Coefficient scalars (first element of vCoeff vectors from solve_epsilons.py)
# ---------------------------------------------------------------------------
DCOEFF_C_INITIAL = 0.69906474
DCOEFF_NC_INITIAL = 0.78259554

VCOEFF_C_INITIAL = np.array([0.69906474, 0., 0., 0., 0.])
VCOEFF_NC_INITIAL = np.array([0.78259554, 0., 0., 0., 0.])


def _array_checksum(arr):
    """Compute a SHA256 hex digest of a numpy array's raw bytes."""
    return hashlib.sha256(arr.tobytes()).hexdigest()


# ---------------------------------------------------------------------------
# VFI: solve_ss with welfare=False
# ---------------------------------------------------------------------------

class TestSolveSS:
    """Run solve_ss on reduced grids and verify output sanity."""

    @pytest.fixture(scope="class")
    def solve_ss_result(self, reduced_grids, reduced_par, reduced_mMarkov):
        """Run solve_ss once (welfare=False) and cache for all tests in class."""
        import household_problem as household_problem

        result = household_problem.solve_ss(
            reduced_grids, reduced_par, reduced_par.iNj, reduced_mMarkov,
            DCOEFF_C_INITIAL, DCOEFF_NC_INITIAL,
            True,   # initial
            True,   # sceptics
            False,  # welfare
        )
        # Returns: (vt_stay_c, vt_stay_nc, vt_renter, b_stay_c, b_stay_nc, b_renter)
        return result

    def test_returns_six_arrays(self, solve_ss_result):
        assert len(solve_ss_result) == 6

    def test_value_functions_finite(self, solve_ss_result):
        vt_stay_c, vt_stay_nc, vt_renter, b_stay_c, b_stay_nc, b_renter = solve_ss_result
        for name, arr in [
            ("vt_stay_c", vt_stay_c),
            ("vt_stay_nc", vt_stay_nc),
            ("vt_renter", vt_renter),
        ]:
            assert np.all(np.isfinite(arr)), f"{name} contains NaN or Inf"

    def test_policy_functions_finite(self, solve_ss_result):
        _, _, _, b_stay_c, b_stay_nc, b_renter = solve_ss_result
        for name, arr in [
            ("b_stay_c", b_stay_c),
            ("b_stay_nc", b_stay_nc),
            ("b_renter", b_renter),
        ]:
            assert np.all(np.isfinite(arr)), f"{name} contains NaN or Inf"

    def test_value_function_shapes(self, solve_ss_result, reduced_grids, reduced_par):
        vt_stay_c, vt_stay_nc, vt_renter, b_stay_c, b_stay_nc, b_renter = solve_ss_result
        iNj = reduced_par.iNj
        k_dim = reduced_grids.vK.size  # 2 (sceptics=True)
        g_dim = reduced_grids.vG.size  # 3 (iXin=3)
        m_dim = reduced_grids.vM.size
        h_dim = reduced_grids.vH.size
        l_dim = reduced_grids.vL.size
        e_dim = reduced_grids.vE.size  # 3 (iNumStates=3)
        x_dim = reduced_grids.vX.size

        expected_stay = (1, iNj, k_dim, g_dim, m_dim, h_dim, l_dim, e_dim)
        expected_rent = (1, iNj, k_dim, g_dim, x_dim, e_dim)

        assert vt_stay_c.shape == expected_stay, f"vt_stay_c shape {vt_stay_c.shape} != {expected_stay}"
        assert vt_stay_nc.shape == expected_stay, f"vt_stay_nc shape {vt_stay_nc.shape} != {expected_stay}"
        assert vt_renter.shape == expected_rent, f"vt_renter shape {vt_renter.shape} != {expected_rent}"
        assert b_stay_c.shape == expected_stay
        assert b_stay_nc.shape == expected_stay
        assert b_renter.shape == expected_rent

    def test_value_functions_not_all_zero(self, solve_ss_result):
        vt_stay_c, vt_stay_nc, vt_renter, _, _, _ = solve_ss_result
        assert not np.all(vt_stay_c == 0), "vt_stay_c is all zeros"
        assert not np.all(vt_stay_nc == 0), "vt_stay_nc is all zeros"
        assert not np.all(vt_renter == 0), "vt_renter is all zeros"

    def test_value_functions_have_reasonable_range(self, solve_ss_result):
        """Value functions should have nonzero spread (not constant)."""
        vt_stay_c, vt_stay_nc, vt_renter, _, _, _ = solve_ss_result
        nonzero_c = vt_stay_c[vt_stay_c != 0]
        nonzero_nc = vt_stay_nc[vt_stay_nc != 0]
        nonzero_r = vt_renter[vt_renter != 0]
        if nonzero_c.size > 1:
            assert np.std(nonzero_c) > 0, "vt_stay_c is constant (zero std)"
        if nonzero_nc.size > 1:
            assert np.std(nonzero_nc) > 0, "vt_stay_nc is constant (zero std)"
        if nonzero_r.size > 1:
            assert np.std(nonzero_r) > 0, "vt_renter is constant (zero std)"

    def test_save_snapshot(self, solve_ss_result, snapshots_dir):
        """Save shapes and checksums to snapshots/ for future comparison."""
        vt_stay_c, vt_stay_nc, vt_renter, b_stay_c, b_stay_nc, b_renter = solve_ss_result
        snapshot_path = os.path.join(snapshots_dir, "solve_ss_reduced.npz")

        np.savez(
            snapshot_path,
            vt_stay_c_shape=np.array(vt_stay_c.shape),
            vt_stay_nc_shape=np.array(vt_stay_nc.shape),
            vt_renter_shape=np.array(vt_renter.shape),
            vt_stay_c_checksum=np.array([hash(_array_checksum(vt_stay_c))]),
            vt_stay_nc_checksum=np.array([hash(_array_checksum(vt_stay_nc))]),
            vt_renter_checksum=np.array([hash(_array_checksum(vt_renter))]),
            # Also save a small representative slice for debugging
            vt_stay_c_slice=vt_stay_c[0, 0, 0, 0, :5, 0, 0, :],
            vt_stay_nc_slice=vt_stay_nc[0, 0, 0, 0, :5, 0, 0, :],
            vt_renter_slice=vt_renter[0, 0, 0, 0, :5, :],
        )
        assert os.path.exists(snapshot_path)


# ---------------------------------------------------------------------------
# VFI: solve_ss with welfare=True
# ---------------------------------------------------------------------------

class TestSolveSSWelfare:
    """Run solve_ss with welfare=True on reduced grids."""

    @pytest.fixture(scope="class")
    def solve_ss_welfare_result(self, reduced_grids, reduced_par, reduced_mMarkov):
        """Run solve_ss once (welfare=True) and cache."""
        import household_problem as household_problem

        result = household_problem.solve_ss(
            reduced_grids, reduced_par, reduced_par.iNj, reduced_mMarkov,
            DCOEFF_C_INITIAL, DCOEFF_NC_INITIAL,
            True,   # initial
            True,   # sceptics
            True,   # welfare
        )
        # Returns: (v_owner_c_wf, v_owner_nc_wf, v_nonowner_wf, b_stay_c, b_stay_nc, b_renter)
        return result

    def test_returns_six_arrays(self, solve_ss_welfare_result):
        assert len(solve_ss_welfare_result) == 6

    def test_welfare_values_finite(self, solve_ss_welfare_result):
        v_owner_c_wf, v_owner_nc_wf, v_nonowner_wf, _, _, _ = solve_ss_welfare_result
        for name, arr in [
            ("v_owner_c_wf", v_owner_c_wf),
            ("v_owner_nc_wf", v_owner_nc_wf),
            ("v_nonowner_wf", v_nonowner_wf),
        ]:
            assert np.all(np.isfinite(arr)), f"{name} contains NaN or Inf"

    def test_welfare_value_shapes(self, solve_ss_welfare_result, reduced_grids, reduced_par):
        v_owner_c_wf, v_owner_nc_wf, v_nonowner_wf, _, _, _ = solve_ss_welfare_result
        iNj = reduced_par.iNj
        k_dim = reduced_grids.vK.size
        g_dim = reduced_grids.vG.size
        m_dim = reduced_grids.vM.size
        h_dim = reduced_grids.vH.size
        l_dim = reduced_grids.vL.size
        e_dim = reduced_grids.vE.size
        x_dim = reduced_grids.vX.size

        expected_stay = (1, iNj, k_dim, g_dim, m_dim, h_dim, l_dim, e_dim)
        expected_rent = (1, iNj, k_dim, g_dim, x_dim, e_dim)

        assert v_owner_c_wf.shape == expected_stay
        assert v_owner_nc_wf.shape == expected_stay
        assert v_nonowner_wf.shape == expected_rent


# ---------------------------------------------------------------------------
# Stationary distribution: stat_dist_finder
# ---------------------------------------------------------------------------

class TestStatDistFinder:
    """Run stat_dist_finder on reduced grids and verify distribution properties."""

    @pytest.fixture(scope="class")
    def stat_dist_result(self, reduced_grids, reduced_par, reduced_mMarkov):
        """Run solve_ss then stat_dist_finder on reduced grids."""
        import household_problem as household_problem
        import simulation as sim

        # First get the value/policy functions
        vt_stay_c, vt_stay_nc, vt_renter, b_stay_c, b_stay_nc, b_renter = \
            household_problem.solve_ss(
                reduced_grids, reduced_par, reduced_par.iNj, reduced_mMarkov,
                DCOEFF_C_INITIAL, DCOEFF_NC_INITIAL,
                True,   # initial
                True,   # sceptics
                False,  # welfare
            )

        bequest_guess = np.zeros(3)
        result = sim.stat_dist_finder(
            True,  # sceptics
            reduced_grids, reduced_par, reduced_mMarkov,
            reduced_par.iNj,
            vt_stay_c[0,], vt_stay_nc[0,], vt_renter[0,],
            b_stay_c[0,], b_stay_nc[0,], b_renter[0,],
            VCOEFF_C_INITIAL, VCOEFF_NC_INITIAL,
            bequest_guess,
            True,  # initial
        )
        return result

    def test_returns_15_elements(self, stat_dist_result):
        """stat_dist_finder returns a 15-element tuple."""
        assert len(stat_dist_result) == 15

    def test_distributions_nonnegative(self, stat_dist_result):
        mDist1_c = stat_dist_result[0]
        mDist1_nc = stat_dist_result[1]
        mDist1_renter = stat_dist_result[2]
        assert np.all(mDist1_c >= 0), "Coastal distribution has negative entries"
        assert np.all(mDist1_nc >= 0), "Non-coastal distribution has negative entries"
        assert np.all(mDist1_renter >= 0), "Renter distribution has negative entries"

    def test_distributions_finite(self, stat_dist_result):
        mDist1_c = stat_dist_result[0]
        mDist1_nc = stat_dist_result[1]
        mDist1_renter = stat_dist_result[2]
        assert np.all(np.isfinite(mDist1_c)), "Coastal distribution has NaN/Inf"
        assert np.all(np.isfinite(mDist1_nc)), "Non-coastal distribution has NaN/Inf"
        assert np.all(np.isfinite(mDist1_renter)), "Renter distribution has NaN/Inf"

    def test_total_distribution_sums_to_one(self, stat_dist_result):
        """Total mass across all three distributions should sum to ~1."""
        mDist1_c = stat_dist_result[0]
        mDist1_nc = stat_dist_result[1]
        mDist1_renter = stat_dist_result[2]
        total_mass = np.sum(mDist1_c) + np.sum(mDist1_nc) + np.sum(mDist1_renter)
        assert np.isclose(total_mass, 1.0, rtol=1e-3), (
            f"Total distribution mass = {total_mass}, expected ~1.0"
        )

    def test_distributions_not_all_zero(self, stat_dist_result):
        mDist1_c = stat_dist_result[0]
        mDist1_nc = stat_dist_result[1]
        mDist1_renter = stat_dist_result[2]
        # At least one of the distributions should have nonzero entries
        assert np.sum(mDist1_c) > 0 or np.sum(mDist1_nc) > 0 or np.sum(mDist1_renter) > 0

    def test_save_snapshot(self, stat_dist_result, snapshots_dir):
        """Save distribution summary statistics to snapshots/."""
        mDist1_c = stat_dist_result[0]
        mDist1_nc = stat_dist_result[1]
        mDist1_renter = stat_dist_result[2]
        snapshot_path = os.path.join(snapshots_dir, "stat_dist_reduced.npz")

        np.savez(
            snapshot_path,
            mass_coastal=np.array([np.sum(mDist1_c)]),
            mass_noncoastal=np.array([np.sum(mDist1_nc)]),
            mass_renter=np.array([np.sum(mDist1_renter)]),
            total_mass=np.array([np.sum(mDist1_c) + np.sum(mDist1_nc) + np.sum(mDist1_renter)]),
            dist_c_shape=np.array(mDist1_c.shape),
            dist_nc_shape=np.array(mDist1_nc.shape),
            dist_renter_shape=np.array(mDist1_renter.shape),
        )
        assert os.path.exists(snapshot_path)
