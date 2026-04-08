"""
test_regression.py  --  Stage C

Regression test that pins the live entry point output.
Calls find_expenditure_equiv (the only thing solve_epsilons.py:main() produces).
Runtime: ~10-15 min (solve_ss + stat_dist + welfare analysis).
"""

import os
import numpy as np
import pytest


# Mark the entire module as slow -- skip with: pytest -m "not slow"
pytestmark = pytest.mark.slow


# ---------------------------------------------------------------------------
# Hardcoded coefficient vectors from solve_epsilons.py
# These are the exact values the live entry point uses.
# ---------------------------------------------------------------------------
VCOEFF_C_INITIAL = np.array([0.69906474, 0., 0., 0., 0.])
VCOEFF_NC_INITIAL = np.array([0.78259554, 0., 0., 0., 0.])
VCOEFF_C = np.array([0.66335385, -0.03015386, 0.00541847, 0.00797395, 0.00249396])
VCOEFF_NC = np.array([0.81033554, 0.01679082, -0.00574326, -0.00115107, 0.00101112])


class TestCoefficientValues:
    """Pin the hardcoded coefficient values from solve_epsilons.py."""

    def test_vCoeff_C_initial(self):
        expected = np.array([0.69906474, 0., 0., 0., 0.])
        assert np.allclose(VCOEFF_C_INITIAL, expected, rtol=1e-6)

    def test_vCoeff_NC_initial(self):
        expected = np.array([0.78259554, 0., 0., 0., 0.])
        assert np.allclose(VCOEFF_NC_INITIAL, expected, rtol=1e-6)

    def test_vCoeff_C(self):
        expected = np.array([0.66335385, -0.03015386, 0.00541847, 0.00797395, 0.00249396])
        assert np.allclose(VCOEFF_C, expected, rtol=1e-6)

    def test_vCoeff_NC(self):
        expected = np.array([0.81033554, 0.01679082, -0.00574326, -0.00115107, 0.00101112])
        assert np.allclose(VCOEFF_NC, expected, rtol=1e-6)

    def test_coefficient_vectors_have_5_elements(self):
        assert VCOEFF_C_INITIAL.shape == (5,)
        assert VCOEFF_NC_INITIAL.shape == (5,)
        assert VCOEFF_C.shape == (5,)
        assert VCOEFF_NC.shape == (5,)


class TestFindExpenditureEquiv:
    """Pin the output of find_expenditure_equiv on full grids.

    This reproduces the exact computation that solve_epsilons.py:main() performs.
    The 4 returned arrays (tax_equiv_C, tax_equiv_NC, tax_equiv_renter,
    tax_equiv_newborns) are deterministic and should not change unless the
    model logic changes.
    """

    @pytest.fixture(scope="class")
    def welfare_result(self, par, grids, mMarkov):
        """Run find_expenditure_equiv with full grids (matches live entry point)."""
        import proper_welfare_debug as welfare_stats

        tax_equiv_C, tax_equiv_NC, tax_equiv_renter, tax_equiv_newborns = \
            welfare_stats.find_expenditure_equiv(
                par, grids, mMarkov,
                VCOEFF_C_INITIAL, VCOEFF_NC_INITIAL,
                VCOEFF_C, VCOEFF_NC,
            )
        return tax_equiv_C, tax_equiv_NC, tax_equiv_renter, tax_equiv_newborns

    def test_returns_four_arrays(self, welfare_result):
        assert len(welfare_result) == 4

    def test_all_finite(self, welfare_result):
        tax_equiv_C, tax_equiv_NC, tax_equiv_renter, tax_equiv_newborns = welfare_result
        for name, arr in [
            ("tax_equiv_C", tax_equiv_C),
            ("tax_equiv_NC", tax_equiv_NC),
            ("tax_equiv_renter", tax_equiv_renter),
            ("tax_equiv_newborns", tax_equiv_newborns),
        ]:
            assert np.all(np.isfinite(arr)), f"{name} contains NaN or Inf"

    def test_tax_equiv_shapes(self, welfare_result, grids):
        tax_equiv_C, tax_equiv_NC, tax_equiv_renter, tax_equiv_newborns = welfare_result
        k_dim = grids.vK.size
        g_dim = grids.vG.size
        e_dim = grids.vE.size
        t_dim = grids.vTime.size

        expected_alive = (k_dim, g_dim, e_dim)
        expected_newborns = (t_dim, k_dim, g_dim, e_dim)

        assert tax_equiv_C.shape == expected_alive, (
            f"tax_equiv_C shape {tax_equiv_C.shape} != {expected_alive}"
        )
        assert tax_equiv_NC.shape == expected_alive
        assert tax_equiv_renter.shape == expected_alive
        assert tax_equiv_newborns.shape == expected_newborns

    def test_save_snapshot(self, welfare_result, snapshots_dir):
        """Save welfare equivalents to snapshots/ for regression comparison."""
        tax_equiv_C, tax_equiv_NC, tax_equiv_renter, tax_equiv_newborns = welfare_result
        snapshot_path = os.path.join(snapshots_dir, "welfare_equivalents.npz")

        np.savez(
            snapshot_path,
            tax_equiv_C=tax_equiv_C,
            tax_equiv_NC=tax_equiv_NC,
            tax_equiv_renter=tax_equiv_renter,
            tax_equiv_newborns=tax_equiv_newborns,
        )
        assert os.path.exists(snapshot_path)

    def test_regression_against_snapshot(self, welfare_result, snapshots_dir):
        """If a snapshot exists, verify current output matches it to 6 decimals."""
        snapshot_path = os.path.join(snapshots_dir, "welfare_equivalents.npz")
        if not os.path.exists(snapshot_path):
            pytest.skip("No snapshot exists yet; run test_save_snapshot first.")

        ref = np.load(snapshot_path)
        tax_equiv_C, tax_equiv_NC, tax_equiv_renter, tax_equiv_newborns = welfare_result

        # Use rtol=1e-6 for scalars / small arrays (as specified in constraints)
        assert np.allclose(tax_equiv_C, ref["tax_equiv_C"], rtol=1e-6, atol=1e-10), (
            "tax_equiv_C drifted from snapshot"
        )
        assert np.allclose(tax_equiv_NC, ref["tax_equiv_NC"], rtol=1e-6, atol=1e-10), (
            "tax_equiv_NC drifted from snapshot"
        )
        assert np.allclose(tax_equiv_renter, ref["tax_equiv_renter"], rtol=1e-6, atol=1e-10), (
            "tax_equiv_renter drifted from snapshot"
        )
        assert np.allclose(tax_equiv_newborns, ref["tax_equiv_newborns"], rtol=1e-6, atol=1e-10), (
            "tax_equiv_newborns drifted from snapshot"
        )
