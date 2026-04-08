"""
conftest.py

Session-scoped fixtures for the regression test suite.
Numba compiles on first import (~3 min); subsequent tests reuse compiled code.
"""

import sys
import os
import pytest
import numpy as np

# ---------------------------------------------------------------------------
# Ensure clean_the_mess/ is on sys.path so bare imports (e.g. "import tauchen")
# resolve from the source directory, matching how the model is normally run.
# ---------------------------------------------------------------------------
_SRC_DIR = os.path.join(os.path.dirname(__file__), os.pardir)
_SRC_DIR = os.path.abspath(_SRC_DIR)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

# ---------------------------------------------------------------------------
# All importable modules (excludes full_calibration which needs nlopt)
# ---------------------------------------------------------------------------
ALL_MODULES = [
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

# ---------------------------------------------------------------------------
# Session fixture: import all njit modules once to trigger Numba compilation
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def compiled_modules():
    """Import every module once so Numba @njit functions are compiled.

    Returns a dict mapping module name -> module object.
    """
    import importlib
    modules = {}
    for name in ALL_MODULES:
        modules[name] = importlib.import_module(name)
    return modules


# ---------------------------------------------------------------------------
# Session fixture: par object (created via construct_jitclass on par_dict)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def par():
    """Create the numba jitclass parameter object from par_epsilons.par_dict."""
    import misc_functions as misc
    import par as parfile
    return misc.construct_jitclass(parfile.par_dict)


# ---------------------------------------------------------------------------
# Session fixture: grids and Markov matrix (full-size, default parameters)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def grids_and_markov(par):
    """Create grids and Markov transition matrix at full (production) size.

    Returns (grids, mMarkov).
    """
    import grid_creation
    grids, mMarkov = grid_creation.create(par)
    return grids, mMarkov


@pytest.fixture(scope="session")
def grids(grids_and_markov):
    """The grids jitclass object (full size)."""
    return grids_and_markov[0]


@pytest.fixture(scope="session")
def mMarkov(grids_and_markov):
    """The Tauchen Markov transition matrix (full size)."""
    return grids_and_markov[1]


# ---------------------------------------------------------------------------
# Session fixture: reduced-size par, grids, mMarkov for fast VFI tests
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def reduced_par():
    """Create par with iXin=3 and iNumStates=3 for faster VFI tests."""
    import copy
    import misc_functions as misc
    import par as parfile

    # Make a shallow copy of par_dict and override grid sizes
    reduced_dict = dict(parfile.par_dict)
    reduced_dict["iXin"] = 3
    reduced_dict["iNumStates"] = 3
    return misc.construct_jitclass(reduced_dict)


@pytest.fixture(scope="session")
def reduced_grids_and_markov(reduced_par):
    """Create grids and Markov matrix on the reduced (iXin=3, iNumStates=3) grid."""
    import grid_creation
    grids, mMarkov = grid_creation.create(reduced_par)
    return grids, mMarkov


@pytest.fixture(scope="session")
def reduced_grids(reduced_grids_and_markov):
    """Reduced grids jitclass object."""
    return reduced_grids_and_markov[0]


@pytest.fixture(scope="session")
def reduced_mMarkov(reduced_grids_and_markov):
    """Reduced Tauchen Markov transition matrix."""
    return reduced_grids_and_markov[1]


# ---------------------------------------------------------------------------
# Hardcoded coefficient vectors from solve_epsilons.py
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def coefficient_vectors():
    """Return the hardcoded coefficient vectors from solve_epsilons.py."""
    return {
        "vCoeff_C_initial": np.array([0.69906474, 0., 0., 0., 0.]),
        "vCoeff_NC_initial": np.array([0.78259554, 0., 0., 0., 0.]),
        "vCoeff_C": np.array([0.66335385, -0.03015386, 0.00541847, 0.00797395, 0.00249396]),
        "vCoeff_NC": np.array([0.81033554, 0.01679082, -0.00574326, -0.00115107, 0.00101112]),
        "vCoeff_C_terminal_RE": np.array([0.58952906, 0., 0., 0., 0.]),
        "vCoeff_NC_terminal_RE": np.array([0.85484033, 0., 0., 0., 0.]),
        "vCoeff_C_terminal_HE": np.array([0.64908636, 0., 0., 0., 0.]),
        "vCoeff_NC_terminal_HE": np.array([0.82124315, 0., 0., 0., 0.]),
    }


# ---------------------------------------------------------------------------
# Path to snapshots directory
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def snapshots_dir():
    """Return the path to the tests/snapshots/ directory, creating it if needed."""
    d = os.path.join(os.path.dirname(__file__), "snapshots")
    os.makedirs(d, exist_ok=True)
    return d
