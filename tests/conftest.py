import sys
import os
import pytest
import numpy as np

# Add clean_the_mess to path so imports work
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'clean_the_mess'))


@pytest.fixture(scope="session")
def par_full():
    """Full-grid parameter object. Created once per session (Numba compiles on first use)."""
    import misc_functions as misc
    import par as parfile
    return misc.construct_jitclass(parfile.par_dict)


@pytest.fixture(scope="session")
def grids_and_markov_full(par_full):
    """Full grids + Markov matrix. Created once per session."""
    import grid_creation
    grids, mMarkov = grid_creation.create(par_full)
    return grids, mMarkov


@pytest.fixture(scope="session")
def par_reduced():
    """Reduced-grid parameter object for fast tests (iXin=3, iNumStates=3)."""
    import misc_functions as misc
    import par as parfile
    par_dict = parfile.par_dict.copy()
    par_dict["iXin"] = 3
    par_dict["iNumStates"] = 3
    return misc.construct_jitclass(par_dict)


@pytest.fixture(scope="session")
def grids_and_markov_reduced(par_reduced):
    """Reduced grids + Markov matrix for fast tests."""
    import grid_creation
    grids, mMarkov = grid_creation.create(par_reduced)
    return grids, mMarkov


# Known good coefficient values from solve_epsilons.py
VCOEFF_C_INITIAL = np.array([0.69906474, 0., 0., 0., 0.])
VCOEFF_NC_INITIAL = np.array([0.78259554, 0., 0., 0., 0.])
VCOEFF_C = np.array([0.66335385, -0.03015386, 0.00541847, 0.00797395, 0.00249396])
VCOEFF_NC = np.array([0.81033554, 0.01679082, -0.00574326, -0.00115107, 0.00101112])
