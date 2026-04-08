"""
misc_functions.py

Utility functions: grid construction, jitclass builder, OLS, net income.
Interpolation functions have been consolidated into interp.py (task 5b.1).
"""

import numpy as np
import numba as nb
from numba import njit

# Re-export interpolation functions for backward compatibility
# (consumers that import misc.interp_3d etc. will still work)
from interp import (binary_search, binary_search_sim,
                    _interp_2d, interp_2d,
                    _interp_3d, interp_3d,
                    _interp_4d, interp_4d)

###########################################################
### mAH =  DoubleGrid(vA, vH)
@njit
def DoubleGrid(vA, vH):
    """
    Returns an (iNa*iNh) x 2 matrix containing all possible combinations
    of savings (first column) and house values (second column).
    """
    iNa = vA.shape[0]
    iNh = vH.shape[0]
    N   = iNa * iNh

    # Allocate the output grid
    mAH = np.empty((N, 2), dtype=vA.dtype)

    # Fill it: for each house-value index j and saving-index i,
    # place (vA[i], vH[j]) at row idx = j*iNa + i
    for j in range(iNh):
        for i in range(iNa):
            idx = j * iNa + i
            mAH[idx, 0] = vA[i]
            mAH[idx, 1] = vH[j]

    return mAH

###########################################################
### Interpolation helper
@njit
def lininterp_zero_crossing(x_min, y_min, x_max, y_max):
    # Calculate the x value where the line between (x1, y1) and (x2, y2) crosses y=0
    x_zero = x_min + (- y_min) * (x_max - x_min) / (y_max - y_min)
    return x_zero

###########################################################
### vMaxRow, vArgMaxRow= maxRow(mMatrix)
@njit
def maxRow(mMatrix):
    """Returns the MaxRow of a matrix.

    Args:
        mMatrix (matrix, float): input matrix

    Returns:
        vMaxRow (vector, float): Maxrow vector
        vArgMaxRow (vector, integer): argument of max row in each col
    """
    iCols = mMatrix.shape[1]
    vMaxRow = np.zeros(iCols, dtype = nb.float64)
    vArgMaxRow = np.zeros(iCols, dtype = nb.int64)
    for i in range(iCols):
        vMaxRow[i] = np.max(mMatrix[:,i])
        vArgMaxRow[i] = np.argmax(mMatrix[:,i])
    return vMaxRow, vArgMaxRow

###########################################################
### par = construct_jitclass(dictionary):
def construct_jitclass(dictionary):
    """
    Convert dictionary into numba object.

    Args:
        dictionary (dictionary): Dictionary with all model parameters or other objects.

    Returns:
        par (numba object): Numba object with all model parameters or other objects.
    """

    parlist = [(key, nb.typeof(val)) for key, val in dictionary.items()]

    @nb.experimental.jitclass(parlist)
    class ParClass:
        def __init__(self):
            pass

    par = ParClass()
    for key, val in dictionary.items():
        setattr(par, key, val)

    return par

###########################################################
### OLS
@njit
def ols_numba(X, Y):
    """Compute OLS estimates using (X'X)^-1 X'Y, Numba optimized."""
    XtX = X.T @ X
    XtY = X.T @ Y
    beta = np.linalg.solve(XtX, XtY)
    return beta

###########################################################
### Net income
@njit
def net_income(par, grids, j, e_index, e_trans_index, mortgage_size):
    if j<par.j_ret:
        pretax_income=np.exp(grids.vChi[j] + grids.vE[e_index]+ grids.vE_trans[e_trans_index])
    else:
        pretax_income=0.7*np.exp(grids.vChi[par.j_ret-1] + grids.vE[e_index])
    posttax_income=pretax_income-par.tau_0*(max(pretax_income-par.r_m*mortgage_size*grids.median_inc,0))**(1-par.tau_1)
    mortgage_rebate=par.tau_0*(pretax_income)**(1-par.tau_1)-(pretax_income-posttax_income)
    ##Normalise with pre-tax median income
    net_income=(posttax_income/grids.median_inc)*(1-par.wf_wedge[0])
    mortgage_rebate=(mortgage_rebate/grids.median_inc)*(1-par.wf_wedge[0])

    return net_income, mortgage_rebate
