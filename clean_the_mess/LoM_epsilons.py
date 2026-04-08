"""
LoM.py

Law of Motion for house prices. Evaluates Chebyshev polynomial
forecasting rule given time index and coefficient vector.
"""
###########################################################
### Imports

from numba import njit


###########################################################
### Functions
###########################################################
@njit
def LoM(grids, t_index, vCoeff):
    """Evaluate Chebyshev price law of motion.

    Works for both coastal and non-coastal — the coefficient vector
    determines which market.
    """
    t = grids.vTime[t_index]
    t_cheby = (2*t - (grids.vTime[0] + grids.vTime[-1])) / (grids.vTime[-1] - grids.vTime[0])
    t_1 = t_cheby
    t_2 = 2*t_cheby**2 - 1
    t_3 = 4*t_cheby**3 - 3*t_cheby
    t_4 = 8*t_cheby**4 - 8*t_cheby**2 + 1

    P = vCoeff[0] + vCoeff[1]*t_1 + vCoeff[2]*t_2 + vCoeff[3]*t_3 + vCoeff[4]*t_4

    return P


# Aliases for backward compatibility during transition
LoM_C = LoM
LoM_NC = LoM
