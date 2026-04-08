"""
LoM.py
"""
###########################################################
### Imports

from numba import njit


###########################################################
### Functions
###########################################################
@njit
def LoM_C(grids,t_index,vCoeff_C):
    t=grids.vTime[t_index]
    t_cheby=(2*t-(grids.vTime[0]+grids.vTime[-1]))/(grids.vTime[-1]-grids.vTime[0])
    t_1=t_cheby
    t_2=2*t_cheby**2-1
    t_3=4*t_cheby**3-3*t_cheby
    t_4=8*t_cheby**4-8*t_cheby**2+1
 

    P_c = vCoeff_C[0]+vCoeff_C[1]*t_1+vCoeff_C[2]*t_2+vCoeff_C[3]*t_3+vCoeff_C[4]*t_4
    
    return P_c
@njit
def LoM_NC(grids,t_index,vCoeff_NC):
    t=grids.vTime[t_index]
    t_cheby=(2*t-(grids.vTime[0]+grids.vTime[-1]))/(grids.vTime[-1]-grids.vTime[0])
    t_1=t_cheby
    t_2=2*t_cheby**2-1
    t_3=4*t_cheby**3-3*t_cheby
    t_4=8*t_cheby**4-8*t_cheby**2+1       

    P_nc = vCoeff_NC[0]+vCoeff_NC[1]*t_1+vCoeff_NC[2]*t_2+vCoeff_NC[3]*t_3+vCoeff_NC[4]*t_4
      
    return P_nc