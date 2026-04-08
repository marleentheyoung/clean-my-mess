"""
buyer_problem.py
"""

import numpy as np
from numba import njit
import utility as ut
import interp as interpfun

NEG_INF = -1e12

@njit
def solve(par, grids, j_index, k_index, g_index, e_index, dP, mVt_stayer, mC_pol_stayer, mVt_stayer_wf, welfare):
    
    j=j_index
    g=grids.vG[g_index]

    mVt = np.ones((grids.vX.size))*NEG_INF
    mVt_wf = np.ones((grids.vX.size))*NEG_INF
    mQt = np.zeros((grids.vX.size))     
    
    max_mortgage_pti=grids.mPTI[j_index,e_index] 

    for x_index in range(grids.vX.size):
        x=grids.vX[x_index]
        for h_index in range(grids.vH.size):
            h=grids.vH[h_index]
            for l_index in range(grids.vL.size):
                ltv = grids.vL[l_index]  
                if (l_index>0 and j_index==par.iNj-1) or ltv>grids.max_ltv:
                    continue   
                if ltv*dP*h < max_mortgage_pti:
                    if l_index==0:
                        buyer_cost=(dP*(1+par.dKappa_buy))*h                      
                        m_buyer= x - buyer_cost 
                    else:
                        buyer_cost=(dP*(1+par.dKappa_buy))*h + ltv*dP*h*par.dZeta + par.dZeta_fixed                       
                        m_buyer= x - buyer_cost + ltv*dP*h                        
                    if m_buyer>=grids.vM[0] and m_buyer<=grids.vM[-1]:
                        Vt_candidate=interpfun.interp_1d(grids.vM,mVt_stayer[:,h_index, l_index],m_buyer)
                        if Vt_candidate > mVt[x_index]:
                            C_pol=interpfun.interp_1d(grids.vM,mC_pol_stayer[:,h_index, l_index],m_buyer)
                            assert C_pol > 0
                            mVt[x_index] = Vt_candidate                                  
                            mQt[x_index] = -1/ut.u_c(j,C_pol,h,g+par.dOmega, par) 
                            if welfare == True:
                                mVt_wf[x_index] = interpfun.interp_1d(grids.vM,mVt_stayer_wf[:,h_index, l_index],m_buyer)

                    if m_buyer>grids.vM[-1]:
                        Vt_candidate=-1/(-1/mVt_stayer[-1,h_index, l_index]+ut.u(j,mC_pol_stayer[-1,h_index, l_index]+m_buyer-grids.vM[-1],h,g+par.dOmega, par)-ut.u(j,mC_pol_stayer[-1,h_index, l_index],h,g+par.dOmega, par))
                        if Vt_candidate > mVt[x_index]:
                            C_pol=mC_pol_stayer[-1,h_index, l_index]+m_buyer-grids.vM[-1]
                            assert C_pol > 0
                            mVt[x_index] = Vt_candidate                                  
                            mQt[x_index] = -1/ut.u_c(j,C_pol,h,g+par.dOmega, par) 
                            if welfare == True:
                                mVt_wf[x_index] = -1/(-1/mVt_stayer_wf[-1,h_index, l_index]+ut.u(j,C_pol,h,g+par.dOmega, par)-ut.u(j,mC_pol_stayer[-1,h_index, l_index],h,g+par.dOmega, par))
       

    assert np.isnan(mVt).sum() == 0
    assert np.isnan(mQt).sum() == 0
                   
    return mVt, mQt, mVt_wf
