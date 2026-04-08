

"""
VFI.py

Purpose:
    Solve for policy and value functions for a given law of motions
"""
###########################################################
### Imports
import numpy as np
import continuation_value_nolearning as continuation_value_epsilons
import stayer_problem as stayer_problem
import stayer_problem_renter as stayer_problem_renter
import buyer_problem as buyer_problem
from numba import njit
import lom as lom
import misc_functions as misc

###########################################################
### Functions
###########################################################



@njit
def solve(grids, par, iNj, mMarkov,vCoeff_C,vCoeff_NC, sceptics=True, welfare=True):
    if sceptics==False:
        k_dim=1
    else:
        k_dim=grids.vK.size        
        
    shape_stay                          = (grids.vTime.size, iNj, k_dim, grids.vG.size, grids.vM.size, grids.vH.size, grids.vL.size, grids.vE.size)
    shape_buy                           = (grids.vTime.size, iNj, k_dim, grids.vG.size, grids.vX.size,  grids.vE.size)
    shape_rent                          = (grids.vTime.size, iNj, k_dim, grids.vG.size, grids.vX.size, grids.vE.size)

    vt_buy_c        = np.zeros(shape_buy, dtype = np.float64) 
    vt_buy_c_wf      = np.zeros(shape_buy, dtype = np.float64) 
    qt_buy_nc       = np.zeros(shape_buy, dtype = np.float64) 
    vt_stay_c       = np.zeros(shape_stay, dtype = np.float64)
    vt_stay_c_wf    = np.zeros(shape_stay, dtype = np.float64)
    qt_stay_c       = np.zeros(shape_stay, dtype = np.float64) 
    vt_buy_nc       = np.zeros(shape_buy, dtype = np.float64)
    vt_buy_nc_wf    = np.zeros(shape_buy, dtype = np.float64)
    qt_buy_c        = np.zeros(shape_buy, dtype = np.float64) 
    vt_stay_nc      = np.zeros(shape_stay, dtype = np.float64)
    vt_stay_nc_wf   = np.zeros(shape_stay, dtype = np.float64)
    qt_stay_nc      = np.zeros(shape_stay, dtype = np.float64) 
    vt_renter  = np.zeros(shape_rent, dtype = np.float64)
    vt_renter_wf= np.zeros(shape_rent, dtype = np.float64)
    qt_renter  = np.zeros(shape_rent, dtype = np.float64) 
    b_stay_c        = np.zeros(shape_stay, dtype = np.float64) 
    b_stay_nc       = np.zeros(shape_stay, dtype = np.float64) 
    b_renter        = np.zeros(shape_rent, dtype = np.float64) 
    c_c             = np.zeros(shape_stay, dtype = np.float64) 
    c_nc            = np.zeros(shape_stay, dtype = np.float64) 
    
    #Matrix of expected welfare on a savings (before interest) grid
    v_owner_c_wf=np.zeros(shape_stay, dtype = np.float64) 
    v_owner_nc_wf=np.zeros(shape_stay, dtype = np.float64) 
    v_nonowner_wf=np.zeros(shape_rent, dtype = np.float64) 
    

    mortgage_size_C = np.zeros((grids.vH.size,grids.vL.size), dtype = np.float64)
    mortgage_size_NC= np.zeros((grids.vH.size,grids.vL.size), dtype = np.float64)
    type_mat = misc.DoubleGrid(np.linspace(0, k_dim-1, k_dim), np.linspace(0, grids.vG.size-1, grids.vG.size))
    
    g_index_nc=int((grids.vG.size-1)/2)
    assert grids.vG[g_index_nc]==1
    
    for type_index in range(type_mat.shape[0]):
        k_index = int(type_mat[type_index, 0])
        g_index = int(type_mat[type_index, 1])
        for t_index in range(grids.vTime.size-1, -1, -1):        
            dP_C = lom.LoM_C(grids,t_index,vCoeff_C)
            dP_NC = lom.LoM_NC(grids,t_index,vCoeff_NC)
            t_index_prime=min(t_index+1,grids.vTime.size-1)
            if t_index==grids.vTime.size-1:
                dPi_S=grids.vPi_S_median[t_index]
                dPi_L=grids.vPi_L[t_index]
                dP_C_prime = lom.LoM_C(grids,t_index,vCoeff_C)
                dP_NC_prime = lom.LoM_NC(grids,t_index,vCoeff_NC)

            else:
                dPi_S=grids.vPi_S_median[t_index+1]    
                dPi_L=grids.vPi_L[t_index+1]
                dP_C_prime = lom.LoM_C(grids,t_index+1, vCoeff_C)
                dP_NC_prime = lom.LoM_NC(grids,t_index+1,vCoeff_NC)
            for h_index in range(grids.vH.size):
                for l_index in range(grids.vL.size):
                    mortgage_size_C[h_index,l_index]=grids.vL[l_index]*grids.vH[h_index]*dP_C
                    mortgage_size_NC[h_index,l_index]=grids.vL[l_index]*grids.vH[h_index]*dP_NC
            for j in range(iNj-1, -1, -1):
                if j == iNj-1:
                    w_c_last,q_c_last, w_c_wf_last = continuation_value_epsilons.solve_last_period_owners_C(par, grids,  dPi_S, dPi_L, k_index, dP_C_prime,mortgage_size_C, welfare)
                    w_nc_last,q_nc_last, w_nc_wf_last  = continuation_value_epsilons.solve_last_period_owners_NC(par, grids,  k_index,  dP_NC_prime,mortgage_size_NC, welfare)
                    w_renter_last,q_renter_last, w_renter_wf_last = continuation_value_epsilons.solve_last_period_renters(par, grids)
                else:
                    coastal_stayer_inputs=precompute_coastal_stayer_inputs(shape_stay,j, k_index, g_index, t_index,
                        vt_stay_c,vt_stay_c_wf,qt_stay_c)
                    noncoastal_stayer_inputs=precompute_noncoastal_stayer_inputs(shape_stay,j, k_index, g_index, t_index,
                        vt_stay_nc,vt_stay_nc_wf,qt_stay_nc)
                    mover_inputs=precompute_mover_inputs(shape_stay, j, k_index, g_index, t_index,
                        vt_renter, vt_buy_c, vt_buy_nc,
                        vt_renter_wf, vt_buy_c_wf, vt_buy_nc_wf,
                        qt_renter, qt_buy_c, qt_buy_nc)
                    

                    
                    w_c_vE,q_c_vE, w_c_wf_vE, v_owner_c_wf[t_index_prime,j+1, k_index,g_index,:,:,:,:] = continuation_value_epsilons.solve_owners_C(par, grids,  j+1, k_index, mMarkov, dPi_S, dPi_L, coastal_stayer_inputs,mover_inputs,dP_C_prime, mortgage_size_C, welfare)
                    w_nc_vE,q_nc_vE, w_nc_wf_vE,  v_owner_nc_wf[t_index_prime,j+1, k_index,g_index,:,:,:,:]  = continuation_value_epsilons.solve_owners_NC(par, grids,  j+1, k_index, mMarkov, noncoastal_stayer_inputs,mover_inputs, dP_NC_prime,  mortgage_size_NC, welfare)
                    w_renter_vE,q_renter_vE, w_renter_wf_vE, v_nonowner_wf[t_index_prime,j+1, k_index,g_index,:,:] = continuation_value_epsilons.solve_renters(par, grids,  j+1, k_index, mMarkov, mover_inputs, welfare)
                                                             
               
                for e_index in range(grids.vE.size): 
                    if j == iNj-1:
                        w_c,q_c, w_c_wf = w_c_last[:,:,:],q_c_last[:,:,:], w_c_wf_last[:,:,:]
                        w_nc,q_nc, w_nc_wf  = w_nc_last[:,:,:],q_nc_last[:,:,:], w_nc_wf_last[:,:,:]
                        w_renter,q_renter, w_renter_wf= w_renter_last[:],q_renter_last[:], w_renter_wf_last[:]
                    else:
                        w_c,q_c, w_c_wf = w_c_vE[:, :, :, e_index],q_c_vE[:, :, :, e_index], w_c_wf_vE[:, :, :, e_index]
                        w_nc,q_nc, w_nc_wf = w_nc_vE[:, :, :, e_index],q_nc_vE[:, :, :, e_index], w_nc_wf_vE[:, :, :, e_index] 
                        w_renter,q_renter, w_renter_wf = w_renter_vE[:, e_index],q_renter_vE[:, e_index], w_renter_wf_vE[:, e_index]
                    vt_stay_c[t_index,j, k_index,g_index,:,:,:,e_index], c_c[t_index,j, k_index,g_index,:,:,:,e_index], qt_stay_c[t_index,j, k_index,g_index,:,:,:,e_index], b_stay_c[t_index,j, k_index,g_index,:,:,:,e_index], vt_stay_c_wf[t_index,j, k_index,g_index,:,:,:,e_index] = stayer_problem.solve(par, grids, j, k_index, g_index, w_c, q_c, w_c_wf, welfare)
                    vt_stay_nc[t_index,j, k_index,g_index,:,:,:,e_index], c_nc[t_index,j, k_index,g_index,:,:,:,e_index],  qt_stay_nc[t_index,j, k_index,g_index,:,:,:,e_index], b_stay_nc[t_index,j, k_index,g_index,:,:,:,e_index], vt_stay_nc_wf[t_index,j, k_index,g_index,:,:,:,e_index] = stayer_problem.solve(par, grids, j, k_index, g_index_nc, w_nc,q_nc, w_nc_wf, welfare)
                   
                    vt_renter[t_index,j, k_index,g_index,:,e_index],qt_renter[t_index,j, k_index,g_index,:,e_index], b_renter[t_index,j, k_index,g_index,:,e_index], vt_renter_wf[t_index,j, k_index,g_index,:,e_index] = stayer_problem_renter.solve(par, grids, j, k_index, g_index, t_index, dP_C,dP_NC, dP_C_prime,dP_NC_prime, w_renter,q_renter, w_renter_wf, welfare)

                    vt_buy_c[t_index,j, k_index,g_index,:,e_index], qt_buy_c[t_index,j, k_index,g_index,:,e_index], vt_buy_c_wf[t_index,j, k_index,g_index,:,e_index] = buyer_problem.solve(par, grids, j, k_index,  g_index, e_index, dP_C, vt_stay_c[t_index,j, k_index,g_index,:,:,:,e_index], c_c[t_index,j, k_index,g_index,:,:,:,e_index], vt_stay_c_wf[t_index,j, k_index,g_index,:,:,:,e_index], welfare)
                    vt_buy_nc[t_index,j, k_index,g_index,:,e_index], qt_buy_nc[t_index,j, k_index,g_index,:,e_index], vt_buy_nc_wf[t_index,j, k_index,g_index,:,e_index] = buyer_problem.solve(par, grids, j, k_index, g_index_nc, e_index, dP_NC, vt_stay_nc[t_index,j, k_index,g_index,:,:,:,e_index], c_nc[t_index,j, k_index,g_index,:,:,:,e_index], vt_stay_nc_wf[t_index,j, k_index,g_index,:,:,:,e_index], welfare)
                
                #We need an additional block of code because the trick of exploiting already computed continuation values doesn't work when t==0 or j==0 
                if welfare==True:
                    if j==0:
                        mover_inputs=precompute_mover_inputs(shape_stay, j-1, k_index, g_index, t_index,
                            vt_renter, vt_buy_c, vt_buy_nc,
                            vt_renter_wf, vt_buy_c_wf, vt_buy_nc_wf,
                            qt_renter, qt_buy_c, qt_buy_nc)
                        
                        _,_, _, v_nonowner_wf[t_index_prime,j, k_index,g_index,:,:] = continuation_value_epsilons.solve_renters(par, grids,  j, k_index, mMarkov, mover_inputs, welfare)
                    if t_index==0 and j<par.iNj-1:
                        dPi_S_0=grids.vPi_S_median[t_index]    
                        dPi_L_0=grids.vPi_L[t_index]
                        coastal_stayer_inputs=precompute_coastal_stayer_inputs(shape_stay,j, k_index, g_index, t_index-1,
                            vt_stay_c,vt_stay_c_wf,qt_stay_c)
                        noncoastal_stayer_inputs=precompute_noncoastal_stayer_inputs(shape_stay,j, k_index, g_index, t_index-1,
                            vt_stay_nc,vt_stay_nc_wf,qt_stay_nc)
                        mover_inputs=precompute_mover_inputs(shape_stay, j, k_index, g_index, t_index-1,
                            vt_renter, vt_buy_c, vt_buy_nc,
                            vt_renter_wf, vt_buy_c_wf, vt_buy_nc_wf,
                            qt_renter, qt_buy_c, qt_buy_nc)                  
    
                        
                        _,_, _, v_owner_c_wf[t_index,j+1, k_index,g_index,:,:,:,:] = continuation_value_epsilons.solve_owners_C(par, grids,  j+1, k_index, mMarkov, dPi_S_0, dPi_L_0, coastal_stayer_inputs,mover_inputs,dP_C_prime, mortgage_size_C, welfare)
                        _,_, _,  v_owner_nc_wf[t_index,j+1, k_index,g_index,:,:,:,:]  = continuation_value_epsilons.solve_owners_NC(par, grids,  j+1, k_index, mMarkov, noncoastal_stayer_inputs,mover_inputs, dP_NC_prime,  mortgage_size_NC, welfare)
                        _,_, _, v_nonowner_wf[t_index,j+1, k_index,g_index,:,:] = continuation_value_epsilons.solve_renters(par, grids,  j+1, k_index, mMarkov, mover_inputs, welfare)
                    if t_index==0 and j==0:
                        mover_inputs=precompute_mover_inputs(shape_stay, j-1, k_index, g_index, t_index-1,
                            vt_renter, vt_buy_c, vt_buy_nc,
                            vt_renter_wf, vt_buy_c_wf, vt_buy_nc_wf,
                            qt_renter, qt_buy_c, qt_buy_nc)
                    
                        _,_, _, v_nonowner_wf[t_index,j, k_index,g_index,:,:] = continuation_value_epsilons.solve_renters(par, grids,  j, k_index, mMarkov, mover_inputs, welfare)

                
    return vt_stay_c, vt_stay_nc, vt_renter, b_stay_c, b_stay_nc, b_renter, v_owner_c_wf, v_owner_nc_wf, v_nonowner_wf
 

@njit
def solve_ss(grids, par, iNj, mMarkov,dCoeff_C, dCoeff_NC, initial = True, sceptics=True, welfare=False):
    
    if sceptics==False:
        k_dim=1
    else:
        k_dim=grids.vK.size        
        
    shape_stay                          = (1,iNj, k_dim, grids.vG.size, grids.vM.size, grids.vH.size, grids.vL.size, grids.vE.size)
    shape_buy                           = (1,iNj, k_dim, grids.vG.size, grids.vX.size,  grids.vE.size)
    shape_rent                          = (1,iNj, k_dim, grids.vG.size, grids.vX.size, grids.vE.size)
   

    vt_buy_c        = np.zeros(shape_buy, dtype = np.float64) 
    vt_buy_c_wf      = np.zeros(shape_buy, dtype = np.float64) 
    qt_buy_nc       = np.zeros(shape_buy, dtype = np.float64) 
    vt_stay_c       = np.zeros(shape_stay, dtype = np.float64)
    vt_stay_c_wf    = np.zeros(shape_stay, dtype = np.float64)
    qt_stay_c       = np.zeros(shape_stay, dtype = np.float64) 
    vt_buy_nc       = np.zeros(shape_buy, dtype = np.float64)
    vt_buy_nc_wf    = np.zeros(shape_buy, dtype = np.float64)
    qt_buy_c        = np.zeros(shape_buy, dtype = np.float64) 
    vt_stay_nc      = np.zeros(shape_stay, dtype = np.float64)
    vt_stay_nc_wf   = np.zeros(shape_stay, dtype = np.float64)
    qt_stay_nc      = np.zeros(shape_stay, dtype = np.float64) 
    vt_renter  = np.zeros(shape_rent, dtype = np.float64)
    vt_renter_wf= np.zeros(shape_rent, dtype = np.float64)
    qt_renter  = np.zeros(shape_rent, dtype = np.float64) 
    b_stay_c        = np.zeros(shape_stay, dtype = np.float64) 
    b_stay_nc       = np.zeros(shape_stay, dtype = np.float64) 
    b_renter        = np.zeros(shape_rent, dtype = np.float64) 
    c_c             = np.zeros(shape_stay, dtype = np.float64) 
    c_nc            = np.zeros(shape_stay, dtype = np.float64) 
    
    #Matrix of expected welfare on a savings (before interest) grid
    v_owner_c_wf=np.zeros(shape_stay, dtype = np.float64) 
    v_owner_nc_wf=np.zeros(shape_stay, dtype = np.float64) 
    v_nonowner_wf=np.zeros(shape_rent, dtype = np.float64) 
      
    mortgage_size_C = np.zeros((grids.vH.size,grids.vL.size), dtype = np.float64)
    mortgage_size_NC= np.zeros((grids.vH.size,grids.vL.size), dtype = np.float64)
    dP_C= dCoeff_C
    dP_NC = dCoeff_NC
    dP_C_prime = dCoeff_C
    dP_NC_prime = dCoeff_NC
    for h_index in range(grids.vH.size):
        for l_index in range(grids.vL.size):
            mortgage_size_C[h_index,l_index]=grids.vL[l_index]*grids.vH[h_index]*dP_C
            mortgage_size_NC[h_index,l_index]=grids.vL[l_index]*grids.vH[h_index]*dP_NC
            
    g_index_nc=int((grids.vG.size-1)/2)
    assert grids.vG[g_index_nc]==1
    
    if initial:
        dPi_L=grids.vPi_L[0]
        t_index=0
    else:
        dPi_L=grids.vPi_L[-1]
        t_index=grids.vTime.size-1
    
    for k_index in range(k_dim):
        for g_index in range(grids.vG.size):
            if initial == True:
                dPi_S=par.vPi_S_median[0]
            else:
                dPi_S = par.vPi_S_median[-1]
            for j in range(iNj-1, -1, -1):
                if j == iNj-1:
                    w_c_last,q_c_last, w_c_wf_last = continuation_value_epsilons.solve_last_period_owners_C(par, grids,  dPi_S, dPi_L, k_index, dP_C_prime,mortgage_size_C, welfare)
                    w_nc_last,q_nc_last, w_nc_wf_last  = continuation_value_epsilons.solve_last_period_owners_NC(par, grids,  k_index, dP_NC_prime,mortgage_size_NC, welfare)
                    w_renter_last,q_renter_last, w_renter_wf_last = continuation_value_epsilons.solve_last_period_renters(par, grids)
                else:
                    coastal_stayer_inputs=precompute_coastal_stayer_inputs(shape_stay,j, k_index, g_index, 0,
                        vt_stay_c,vt_stay_c_wf,qt_stay_c)
                    noncoastal_stayer_inputs=precompute_noncoastal_stayer_inputs(shape_stay,j, k_index, g_index, 0,
                        vt_stay_nc,vt_stay_nc_wf,qt_stay_nc)
                    mover_inputs=precompute_mover_inputs(shape_stay, j, k_index, g_index, 0,
                        vt_renter, vt_buy_c, vt_buy_nc,
                        vt_renter_wf, vt_buy_c_wf, vt_buy_nc_wf,
                        qt_renter, qt_buy_c, qt_buy_nc)
                    
                    w_c_vE,q_c_vE, w_c_wf_vE, v_owner_c_wf[0,j+1, k_index,g_index,:,:,:,:] = continuation_value_epsilons.solve_owners_C(par, grids,  j+1, k_index, mMarkov, dPi_S,dPi_L, coastal_stayer_inputs,mover_inputs, dP_C_prime, mortgage_size_C, welfare)
                    w_nc_vE,q_nc_vE, w_nc_wf_vE, v_owner_nc_wf[0,j+1, k_index,g_index,:,:,:,:] = continuation_value_epsilons.solve_owners_NC(par, grids,  j+1, k_index, mMarkov, noncoastal_stayer_inputs,mover_inputs, dP_NC_prime,  mortgage_size_NC, welfare)
                    w_renter_vE,q_renter_vE, w_renter_wf_vE, v_nonowner_wf[0,j+1, k_index,g_index,:,:] = continuation_value_epsilons.solve_renters(par, grids,  j+1, k_index, mMarkov, mover_inputs, welfare)
    
                                            
                      
                for e_index in range(grids.vE.size): 
                    if j == iNj-1:
                        w_c,q_c, w_c_wf = w_c_last[:,:,:],q_c_last[:,:,:], w_c_wf_last[:,:,:]
                        w_nc,q_nc, w_nc_wf  = w_nc_last[:,:,:],q_nc_last[:,:,:], w_nc_wf_last[:,:,:]
                        w_renter,q_renter, w_renter_wf= w_renter_last[:],q_renter_last[:], w_renter_wf_last[:]
                    else:
                        w_c,q_c, w_c_wf= w_c_vE[:, :, :, e_index],q_c_vE[:, :, :, e_index], w_c_wf_vE[:, :, :, e_index]
                        w_nc,q_nc, w_nc_wf = w_nc_vE[:, :, :, e_index],q_nc_vE[:, :, :, e_index], w_nc_wf_vE[:, :, :, e_index] 
                        w_renter,q_renter, w_renter_wf = w_renter_vE[:, e_index],q_renter_vE[:, e_index], w_renter_wf_vE[:, e_index]
                        
                    
                    vt_stay_c[0,j, k_index,g_index,:,:,:,e_index], c_c[0,j, k_index,g_index,:,:,:,e_index], qt_stay_c[0,j, k_index,g_index,:,:,:,e_index], b_stay_c[0,j, k_index,g_index,:,:,:,e_index], vt_stay_c_wf[0,j, k_index,g_index,:,:,:,e_index] = stayer_problem.solve(par, grids, j, k_index, g_index, w_c, q_c, w_c_wf, welfare)
                    vt_stay_nc[0,j, k_index,g_index,:,:,:,e_index], c_nc[0,j, k_index,g_index,:,:,:,e_index],  qt_stay_nc[0,j, k_index,g_index,:,:,:,e_index], b_stay_nc[0,j, k_index,g_index,:,:,:,e_index], vt_stay_nc_wf[0,j, k_index,g_index,:,:,:,e_index] = stayer_problem.solve(par, grids, j, k_index, g_index_nc, w_nc,q_nc, w_nc_wf, welfare)
                    vt_renter[0,j, k_index,g_index,:,e_index],qt_renter[0,j, k_index,g_index,:,e_index], b_renter[0,j, k_index,g_index,:,e_index], vt_renter_wf[0,j, k_index,g_index,:,e_index] = stayer_problem_renter.solve(par, grids, j, k_index, g_index, t_index, dP_C,dP_NC, dP_C,dP_NC, w_renter,q_renter, w_renter_wf, welfare)
    
                    vt_buy_c[0,j, k_index,g_index,:,e_index], qt_buy_c[0,j, k_index,g_index,:,e_index], vt_buy_c_wf[0,j, k_index,g_index,:,e_index] = buyer_problem.solve(par, grids, j, k_index,  g_index, e_index, dP_C, vt_stay_c[0,j, k_index,g_index,:,:,:,e_index], c_c[0,j, k_index,g_index,:,:,:,e_index], vt_stay_c_wf[0,j, k_index,g_index,:,:,:,e_index], welfare)
                    vt_buy_nc[0,j, k_index,g_index,:,e_index], qt_buy_nc[0,j, k_index,g_index,:,e_index], vt_buy_nc_wf[0,j, k_index,g_index,:,e_index] = buyer_problem.solve(par, grids, j, k_index, g_index_nc, e_index, dP_NC, vt_stay_nc[0,j, k_index,g_index,:,:,:,e_index], c_nc[0,j, k_index,g_index,:,:,:,e_index], vt_stay_nc_wf[0,j, k_index,g_index,:,:,:,e_index], welfare)
    
    
    
                #We need an additional block of code because the trick of exploiting already computed continuation values doesn't work when t==0 or j==0 
                if welfare==True:
                    if j==0:
                        mover_inputs=precompute_mover_inputs(shape_stay, j-1, k_index, g_index, 0,
                            vt_renter, vt_buy_c, vt_buy_nc,
                            vt_renter_wf, vt_buy_c_wf, vt_buy_nc_wf,
                            qt_renter, qt_buy_c, qt_buy_nc)
                        
                        _,_, _, v_nonowner_wf[0,j, k_index,g_index,:,:] = continuation_value_epsilons.solve_renters(par, grids,  j, k_index, mMarkov, mover_inputs, welfare)


    if welfare == False:                      
        return vt_stay_c, vt_stay_nc, vt_renter, b_stay_c, b_stay_nc, b_renter
    else:
        return v_owner_c_wf, v_owner_nc_wf, v_nonowner_wf, b_stay_c, b_stay_nc, b_renter


@njit
def precompute_coastal_stayer_inputs(shape_stay,j_index, k_index, g_index, t_index,
    vt_stay_c,vt_stay_c_wf,qt_stay_c):

    if t_index==shape_stay[0]-1:
        coastal_stayer_inputs = {
            'vt_stay_c_input': vt_stay_c[t_index,j_index+1, k_index, g_index, :,:, :, : ],
            'vt_stay_c_input_wf': vt_stay_c_wf[t_index,j_index+1, k_index, g_index, :,:, :, :],
            'qt_stay_c_input': qt_stay_c[t_index,j_index+1, k_index, g_index, :,:, :, :],
             }
    else:
        coastal_stayer_inputs = {
            'vt_stay_c_input': vt_stay_c[t_index+1,j_index+1, k_index, g_index, :,:, :, : ],
            'vt_stay_c_input_wf': vt_stay_c_wf[t_index+1,j_index+1, k_index, g_index,:,:, :, :],
            'qt_stay_c_input': qt_stay_c[t_index+1,j_index+1, k_index, g_index, :,:, :, :],
             }

    return coastal_stayer_inputs

@njit
def precompute_noncoastal_stayer_inputs(shape_stay,j_index, k_index, g_index, t_index,
    vt_stay_nc,vt_stay_nc_wf,qt_stay_nc):
    
    if t_index==shape_stay[0]-1:
        noncoastal_stayer_inputs = {
            'vt_stay_nc_input': vt_stay_nc[t_index,j_index+1, k_index, g_index, :,:, :, :],
            'vt_stay_nc_input_wf': vt_stay_nc_wf[t_index,j_index+1, k_index, g_index,:,:, :, :],
            'qt_stay_nc_input': qt_stay_nc[t_index,j_index+1, k_index, g_index, :,:, :, :],
             }
    else:
        noncoastal_stayer_inputs = {
            'vt_stay_nc_input': vt_stay_nc[t_index+1,j_index+1, k_index, g_index, :,:, :, :],
            'vt_stay_nc_input_wf': vt_stay_nc_wf[t_index+1,j_index+1, k_index, g_index,:,:, :, :],
            'qt_stay_nc_input': qt_stay_nc[t_index+1,j_index+1, k_index, g_index, :,:, :, :],
             }
    return noncoastal_stayer_inputs


@njit
def precompute_mover_inputs(shape_stay,j_index, k_index, g_index,t_index,
    vt_renter, vt_buy_c, vt_buy_nc,
    vt_renter_wf, vt_buy_c_wf, vt_buy_nc_wf,
    qt_renter, qt_buy_c, qt_buy_nc):
    
    if t_index==shape_stay[0]-1:
        mover_inputs = {
            'vt_renter_input': vt_renter[t_index,j_index+1, k_index, g_index,  :, : ],
            'vt_buy_c_input': vt_buy_c[t_index,j_index+1, k_index, g_index, :, :],
            'vt_buy_nc_input': vt_buy_nc[t_index,j_index+1, k_index, g_index,  :, : ],
            'vt_renter_input_wf': vt_renter_wf[t_index,j_index+1, k_index, g_index,  :, :],
            'vt_buy_c_input_wf': vt_buy_c_wf[t_index,j_index+1, k_index, g_index,  :, : ],
            'vt_buy_nc_input_wf': vt_buy_nc_wf[t_index,j_index+1, k_index, g_index,  :, :],
            'qt_renter_input': qt_renter[t_index,j_index+1, k_index, g_index,  :, :],
            'qt_buy_c_input': qt_buy_c[t_index,j_index+1, k_index, g_index,  :, :],
            'qt_buy_nc_input': qt_buy_nc[t_index,j_index+1, k_index, g_index,  :, : ]
        }
    else:
        mover_inputs = {
            'vt_renter_input': vt_renter[t_index+1,j_index+1, k_index, g_index,  :, : ],
            'vt_buy_c_input': vt_buy_c[t_index+1,j_index+1, k_index, g_index,  :, :],
            'vt_buy_nc_input': vt_buy_nc[t_index+1,j_index+1, k_index, g_index, :, : ],
            'vt_renter_input_wf': vt_renter_wf[t_index+1,j_index+1, k_index, g_index, :, :],
            'vt_buy_c_input_wf': vt_buy_c_wf[t_index+1,j_index+1, k_index, g_index, :, : ],
            'vt_buy_nc_input_wf': vt_buy_nc_wf[t_index+1,j_index+1, k_index, g_index, :, :],
            'qt_renter_input': qt_renter[t_index+1,j_index+1, k_index, g_index, :, :],
            'qt_buy_c_input': qt_buy_c[t_index+1,j_index+1, k_index, g_index, :, :],
            'qt_buy_nc_input': qt_buy_nc[t_index+1,j_index+1, k_index, g_index, :, : ]
        }

    return mover_inputs

@njit
def compute_p_left(grid, x, i_left):
    
    x_left = grid[i_left]
    x_right = grid[i_left + 1]
    p_left = (x_right - x) / (x_right - x_left)

    return p_left


        