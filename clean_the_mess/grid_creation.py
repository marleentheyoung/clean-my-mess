import misc_functions as misc    
import tauchen as tauch
import numpy as np
import grids as grid
from scipy.optimize import brentq

def create(par, experiment=False):
       
    # create grids
    mMarkov, vE = tauch.tauchen(par.dRho, par.dSigmaeps, par.iNumStates, par.iM, par.time_increment)
    vPi_E=tauch.initial_dist(par, vE)
    mPi_E=tauch.weight_matrix(par, vE, vPi_E, mMarkov)
    
    vE_trans=np.zeros(1)
    vE_combined=vE
    vChi = tauch.lifecycle(par,par.j_ret)
    
    #We do not use fully transitory income shocks in the end, but have a structure that allows for it below
    mMarkov_trans=np.ones((1,1))    
    #mMarkov_trans, vE_trans = tauch.tauchen(0, par.dSigmaeps_trans, par.iNumTrans, par.iM)     
    #vStationary_E = tauch.invar_dist(mMarkov)
    #vE_combined, vStationary_E_combined = tauch.combine_vectors(vE, vE_trans, vStationary_E, mMarkov_trans[0,:])    
    
    median_inc_pretax = tauch.median_inc(vChi, vE, mPi_E)  
    #For simplicity, and to prevent the income normalisation from changing with endogenous model outcomes, we ignore the mortgage rebate in the median income calculation
    median_inc=median_inc_pretax-par.tau_0*(median_inc_pretax)**(1-par.tau_1)
    min_inc=(np.exp(vChi[0] + vE[0])-par.tau_0*np.exp(vChi[0] + vE[0])**(1-par.tau_1))/median_inc
   
    #construct lti vector as function of age
    mPTI=np.zeros((par.iNj, vE.size))
    for j_index in range(par.iNj-1):
        for e_index in range(vE.size):
            mPTI[j_index,e_index]=max_mortgage_size(par, j_index, e_index, vChi, vE)/median_inc
        

    if experiment:
        vPi_S_median=par.vPi_S_median[int((2026-1998)/par.time_increment):]
    else:
        vPi_S_median=par.vPi_S_median    
    vZ= np.array([1, 0.9, 0.7, 0.3])
    vPDF_z= np.array([1, 0.4, 0.4, 0.2])


    vL_sim=np.linspace(0, 1.5, 35)
    #vH=  np.array([1.50, 1.92, 2.46, 3.15, 4.03, 5.15])
    vH=np.linspace(1.50,par.h_max,3)
    vH_renter=np.array([1.17, 1.92])
    
    vPi_L=np.ones(len(vPi_S_median))*vPi_S_median[0]
    
    
    #We want to have equal and narrow grid spacing over the range where discrete decisions vary with income. 
    #Beyond the point where the biggest house can be bought with no mortgage and value functions are essentially linear, wider grid spacing is fine
    #We use a wider cut-off than vH[-1] because of the possibility of holding cash in hand and mortgage debt simultaneously (house price does not exceed 1 under normal calibration)
    
    max_income=np.exp(np.max(vChi)+np.max(vE_combined)-np.log(median_inc))
    #vX=grid.nonlinspace_jit(min_inc, par.iBmax*(1+par.r)+max_income, par.iNb, 1.4)
    #The lowest value of vX should be s.t. the smallest rental unit remains affordable with pos consumption. 
    vX=grid.nonlinspace_jit((1-(1-par.dDelta)/(1+par.r))*vH_renter[0]+par.dPsi, par.iBmax*(1+par.r)+max_income, par.iNb, 1.4)
    #vX=grid.nonlinspace_jit((1-(1-par.dDelta)/(1+par.r))*vH_renter[0]+par.dPsi, 10, par.iNb, 1.4)
    vM=grid.nonlinspace_jit(0.01, par.iBmax*(1+par.r)+max_income, par.iNb, 1.4)
    vB=grid.nonlinspace_jit(0, par.iBmax, par.iNb, 1.4)
    vX_sim=grid.nonlinspace_jit(0, par.iBmax, par.iNb*2, 1)
    vM_sim=vX_sim #vX_sim and vM_sim are clunkily named savings grids for simulation with twice the grid points
    
    
    grids_dict = {'vB': vB,
                  "vH":  vH,
                  "vH_renter":  vH_renter,
                  'vX':vX,
                  'vX_sim':vX_sim,
                  'vM':vM,
                  'vM_sim':vM_sim,
                  "vK": np.array([0,1]), # 0 is realist, 1 is optimist,
                  'vG': np.linspace(par.dXi_min, par.dXi_max, par.iXin),
                  'vL': np.linspace(0, 1.3, 20),
                  'vL_sim': vL_sim,
                  'vE': vE,
                  'vE_trans': vE_trans, 
                  'median_inc': median_inc,
                  'median_inc_pretax': median_inc_pretax,
                  'vTime': np.arange(0,len(vPi_S_median)),
                  'vZ': vZ,   
                  'vPDF_z':vPDF_z,
                  'vChi': vChi,
                  'min_inc': min_inc,
                  #'mIncome': np.exp(vChi[:, np.newaxis, np.newaxis] + vE[np.newaxis, :, np.newaxis] + vE_trans[np.newaxis, np.newaxis, :]) /median_inc,
                  #'mIncome_pers': mIncome_pers,
                  #'vIncome_ret': mIncome_pers[par.j_ret-1,:]*0.7,
                  'median_inc': median_inc,
                  'mMarkov_trans': mMarkov_trans[0,:],
                  'vEpsilon': np.array([0,1]),
                  'vLkeps':np.linspace(0,5,2),
                  'mPTI': mPTI,
                  'vPi_S_median': vPi_S_median,
                  'vTypes': np.array([0.58, 0.42]),
                  'max_ltv': par.max_ltv, 
                  'vPi_E': vPi_E,
                  'vPi_L': vPi_L}

    grids = misc.construct_jitclass(grids_dict)
    
    return grids, mMarkov

def net_payment_frac(mortgage_size, par, j, e_index, vChi, vE):
    if j<par.j_ret:
        pretax_income_pers=np.exp(vChi[j] + vE[e_index])
    else:
        pretax_income_pers=0.7*np.exp(vChi[par.j_ret-1] + vE[e_index])   
    posttax_income=pretax_income_pers-par.tau_0*(max(pretax_income_pers-par.r_m*mortgage_size,0))**(1-par.tau_1)
    mortgage_rebate=par.tau_0*(pretax_income_pers)**(1-par.tau_1)-(posttax_income-pretax_income_pers)
    payment_next = (par.r_m*(1+par.r_m)**(par.iNj-(j+1))/((1+par.r_m)**(par.iNj-(j+1))-1))*mortgage_size-mortgage_rebate
    pti_gap=payment_next/posttax_income-par.lambda_pti
    return pti_gap


def max_mortgage_size(par, j, e_index, vChi, vE,
                      m_min=0.0, m_max=10000000):
    """
    Finds the maximum mortgage size such that
    net_payment_frac(...) <= par.pti
    """

    # If even zero mortgage violates the constraint, return 0
    if net_payment_frac(m_min, par, j, e_index, vChi, vE) > 0:
        return 0.0

    # Ensure upper bound actually violates the constraint
    if net_payment_frac(m_max, par, j, e_index, vChi, vE) < 0:
        raise ValueError("m_max too small — increase the upper search bound.")

    # Solve pti(m) = par.pti
    m_star = brentq(net_payment_frac, m_min, m_max, args=(par, j, e_index, vChi, vE))

    return m_star