


"""
solve.py

Purpose:
    Solve the model
"""
###########################################################
### Imports
import numpy as np
import numba as nb
import pandas as pd
import time
import matplotlib.pyplot as plt
from numba import njit
import misc_functions as misc
import grids as grid
import tauchen as tauch
import par_epsilons as parfile
import simulate_initial_joint as initial_joint
import household_problem_epsilons_nolearning as household_problem
import simulation as sim
import equilibrium as equil
# import equilibrium_debug as equilibrium_debug
import LoM_epsilons as lom
import quantecon as qe
import utility_epsilons as ut
import interp as interp
import buyer_problem_simulation as buy_sim
import continuation_value_nolearning as continuation_value_epsilons
import stayer_problem as stayer_problem
import stayer_problem_renter as stayer_problem_renter
import buyer_problem_epsilons as buyer_problem_epsilons
import pandas as pd
import grid_creation as grid_creation
#import error_statistics as err
import moments as mom
import proper_welfare_debug as welfare_stats
from numba import config
from scipy.stats import norm
import moments as find_moments
import experiments as experiments
import plot_creation as plot_creation
###########################################################
### main
def main():
    # import parameters
    #vCoeff_C_initial=np.array([0.72392258, 0.,         0.,         0.,         0.        ])
    #vCoeff_NC_initial=np.array([0.76693964, 0.,         0.,         0.,         0.        ])
    vCoeff_C_initial=np.array([0.69906474, 0.,         0.,         0.,         0.        ])
    vCoeff_NC_initial=np.array([0.78259554, 0.,         0.,         0.,         0.        ])
    vCoeff_C_terminal_RE=np.array([0.58952906 , 0.,0.,0.,0. ])
    vCoeff_NC_terminal_RE=np.array([0.85484033,0.,0.,0.,0.])
    vCoeff_C_terminal_HE=np.array([0.64908636, 0.,0.,0.,0. ])
    vCoeff_NC_terminal_HE=np.array([0.82124315,0.,0.,0.,0.])
    
    #vCoeff_C_initial=np.array([0.56, 0.,         0.,         0.,         0.        ])
    #vCoeff_NC_initial=np.array([0.85, 0.,         0.,         0.,         0.        ])
    #vCoeff_C_initial=np.array([0.6732231829966053, 0.,         0.,         0.,         0.        ])
    #vCoeff_NC_initial=np.array([0.7170390048443882, 0.,         0.,         0.,         0.        ])
    

    method='secant'

    par = misc.construct_jitclass(parfile.par_dict)

    
    
    
    
 
    # create grids
    
    mMarkov, vE = tauch.tauchen(par.dRho, par.dSigmaeps, par.iNumStates, par.iM, par.time_increment)
    grids, mMarkov=grid_creation.create(par)
    
    
    #Create initial guess for house prices - coastal price falls one to one with flood risk, and noncoastal rises by less than half 
    #t_cheby=(2*grids.vTime-(grids.vTime[0]+grids.vTime[-1]))/(grids.vTime[-1]-grids.vTime[0])
    #t_1=t_cheby
    #t_2=2*t_cheby**2-1
    #t_3=4*t_cheby**3-3*t_cheby
    #t_4=8*t_cheby**4-8*t_cheby**2+1
    
    #X = np.column_stack((t_1, t_2, t_3, t_4))    
    #X = np.column_stack((np.ones(len(X)), X))
    #y_c=(1-1*(par.vPi_S_median-par.vPi_S_median[0]))*vCoeff_C_initial[0]
    #beta_c = np.linalg.inv(X.T @ X) @ X.T @ y_c
    #y_nc=(1+0.25*(par.vPi_S_median-par.vPi_S_median[0]))*vCoeff_NC_initial[0]
    #beta_nc = np.linalg.inv(X.T @ X) @ X.T @ y_nc

    vCoeff_C=np.array([ 0.66335385, -0.03015386,  0.00541847,  0.00797395,  0.00249396])
    vCoeff_NC=np.array([ 0.81033554,  0.01679082, -0.00574326, -0.00115107,  0.00101112])
    #NOT CONVERGED YET 
    vCoeff_C_RE=np.array([ 0.6355361, -0.05750348,0.00171657, 0.00611094,0.00187107])
    vCoeff_NC_RE=np.array([ 0.82617263, 0.03256824, -0.00530541,-0.00385609,0.00083488])
    
    
    vCoeff_C_experiment=np.array([ 0.62190337, -0.04657477,  0.00822706,  0.00254822,  0.0029312 ])
    vCoeff_NC_experiment=np.array([ 8.36567710e-01,  2.38785227e-02, -3.96488165e-03, -4.07334828e-04, 2.94338367e-03])
    tax_equiv_C, tax_equiv_NC, tax_equiv_renter, tax_equiv_newborns=welfare_stats.find_expenditure_equiv(par,grids,mMarkov, vCoeff_C_initial, vCoeff_NC_initial, vCoeff_C, vCoeff_NC)
    print(tax_equiv_newborns)
    print(tax_equiv_C)
    print(tax_equiv_NC)
    print(tax_equiv_renter)
    """
    method='secant'
    func=False
    initial=True
    sceptics=False 
    welfare=True
    # run and save SS without welfare
    v_owner_c_wf_SS, v_owner_nc_wf_SS, v_nonowner_wf_SS, _, _, _=household_problem.solve_ss(grids, par, par.iNj, mMarkov,vCoeff_C_initial[0], vCoeff_NC_initial[0], initial, sceptics, welfare)
    vt_stay_c, vt_stay_nc, vt_renter, b_stay_c, b_stay_nc, b_renter, v_owner_c_wf, v_owner_nc_wf, v_nonowner_wf = household_problem.solve(grids, par, par.iNj, mMarkov,vCoeff_C_RE,vCoeff_NC_RE, sceptics, welfare)
  
    

    
    

    sceptics=False
    welfare_loss_total, welfare_loss_alive_C, welfare_loss_alive_NC, welfare_loss_alive_renters, welfare_loss_newborns, welfare_loss_newborns_oldweights=welfare_stats.solve(par, grids, mMarkov, vCoeff_C_initial, vCoeff_NC_initial, vCoeff_C_RE, vCoeff_NC_RE, sceptics)
    print(welfare_loss_alive_C)
    print(welfare_loss_alive_NC)
    print(welfare_loss_alive_renters)
    print(welfare_loss_newborns)
    print(welfare_loss_newborns_oldweights)
    
    initial=False
    sceptics=True
    dP_C_guess, dP_NC_guess, vCoeff_C_initial, vCoeff_NC_initial, mDist0_c, mDist0_nc, mDist0_renter, rental_stock_C0, rental_stock_NC0, coastal_beq0, noncoastal_beq0, savings_beq0, _, _ = equil.initialise_coefficients_ss(par, grids, method, par.iNj, mMarkov, vCoeff_C_terminal_HE, vCoeff_NC_terminal_HE, initial, sceptics)
    
    sceptics=False
    dP_C_vec, dP_NC_vec, vCoeff_C, vCoeff_NC, iteration, vt_stay_c, vt_stay_nc, vt_renter, b_stay_c, b_stay_nc, b_renter, vt_stay_c_wf, vt_stay_nc_wf, vt_renter_wf=equil.find_coefficients(par, grids, method, sceptics, par.iNj, mMarkov, vCoeff_C, vCoeff_NC,vCoeff_C_initial[0], vCoeff_NC_initial[0],mDist0_c, mDist0_nc, mDist0_renter, rental_stock_C0, rental_stock_NC0, coastal_beq0, noncoastal_beq0, savings_beq0)
    df = pd.DataFrame(vCoeff_C)
    df.to_excel("dP_C_vec.xslx")    
    df = pd.DataFrame(vCoeff_NC)
    df.to_excel("dP_NC_vec.xslx")

    
    func=False
    price_history, dP_C_vec_experiment, dP_NC_vec_experiment, vCoeff_C_experiment, vCoeff_NC_experiment, vt_stay_c, vt_stay_nc, vt_renter, b_stay_c, b_stay_nc, b_renter, vt_stay_c_wf, vt_stay_nc_wf, vt_renter_wf=experiments.full_information_experiment(par, func, method,  vCoeff_C, vCoeff_NC, vCoeff_C_experiment, vCoeff_NC_experiment, vCoeff_C_initial, vCoeff_NC_initial)
    df = pd.DataFrame(vCoeff_C_experiment)
    df.to_excel("dP_C_vec_experiment.xslx")    
    df = pd.DataFrame(vCoeff_NC_experiment)
    df.to_excel("dP_NC_vec_experiment.xslx")
    

    vt_stay_c, vt_stay_nc, vt_renter, b_stay_c, b_stay_nc, b_renter, vt_stay_c_wf, vt_stay_nc_wf, vt_renter_wf = household_problem.solve(grids, par, dPi_L, par.iNj, mMarkov,vCoeff_C,vCoeff_NC)
    for t_index in range(grids.vTime.size):
        plt.plot(grids.vM, vt_stay_c[t_index,5, 0,0 , :, 0, 0,2],label=f'{t_index}')        
        plt.title( 'vt, J = -5')
        plt.xlabel('Cash in hand')
        plt.ylabel('Cons')
        plt.legend(loc = 4)
        plt.show()
        plt.clf()    
        
 
       
    for l_index in range(grids.vL.size):
        plt.plot(grids.vM, vt_stay_c[28,5, 0,0 , :, 0, l_index,2],label=f'{l_index}')        
        plt.title( 'vt, J = -5')
        plt.xlabel('Cash in hand')
        plt.ylabel('Cons')
        plt.legend(loc = 4)
        plt.show()
        plt.clf()    
        
    for l_index in range(grids.vL.size):
        plt.plot(grids.vM, b_stay_c[28,5, 0,0 , :, 0, l_index,2],label=f'{l_index}')        
        plt.title( 'b, J = 5')
        plt.xlabel('Cash in hand')
        plt.ylabel('Cons')
        plt.legend(loc = 4)
        plt.show()
        plt.clf()    
        
    for l_index in range(grids.vL.size):
        plt.plot(grids.vM, vt_stay_nc[28,5, 0,0 , :, 0, l_index,2],label=f'{l_index}')        
        plt.title( 'vt, J = -5')
        plt.xlabel('Cash in hand')
        plt.ylabel('Cons')
        plt.legend(loc = 4)
        plt.show()
        plt.clf()    
        
    for l_index in range(grids.vL.size):
        plt.plot(grids.vM, b_stay_nc[28,5, 0,0 , :, 0, l_index,2],label=f'{l_index}')        
        plt.title( 'b, J = 5')
        plt.xlabel('Cash in hand')
        plt.ylabel('Cons')
        plt.legend(loc = 4)
        plt.show()
        plt.clf()   
   
    #initial_welfare, initial_welfare_vec=welfare.welfare_calculator_initial(method, par, grids, dPi_L, par.iNj, mMarkov, vCoeff_C_initial,vCoeff_NC_initial, max_ltv)
    #print(initial_welfare)
    #print(initial_welfare_vec)

    dP_C_guess, dP_NC_guess, vCoeff_C_initial, vCoeff_NC_initial, mDist0_c_initial, mDist0_nc_initial, mDist0_renter_initial, rental_stock_C0_initial, rental_stock_NC0_initial, coastal_beq0_initial, noncoastal_beq0_initial, savings_beq0_initial, no_beq, iteration=equil.initialise_coefficients_initial(par, grids, method, dPi_L, par.iNj, mMarkov, vCoeff_C_initial, vCoeff_NC_initial, np.zeros((3)))
    
    HO_C_share, HO_NC_share, R_C_share, R_NC_share, total_NW_HO_C, total_NW_HO_NC, total_NW_R, total_NW_HO, total_NW_age_9, total_NW_age_17, total_NW_all_ages, median_NW_age_9, median_NW_age_17, median_NW_all_ages, thirtythree_percentile_NW_age_17, sixtyseven_percentile_NW_age_17, tenth_percentile_housing, median_housing, ninetieth_percentile_housing, cumdens_housing_all_ages, NW_housing_share_sorted=find_moments.calc_moments(par, grids, 0, mDist0_c_initial, mDist0_nc_initial,mDist0_renter_initial, grids.vPi_S_median[0], dPi_L, vCoeff_C_initial, vCoeff_NC_initial)
  
    res=np.zeros((5))
    res[0]=(median_NW_all_ages-1.2)/1.2
    res[1]=(median_housing-0.5)/0.5
    res[2]=(median_NW_age_17/median_NW_age_9-1.51)/1.51
    res[3]=(no_beq-0.2782155)/0.2782155
    res[4]=(HO_C_share+HO_NC_share-0.66)/0.66
    
    print(total_NW_all_ages)
    print(median_NW_all_ages)
    print(median_housing)
    print(median_NW_age_17/median_NW_age_9)
    print(no_beq)
    print(HO_C_share+HO_NC_share)
    
    print("First residual:", res[0])
    print("Second residual:", res[1])
    print("Third residual", res[2])
    print("Fourth residual", res[3])
    print("Fifth residual", res[4])
    
    #dP_C_vec, dP_NC_vec, vCoeff_C, vCoeff_NC, iteration, vt_stay_c, vt_stay_nc, vt_renter, b_stay_c, b_stay_nc, b_renter, vt_stay_c_wf, vt_stay_nc_wf, vt_renter_wf=equil.find_coefficients(par, grids, method, False, dPi_L, par.iNj, mMarkov, vCoeff_C_RE, vCoeff_NC_RE,vCoeff_C_initial, vCoeff_NC_initial,mDist0_c_initial, mDist0_nc_initial, mDist0_renter_initial, rental_stock_C0_initial, rental_stock_NC0_initial, coastal_beq0_initial, noncoastal_beq0_initial, savings_beq0_initial)
    #df = pd.DataFrame(vCoeff_C)
    #df.to_excel("dP_C_vec_RE.xslx")
    #df = pd.DataFrame(vCoeff_NC)
    #df.to_excel("dP_NC_vec_RE.xslx")
    
 
 
    mDist0_c = np.zeros((par.iNj, grids.vK.size, grids.vG.size, grids.vM_sim.size, grids.vH.size, grids.vL_sim.size,grids.vE.size))
    mDist0_nc = np.zeros((par.iNj, grids.vK.size, grids.vG.size, grids.vM_sim.size, grids.vH.size, grids.vL_sim.size,grids.vE.size))
    mDist0_renter = np.zeros((par.iNj, grids.vK.size, grids.vG.size, grids.vX_sim.size, grids.vE.size))
    
    mDist0_c[:,0,:,:,:,:,:]=grids.vTypes[0]*mDist0_c_initial[:,0,:,:,:,:,:]
    mDist0_nc[:,0,:,:,:,:,:]=grids.vTypes[0]*mDist0_nc_initial[:,0,:,:,:,:,:]
    mDist0_renter[:,0,:,:,:]=grids.vTypes[0]*mDist0_renter_initial[:,0,:,:,:]
    
    mDist0_c[:,1,:,:,:,:,:]=grids.vTypes[1]*mDist0_c_initial[:,0,:,:,:,:,:]
    mDist0_nc[:,1,:,:,:,:,:]=grids.vTypes[1]*mDist0_nc_initial[:,0,:,:,:,:,:]
    mDist0_renter[:,1,:,:,:]=grids.vTypes[1]*mDist0_renter_initial[:,0,:,:,:]
    
    dP_C_vec, dP_NC_vec, vCoeff_C, vCoeff_NC, iteration, vt_stay_c, vt_stay_nc, vt_renter, b_stay_c, b_stay_nc, b_renter, vt_stay_c_wf, vt_stay_nc_wf, vt_renter_wf=equil.find_coefficients(par, grids, method, True, dPi_L, par.iNj, mMarkov, vCoeff_C, vCoeff_NC,vCoeff_C_initial, vCoeff_NC_initial,mDist0_c, mDist0_nc, mDist0_renter, rental_stock_C0_initial, rental_stock_NC0_initial, coastal_beq0_initial, noncoastal_beq0_initial, savings_beq0_initial)
    df = pd.DataFrame(vCoeff_C)
    df.to_excel("dP_C_vec_HE.xslx")
    df = pd.DataFrame(vCoeff_NC)
    df.to_excel("dP_NC_vec_HE.xslx")
   
    
    results_vector[0:5]=vCoeff_C_initial
    results_vector[5:10]=vCoeff_NC_initial
   
    sceptics=False
    t0 = time.time()
    dP_C_vec, dP_NC_vec, vCoeff_C_RE, vCoeff_NC_RE, iteration=equil.find_coefficients(par, grids, method, sceptics, dPi_L, iNj, mMarkov, vCoeff_C_RE, vCoeff_NC_RE,vCoeff_C_initial,vCoeff_NC_initial, max_ltv)
    t1 = time.time()
    print('Computation time:', t1-t0)
    results_vector[10:15]=vCoeff_C_RE
    results_vector[15:20]=vCoeff_NC_RE
    print(results_vector)
    
    vCoeff_C=np.array([  0.5959662,  -0.01149166,
  0.01003718, -0.00412208,  0.00270404])
    vCoeff_NC=np.array([  0.64859349, -0.0051976,   0.00587133,
 -0.00523878,  0.00510949])
    
    sceptics=True
    t0 = time.time()
    dP_C_vec, dP_NC_vec, vCoeff_C, vCoeff_NC, iteration=equil.find_coefficients(par, grids, method, sceptics, dPi_L, iNj, mMarkov, vCoeff_C, vCoeff_NC,vCoeff_C_initial,vCoeff_NC_initial, max_ltv)
    t1 = time.time()
    print('Computation time:', t1-t0)
    results_vector[20:25]=vCoeff_C
    results_vector[25:30]=vCoeff_NC
    print(results_vector)

    df = pd.DataFrame(results_vector)
    df.to_excel("results_vector.xlsx")
   
  
    #mDist1_c, mDist1_nc, mDist1_renter, rental_stock, coastal_beq, noncoastal_beq, savings_beq=sim.stat_dist_finder(True, grids, par, mMarkov, dPi_L, iNj, vt_stay_c[0,], vt_stay_nc[0,], vt_renter[0,], b_stay_c[0,], b_stay_nc[0,], b_renter[0,], vCoeff_C,vCoeff_NC)
    #for t_index in range(9):
    #    dP_C=lom.LoM_C(grids, t_index, vCoeff_C)
    #    dP_NC=lom.LoM_NC(grids, t_index, vCoeff_NC)
    #    func='initialise'
    #    sceptics=True
    #    mDist1_c, mDist1_nc, mDist1_renter, stock_demand_rental, coastal_beq, noncoastal_beq, savings_beq=sim.update_dist_continuous(sceptics,False, 0, func, grids, par, t_index, mMarkov, dPi_L, iNj, mDist1_c, mDist1_nc, mDist1_renter, dP_C, dP_NC, vt_stay_c[t_index,], vt_stay_nc[t_index,],  vt_renter[t_index,], b_stay_c[t_index,], b_stay_nc[t_index,], b_renter[t_index,],  coastal_beq, noncoastal_beq, savings_beq,vCoeff_C,vCoeff_NC)
        
    
   
    #plt.plot(grids.vTime, a3*100)
    #plt.show()
    
   
    plt.figure(figsize=(8,5))
    
    initial_price_c_re=lom.LoM_C(grids, 0, vCoeff_C_RE)
    initial_price_nc_re=lom.LoM_NC(grids, 0, vCoeff_NC_RE)
    initial_price_c=lom.LoM_C(grids, 0, vCoeff_C)
    initial_price_nc=lom.LoM_NC(grids, 0, vCoeff_NC)
    
    # Plot HE lines first (solid) and save their Line2D objects
    line1, = plt.plot(
        grids.vTime,
        (lom.LoM_C(grids, np.arange(grids.vTime.size), vCoeff_C)/initial_price_c - 1) * 100,
        label="Coastal housing, HE", linewidth=2
    )
    
    line2, = plt.plot(
        grids.vTime,
        (lom.LoM_NC(grids, np.arange(grids.vTime.size), vCoeff_NC)/initial_price_nc - 1) * 100,
        label="Inland housing, HE", linewidth=2
    )
    
    # Plot RE lines with the same colors as HE lines but dotted
    plt.plot(
        grids.vTime,
        (lom.LoM_C(grids, np.arange(grids.vTime.size), vCoeff_C_RE)/initial_price_c_re - 1) * 100,
        label="Coastal housing, RE", linewidth=2, linestyle=':', color=line1.get_color()
    )
    
    plt.plot(
        grids.vTime,
        (lom.LoM_NC(grids, np.arange(grids.vTime.size), vCoeff_NC_RE)/initial_price_nc_re - 1) * 100,
        label="Inland housing, RE", linewidth=2, linestyle=':', color=line2.get_color()
    )
    
    plt.xlabel("Years from initial steady state")
    plt.ylabel("Change from initial price $p_0$ (%)")
    plt.title("House price dynamics")
    
    plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    
    #par, grids, iNj, mMarkov=grid_creation.create(vCoeff_C, vCoeff_NC,vCoeff_C_initial, vCoeff_NC_initial, max_ltv, True)
    #initial_welfare, initial_welfare_vec=welfare.welfare_calculator_initial(method, par, grids, dPi_L, iNj, mMarkov, vCoeff_C, vCoeff_NC, vCoeff_C_RE, vCoeff_NC_RE, vCoeff_C_initial,vCoeff_NC_initial, max_ltv)
    #print(initial_welfare, initial_welfare_vec)
    #par, grids, iNj, mMarkov=grid_creation.create(vCoeff_C_RE, vCoeff_NC_RE,vCoeff_C_initial, vCoeff_NC_initial, max_ltv, False)
    #re_welfare, partial_welfare, re_welfare_vec, partial_welfare_mat=welfare.welfare_calculator_re(method, par, grids, dPi_L, iNj, mMarkov, vCoeff_C, vCoeff_NC, vCoeff_C_RE, vCoeff_NC_RE, vCoeff_C_initial,vCoeff_NC_initial, max_ltv)
    #print(re_welfare, partial_welfare, re_welfare_vec, partial_welfare_mat)
    #par, grids, iNj, mMarkov=grid_creation.create(vCoeff_C, vCoeff_NC,vCoeff_C_initial, vCoeff_NC_initial, max_ltv, False)
    
    #he_welfare, he_welfare_mat=welfare.welfare_calculator_he(method, par, grids, dPi_L, iNj, mMarkov, vCoeff_C, vCoeff_NC, vCoeff_C_RE, vCoeff_NC_RE, vCoeff_C_initial,vCoeff_NC_initial, max_ltv)
    #print(he_welfare, he_welfare_mat)
   
    
    
    max_ltv=0.95
    # Initialisation
    #iNk = grids.vK.size
    #iNg = grids.vG.size
    #iNAlpha = grids.vAlpha.size
    #iNb = grids.vB.size
    #iNm_sim = grids.vM_sim.size
    #iNx_sim = grids.vX_sim.size
    #iNh = grids.vH.size
    #iNe = grids.vE.size
    #iNeps = grids.vEpsilon.size
    #iNLkeps = grids.vLkeps.size
    #iNprice_nc = grids.vPrice_nc.size
    #iNprice_c = grids.vPrice_c.size
    
    #Estimation
    

        

    # excess_demand_C = np.zeros((grids.vAlpha.size))
    # excess_demand_NC= np.zeros((grids.vAlpha.size))
    # net_demand_C= np.zeros((grids.vAlpha.size))
    # net_demand_NC= np.zeros((grids.vAlpha.size))
    # coastal_beq= np.zeros((grids.vAlpha.size))
    # noncoastal_beq= np.zeros((grids.vAlpha.size))
    # investment_C= np.zeros((grids.vAlpha.size))
    # investment_NC= np.zeros((grids.vAlpha.size))
    # depreciation_C= np.zeros((grids.vAlpha.size))
    # depreciation_NC= np.zeros((grids.vAlpha.size))
    # stock_demand_rental= np.zeros((grids.vAlpha.size))
    # rental_stock= np.zeros((grids.vAlpha.size))
    
    # x_matrix = np.ones((grids.vAlpha.size, 4))
    # alpha_cheby = 2*grids.vAlpha-1
    # x_matrix[:,1] = alpha_cheby
    # x_matrix[:,2] = 2*alpha_cheby**2-1
    # x_matrix[:,3] = 4*alpha_cheby**3-3*alpha_cheby
    #vCoeff_C=np.array([0.90591673,-0.04163735,0.,0.,0.,0.,0.01602057,-0.00592333,0.,0.] ) #Initial coefficient guess
    #vCoeff_NC=np.array([0.92685039,-0.01859749,0.,0.,0.,0.,0.0173117, -0.00703509,0.,0.])
    #Simulated_eps=0
    #Simulated_Leps=0
    #Simulated_L2eps=0
    #Simulated_L3eps=0
    #Simulated_Lkeps=0  

    #t0 = time.time()
    #welfare_out=welfare.welfare_calculator(grids, method, par, dPi_L, iNj, mMarkov, vCoeff_C, vCoeff_NC)
    #df = pd.DataFrame(welfare_out)
    #df.to_excel("welfare.xlsx")
    #t1 = time.time()
    #print('Computation time:', t1-t0)
   

    #t0 = time.time()
    #dP_C_vec, dP_NC_vec, vCoeff_C, vCoeff_NC, iteration=equil_realists.find_coefficients(method, grids, par, dPi_L, iNj, mMarkov, vCoeff_C, vCoeff_NC)
    #t1 = time.time()
    #print('Computation time:', t1-t0)
    
    #t0 = time.time()
    #welfare_out=welfare.welfare_calculator(grids, method, par, dPi_L, iNj, mMarkov, vCoeff_C, vCoeff_NC)
    #df = pd.DataFrame(welfare_out)
    #df.to_excel("welfare_realists.xlsx")
    #t1 = time.time()
    #print('Computation time:', t1-t0)
    
    #t0 = time.time()
    #vt_stay_c, vt_stay_nc, vt_renter, b_stay_c, b_stay_nc, b_renter=household_problem.solve(grids, par, dPi_L, iNj, mMarkov,vCoeff_C,vCoeff_NC)
    #t1 = time.time()
    #print('Computation time:', t1-t0)
    
    #for g_index in range(grids.vG.size):
    ##    for t_index in range(grids.vTime.size):
    #        plt.plot(grids.vM, vt_stay_c[t_index,0, 0, g_index, 0,:,1,0,1],  label=f'{t_index}')
    #        #plt.plot(grids.vM, vt_stay_c_wf[0, 0, 1, g_index, 0,:,1,0,1],  label='optimist welfare')
    #        plt.title( 'Coastal')
    #        plt.xlabel('Cash in hand')
    #        plt.ylabel('Welfare')
    #        plt.legend(loc = 4)
    #    plt.show()
    #    plt.clf()
    #for g_index in range(grids.vG.size):
    #    for t_index in range(grids.vTime.size): 
    #        plt.plot(grids.vM, vt_stay_nc[t_index, 0, 0, g_index, 0,:,1,0,1], label=f'{t_index}')    
    #        #plt.plot(grids.vM, vt_stay_nc_wf[0, 0, 1, g_index, 0,:,1,0,1], label = 'optimist welfare')        
    #        plt.title('noncoastal')
    #        plt.xlabel('Cash in hand')
    #        plt.ylabel('Welfare')
    #        plt.legend(loc = 4)
    #    plt.show()
    #    plt.clf()   
    
   
    
    
    vt_stay_c, vt_stay_nc, vt_renter, b_stay_c, b_stay_nc, b_renter = household_problem.solve_initial(grids, par, dPi_L, par.iNj, mMarkov, vCoeff_C_initial[0],vCoeff_NC_initial[0])
    plt.plot(grids.vM, vt_stay_c[0,0, 0, 1, :, 1, 0,3],  label='coastal, L = 0, H = 1')
    plt.title( 'vt, J = -5')
    plt.xlabel('Cash in hand')
    plt.ylabel('Cons')
    plt.legend(loc = 4)
    plt.show()
    plt.clf()
    
    plt.plot(grids.vM, vt_stay_nc[0,0, 0,1 , :, 1, 0,3], label = 'noncoastal, L = 0, H = 1')        
    plt.title( 'vt, J = -5')
    plt.xlabel('Cash in hand')
    plt.ylabel('Cons')
    plt.legend(loc = 4)
    plt.show()
    plt.clf()    
    
    plt.plot(grids.vM, vt_stay_c[0,0, 0, 1, :, 1, -5,3],  label='coastal, L = -5, H = 1')
    plt.title( 'vt, J = -5')
    plt.xlabel('Cash in hand')
    plt.ylabel('Cons')
    plt.legend(loc = 4)
    plt.show()
    plt.clf()
    
    plt.plot(grids.vM, vt_stay_nc[0,0, 0,1 , :, 1, -5,3], label = 'noncoastal, L = -5, H = 1')        
    plt.title( 'vt, J = -5')
    plt.xlabel('Cash in hand')
    plt.ylabel('Cons')
    plt.legend(loc = 4)
    plt.show()
    plt.clf()    

    for l_index in range(grids.vL.size):
        plt.plot(grids.vM, b_stay_nc[0,-1,0,0,:,0,l_index,0], label=f'{l_index}')
    plt.legend(loc = 4)
    plt.show()
    plt.clf()  
    
    plt.plot(grids.vM, w_c_vE[:,0,0])
    plt.legend(loc = 4)
    plt.show()
    plt.clf() 
    
    plt.plot(grids.vM, q_c_vE[:,0,0])
    plt.legend(loc = 4)
    plt.show()
    plt.clf() 
    
    plt.plot(grids.vM, q_c_vE[:,0,-5])
    plt.legend(loc = 4)
    plt.show()
    plt.clf() 
    
        
    plt.plot(grids.vM, q_c_vE[:,-1,0])
    plt.legend(loc = 4)
    plt.show()
    plt.clf() 
    
    plt.plot(grids.vM, w_nc_vE[:,0,0])
    plt.legend(loc = 4)
    plt.show()
    plt.clf()  
    
    plt.plot(grids.vM, q_nc_vE[:,0,0])
    plt.legend(loc = 4)
    plt.show()
    plt.clf() 
    
    plt.plot(grids.vM, q_nc_vE[:,0,-5])
    plt.legend(loc = 4)
    plt.show()
    plt.clf() 

    plt.plot(grids.vM, q_nc_vE[:,-1,0])
    plt.legend(loc = 4)
    plt.show()
    plt.clf() 
        
    plt.plot(grids.vM, b_stay_nc[0,-1,0,0,:,0,0,0])
    plt.legend(loc = 4)
    plt.show()
    plt.clf()  
    
    plt.plot(grids.vM, b_stay_nc[0,-1,0,0,:,0,-5,0])
    plt.legend(loc = 4)
    plt.show()
    plt.clf()  
    
    for l_index in range(grids.vL.size):
        plt.plot(grids.vM, vt_stay_nc[0,-1,0,0,:,0,l_index,0], label=f'{l_index}')
    plt.legend(loc = 4)
    plt.show()
    plt.clf()  
    
    for l_index in range(grids.vL.size):
        plt.plot(grids.vM, vt_stay_nc[0,-1,0,0,:,0,0,0])
    plt.legend(loc = 4)
    plt.show()
    plt.clf()  
    
    plt.plot(grids.vM, vt_stay_nc[0,-1,0,0,:,0,-5,0])
    plt.legend(loc = 4)
    plt.show()
    plt.clf()  
  
    initialise=True
    #t0 = time.time()
    mDist1_c, mDist1_nc, mDist1_renter, rental_stock_C, rental_stock_NC, coastal_beq, noncoastal_beq, savings_beq, vcoastal_beq, vnoncoastal_beq, vsavings_beq, no_beq=sim.stat_dist_finder(False, grids, par, mMarkov, dPi_L, par.iNj, vt_stay_c[0,], vt_stay_nc[0,], vt_renter[0,], b_stay_c[0,], b_stay_nc[0,], b_renter[0,], vCoeff_C_initial,vCoeff_NC_initial, np.zeros((3)))
    #t1 = time.time()
    HO_C_share, HO_NC_share, R_C_share, R_NC_share, total_NW_HO_C, total_NW_HO_NC, total_NW_R, total_NW_HO, total_NW_age_15, total_NW_age_27, total_NW_all_ages, median_NW_age_15, median_NW_age_27, median_NW_all_ages, thirtythree_percentile_NW_age_27, sixtyseven_percentile_NW_age_27, tenth_percentile_housing, median_housing, ninetieth_percentile_housing, cumdens_housing_all_ages, NW_housing_share_sorted=find_moments.calc_moments(par, grids, 0, mDist1_c, mDist1_nc,mDist1_renter, grids.vPi_S_median[0], dPi_L, vCoeff_C_initial, vCoeff_NC_initial)
    res=np.zeros((3))
    res[0]=(median_NW_all_ages-1.2)/1.2
    res[1]=(median_housing-0.5)/0.5
    res[2]=(median_NW_age_27/median_NW_age_15-1.51)/1.51
    #print(HO_C_share, HO_NC_share, R_C_share, R_NC_share, total_NW_HO_C, total_NW_HO_NC, total_NW_R, total_NW_HO, total_NW_age_15, total_NW_age_27, total_NW_all_ages, median_NW_age_15, median_NW_age_27, median_NW_all_ages, thirtythree_percentile_NW_age_27, sixtyseven_percentile_NW_age_27, tenth_percentile_housing, median_housing, ninetieth_percentile_housing)
    print("First residual:", res[0])
    print("Second residual:", res[1])
    print("Third residual", res[2])

    
    #print('Computation time:', t1-t0)
    print("Fraction no bequests:", no_beq)
    print('Bequests:', coastal_beq, noncoastal_beq, savings_beq)
   # print("Bequest vector:", vcoastal_beq, vnoncoastal_beq, vsavings_beq)
    print(cumdens_housing_all_ages)
    print(NW_housing_share_sorted)
   
    plt.plot(cumdens_housing_all_ages, NW_housing_share_sorted)
    plt.title('distribution over housing share')
    plt.legend()
    plt.show()
    plt.clf()
   
    initialise=True
    dP_C=lom.LoM_C(grids, 0,vCoeff_C_initial)
    dP_NC=lom.LoM_NC(grids, 0,vCoeff_NC_initial)
    excess_demand_C_stock, excess_demand_NC_stock, net_demand_C, net_demand_NC, investment_C, investment_NC, stock_demand_rental_C, rental_stock_C, stock_demand_rental_NC, rental_stock_NC=sim.excess_demand_continuous(False, initialise, grids, par,0, mMarkov, dPi_L, par.iNj, mDist1_c, mDist1_nc, mDist1_renter, dP_C, dP_NC,  vt_stay_c[0,], vt_stay_nc[0,], vt_renter[0,], b_stay_c[0,],  b_stay_nc[0,], b_renter[0,], rental_stock_C, rental_stock_NC, coastal_beq, noncoastal_beq, savings_beq,vCoeff_C_initial,vCoeff_NC_initial,vCoeff_C_initial[0],vCoeff_NC_initial[0])
    print(net_demand_C,coastal_beq)
    print(net_demand_NC,noncoastal_beq)
    print(stock_demand_rental_C, rental_stock_C)
    print(stock_demand_rental_NC, rental_stock_NC)
    plt.plot(grids.vX, b_renter[0,-1,0,0,:,0])
    
    sum_mDist1_c = np.sum(mDist1_c, axis=6)
    sum_mDist1_c = np.sum(sum_mDist1_c, axis=4)
    sum_mDist1_c = np.sum(sum_mDist1_c, axis=3)
    sum_mDist1_c = np.sum(sum_mDist1_c, axis=2)
    sum_mDist1_c = np.sum(sum_mDist1_c, axis=1)
   
    
    sum_mDist1_nc = np.sum(mDist1_nc, axis=6)
    sum_mDist1_nc = np.sum(sum_mDist1_nc, axis=4)
    sum_mDist1_nc = np.sum(sum_mDist1_nc, axis=3)
    sum_mDist1_nc = np.sum(sum_mDist1_nc, axis=2)
    sum_mDist1_nc = np.sum(sum_mDist1_nc, axis=1)

    
    sum_renters=np.sum(mDist1_renter, axis=4)
    sum_renters=np.sum(sum_renters, axis=3)
    sum_renters=np.sum(sum_renters, axis=2)
    sum_renters=np.sum(sum_renters, axis=1)
    
    summed_mortgages=np.sum(sum_mDist1_c, axis = 0)
    plt.figure(figsize=(8,5))
    plt.plot(
        grids.vL_sim[1:], 
        summed_mortgages[1:] / np.sum(summed_mortgages[1:]),
        linewidth=2.5,
        label="Coastal"
    )
    
    plt.title("Distribution over mortgage loan size", fontsize=14)
    plt.xlabel("Loan to equity ratio", fontsize=12)
    plt.ylabel("Conditional probability density", fontsize=12)
    
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.show()
    plt.clf()
    
    for j in range(par.iNj):
        plt.plot(grids.vL_sim, sum_mDist1_c[j,:],label='coastal')
        plt.plot(grids.vL_sim, sum_mDist1_nc[j,:],label='noncoastal')   
        plt.title(f'Probability Mass Function by Age Group (j = {j})')
        plt.xlabel('Loan to equity ratio')
        plt.ylabel('Probability Mass')
    
        # Add legend and annotation for clarity
        plt.legend()
        plt.text(0.05, 0.95,
        'Note: Each point shows discrete probability mass.\n'
        'Heights sum to 1 across all points.',
        transform=plt.gca().transAxes,
        fontsize=8, va='top', ha='left')           
        # Show and clear plot for next iteration
        plt.show()
        plt.clf()

    #print("Alpha=",alpha)
    #print(sum_mDist1_c/sumsum_mDist1_c,sum_mDist1_nc/sumsum_mDist1_nc)
    #print(sumsum_mDist1_c,sumsum_mDist1_nc)
   
    print("Total mass:", np.sum(sum_mDist1_c)+np.sum(sum_mDist1_nc)+np.sum(sum_renters))
    plt.plot(grids.vL_sim, np.sum(sum_mDist1_c, axis = 0),label='coastal')
    plt.plot(grids.vL_sim, np.sum(sum_mDist1_nc, axis = 0),label='noncoastal')
    plt.title('mortgage mass, all ages')
    plt.legend()
    plt.show()
    plt.clf()
    
    gridsize_left=grids.vM_sim[1]-grids.vM_sim[0]
    for m_index_sim in range(0,grids.vM_sim.size-1):
        gridsize_right=0.5*(grids.vM_sim[m_index_sim+1]-grids.vM_sim[m_index_sim])
        mDist1_c[:,:,:,m_index_sim,:,:,:]=mDist1_c[:,:,:,m_index_sim,:,:,:]/(gridsize_left+gridsize_right)
        mDist1_nc[:,:,:,m_index_sim,:,:,:]=mDist1_nc[:,:,:,m_index_sim,:,:,:]/(gridsize_left+gridsize_right)
        gridsize_left=gridsize_right
    mDist1_c[:,:,:,-1,:,:,:]=mDist1_c[:,:,:,-1,:,:,:]/(gridsize_right+gridsize_left)
    
    gridsize_left=grids.vX_sim[1]-grids.vX_sim[0]
    for x_index_sim in range(0,grids.vX_sim.size-1):
        gridsize_right=0.5*(grids.vX_sim[x_index_sim+1]-grids.vX_sim[x_index_sim])
        mDist1_renter[:,:,:,x_index_sim,:]=mDist1_renter[:,:,:,x_index_sim,:]/(gridsize_left+gridsize_right)
        gridsize_left=gridsize_right
    mDist1_renter[:,:,:,-1,:]=mDist1_renter[:,:,:,-1,:]/(gridsize_right+gridsize_left)
    

    
    #sum_mDist1_nc = np.sum(sum_mDist1_nc, axis=1)
    #sum_mDist1_nc = np.sum(sum_mDist1_nc, axis=0)
        
    #sum_mDist1_c = np.sum(sum_mDist1_c, axis=1)
    #sum_mDist1_c = np.sum(sum_mDist1_c, axis=0)
    
    sum_mDist1_c = np.sum(mDist1_c, axis=6)
    sum_mDist1_c = np.sum(sum_mDist1_c, axis=5)
    sum_mDist1_c = np.sum(sum_mDist1_c, axis=4)
    sum_mDist1_c = np.sum(sum_mDist1_c, axis=2)
    sum_mDist1_c = np.sum(sum_mDist1_c, axis=1)
    
    
    #TRANSFORM INTO DENSITY
    #for m_index_sim in range(grids.vM_sim.size):
    #     if m_index_sim==0:
    #         sum_mDist1_c[:,m_index_sim]=sum_mDist1_c[:,m_index_sim]/(0.5*(grids.vM_sim[m_index_sim+1]-grids.vM_sim[m_index_sim]))
    #     elif m_index_sim==grids.vM_sim.size-1:
    #         sum_mDist1_c[:,m_index_sim]=sum_mDist1_c[:,m_index_sim]/(0.5*(grids.vM_sim[m_index_sim]-grids.vM_sim[m_index_sim-1]))
    #     else: 
    #         sum_mDist1_c[:,m_index_sim]=sum_mDist1_c[:,m_index_sim]/(0.5*(grids.vM_sim[m_index_sim]-grids.vM_sim[m_index_sim-1])+0.5*(grids.vM_sim[m_index_sim+1]-grids.vM_sim[m_index_sim]))
    
    sum_mDist1_nc = np.sum(mDist1_nc, axis=6)
    sum_mDist1_nc = np.sum(sum_mDist1_nc, axis=5)
    sum_mDist1_nc = np.sum(sum_mDist1_nc, axis=4)
    sum_mDist1_nc = np.sum(sum_mDist1_nc, axis=2)
    sum_mDist1_nc = np.sum(sum_mDist1_nc, axis=1)
    
    #TRANSFORM INTO DENSITY
    #for m_index_sim in range(grids.vM_sim.size):
    #     if m_index_sim==0:
    #         sum_mDist1_nc[:,m_index_sim]=sum_mDist1_nc[:,m_index_sim]/(0.5*(grids.vM_sim[m_index_sim+1]-grids.vM_sim[m_index_sim]))
    #     elif m_index_sim==grids.vM_sim.size-1:
    #         sum_mDist1_nc[:,m_index_sim]=sum_mDist1_nc[:,m_index_sim]/(0.5*(grids.vM_sim[m_index_sim]-grids.vM_sim[m_index_sim-1]))
    #     else: 
    #         sum_mDist1_nc[:,m_index_sim]=sum_mDist1_nc[:,m_index_sim]/(0.5*(grids.vM_sim[m_index_sim]-grids.vM_sim[m_index_sim-1])+0.5*(grids.vM_sim[m_index_sim+1]-grids.vM_sim[m_index_sim]))

    
    sum_renters=np.sum(mDist1_renter, axis=4)
    sum_renters=np.sum(sum_renters, axis=2)
    sum_renters=np.sum(sum_renters, axis=1)
  
    #for x_index_sim in range(grids.vX_sim.size):
    #     if x_index_sim==0:
    #         sum_renters[:,x_index_sim]=sum_renters[:,x_index_sim]/(0.5*(grids.vX_sim[x_index_sim+1]-grids.vX_sim[x_index_sim]))
    #     elif x_index_sim==grids.vX_sim.size-1:
    #         sum_renters[:,x_index_sim]=sum_renters[:,x_index_sim]/(0.5*(grids.vX_sim[x_index_sim]-grids.vX_sim[x_index_sim-1]))
    #     else: 
    #         sum_renters[:,x_index_sim]=sum_renters[:,x_index_sim]/(0.5*(grids.vX_sim[x_index_sim]-grids.vX_sim[x_index_sim-1])+0.5*(grids.vX_sim[x_index_sim+1]-grids.vX_sim[x_index_sim]))


    
    
    
    plt.plot(grids.vM_sim, np.sum(sum_mDist1_c, axis = 0),label='coastal')
    plt.plot(grids.vM_sim, np.sum(sum_mDist1_nc, axis = 0),label='noncoastal')
    plt.plot(grids.vX_sim, np.sum(sum_renters, axis = 0),label='renters')
    plt.title('cih mass, all ages')
    plt.legend()
    plt.show()
    plt.clf()
    

    summed_savings = np.sum(sum_mDist1_c, axis=0)
    
    plt.figure(figsize=(8,5))
    plt.plot(
        grids.vM_sim, 
        summed_savings / np.sum(summed_savings),
        linewidth=2.5,
        label="Coastal"
    )
    
    plt.title("Distribution over savings", fontsize=14)
    plt.xlabel("Savings", fontsize=12)
    plt.ylabel("Conditional probability density", fontsize=12)
    
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.show()
    plt.clf()
    
    for j in range(par.iNj):
        plt.plot(grids.vM_sim, sum_mDist1_c[j,:]*par.iNj,label='coastal')
        plt.plot(grids.vM_sim, sum_mDist1_nc[j,:]*par.iNj,label='noncoastal')
        plt.plot(grids.vX_sim, sum_renters[j,:]*par.iNj,label='renters')
        plt.title(f'Probability Mass Function by Age Group (j = {j})')
        plt.xlabel('Cash in hand value')
        plt.ylabel('Probability Mass')
    
        # Add legend and annotation for clarity
        plt.legend()
        plt.text(0.05, 0.95,
        'Note: Each point shows discrete probability mass.\n'
        'Heights sum to 1 across all points.',
        transform=plt.gca().transAxes,
        fontsize=8, va='top', ha='left')           
        # Show and clear plot for next iteration
        plt.show()
        plt.clf()
    


   
 
 
    #func='initialise'    
    #excess_demand_C_stock, excess_demand_NC_stock, net_demand_C, net_demand_NC, investment_C, investment_NC, stock_demand_rental, rental_stock=sim.excess_demand_continuous(func, grids, par,0, mMarkov, dPi_L, iNj, mDist1_c, mDist1_nc, mDist1_renter, dP_C, dP_NC,  vt_stay_c[0,], vt_stay_nc[0,], vt_renter[0,], b_stay_c[0,],  b_stay_nc[0,], b_renter[0,], rental_stock, coastal_beq, noncoastal_beq, savings_beq,vCoeff_C,vCoeff_NC)
    #print(net_demand_C,coastal_beq)
    #print(net_demand_NC,noncoastal_beq)
    #print(stock_demand_rental, rental_stock)
    
    #dP_C_guess, dP_NC_guess, vCoeff_C, vCoeff_NC, iteration=equil.initialise_coefficients(method, grids, par, dPi_L, dPi_S, iNj, mMarkov, vCoeff_C, vCoeff_NC)
    #dP_C_vec, dP_NC_vec, vCoeff_C, vCoeff_NC, iteration=equil.find_coefficients(method, grids, par, dPi_L, dPi_S, iNj, mMarkov, vCoeff_C, vCoeff_NC)
    
    print('price C lom',dP_C_lom)
    print('price NC lom',dP_NC_lom)#vCoeff_in_C = vCoeff_C.copy()
    

    beta_C, beta_NC, r_sqrd_C, r_sqrd_NC, demand_error_C, demand_error_NC=err.prediction_errors(method, grids, par, dPi_L, dPi_S, iNj, mMarkov, vCoeff_C, vCoeff_NC)
    print(beta_C, beta_NC, r_sqrd_C, r_sqrd_NC, demand_error_C, demand_error_NC)

    
    t0 = time.time()
    dP_C_guess, dP_NC_guess, vCoeff_C, vCoeff_NC, iteration = equil.initialise_coefficients(method, grids, par, dPi_L, dPi_S, iNj, mMarkov, vCoeff_C, vCoeff_NC)
    t1 = time.time()
    print("Initialisation time", t1-t0)
    print(vCoeff_C, vCoeff_NC)
    t0 = time.time()
    dP_C_vec, dP_NC_vec, vCoeff_C, vCoeff_NC, iteration = equil.find_coefficients(method, grids, par, dPi_L, dPi_S, iNj, mMarkov, vCoeff_C, vCoeff_NC)
    t1 = time.time()
    print("Finding time", t1-t0)
    print(vCoeff_C, vCoeff_NC)
   
    #w_c, w_nc, w_renter, vt_buy_c,vt_stay_c, vt_buy_nc,vt_stay_nc, vt_stay_renter, c_c, c_nc, c_renter, h_c,h_nc, l_c, l_nc, h_renter, b_c_stay, b_nc_stay, b_c_buy, b_nc_buy, b_renter = household_problem_epsilons.solve(grids, par, dPi_L, dPi_S, iNj, mMarkov,vCoeff_C,vCoeff_NC)
    plt.plot(
        grids.vAlpha,
        100 * (dP_C_lom - lom.LoM_C(1, 0, 0, 0, 0, vCoeff_C)) / lom.LoM_C(1, 0, 0, 0, 0, vCoeff_C),
        label="Coastal",
        linestyle="-",
        linewidth=2
    )
    plt.plot(
        grids.vAlpha,
        100 * (dP_NC_lom - lom.LoM_NC(1, 0, 0, 0, 0, vCoeff_NC)) / lom.LoM_NC(1, 0, 0, 0, 0, vCoeff_NC),
        label="Non-coastal",
        linestyle="--",
        linewidth=2
    )
    
    # Axis labels with Greek alpha
    plt.xlabel(r"Belief weight $\alpha$", fontsize=12)
    plt.ylabel("Value relative to RE, %", fontsize=12)
    
    # Title (optional)
    #plt.title("Relative Value vs. Belief Parameter", fontsize=14)
    
    # Add legend, grid, and style
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    
    plt.show()
    
    dP_C_lom_vec=np.zeros((11))
    dP_NC_lom_vec=np.zeros((11)) 
    price_drop_c_vec=np.zeros((11))
    price_drop_nc_vec=np.zeros((11))
    price_drop_c_vec_belief=np.zeros((11))
    price_drop_nc_vec_belief=np.zeros((11))
    price_drop_c_vec_disr=np.zeros((11))
    price_drop_nc_vec_disr=np.zeros((11))
    a_vec=np.zeros((11)) 
    for a_ten in range(11):
        a_vec[a_ten]=a_ten*0.1
        a=a_ten*0.1
        a_prime_flood =(a*dPi_S)/(a*dPi_S+(1-a)*dPi_L)
        a_prime_noflood = (a*(1-dPi_S))/(a*(1-dPi_S)+(1-a)*(1-dPi_L))
        print(a_prime_flood-a)
        dP_C_lom_vec[a_ten]=lom.LoM_C(a, 0, 0, 0, 0,vCoeff_C)
        dP_NC_lom_vec[a_ten]=lom.LoM_NC(a, 0, 0, 0, 0,vCoeff_C)
        price_drop_c_vec[a_ten]=100*(lom.LoM_C(a_prime_flood, 1, 0, 0, 0,vCoeff_C)-lom.LoM_C(a, 0, 0, 0, 0,vCoeff_C))/lom.LoM_C(a, 0, 0, 0, 0,vCoeff_C)
        price_drop_nc_vec[a_ten]=100*(lom.LoM_NC(a_prime_flood, 1, 0, 0, 0,vCoeff_NC)-lom.LoM_NC(a, 0, 0, 0, 0,vCoeff_NC))/lom.LoM_NC(a, 0, 0, 0, 0,vCoeff_NC)
        price_drop_c_vec_belief[a_ten]=100*(lom.LoM_C(a_prime_flood, 0, 0, 0, 0,vCoeff_C)-lom.LoM_C(a, 0, 0, 0, 0,vCoeff_C))/lom.LoM_C(a, 0, 0, 0, 0,vCoeff_C)
        price_drop_nc_vec_belief[a_ten]=100*(lom.LoM_NC(a_prime_flood, 0, 0, 0, 0,vCoeff_NC)-lom.LoM_NC(a, 0, 0, 0, 0,vCoeff_NC))/lom.LoM_NC(a, 0, 0, 0, 0,vCoeff_NC)
        price_drop_c_vec_disr[a_ten]=100*(lom.LoM_C(a, 1, 0, 0, 0,vCoeff_C)-lom.LoM_C(a, 0, 0, 0, 0,vCoeff_C))/lom.LoM_C(a, 0, 0, 0, 0,vCoeff_C)
        price_drop_nc_vec_disr[a_ten]=100*(lom.LoM_NC(a, 1, 0, 0, 0,vCoeff_NC)-lom.LoM_NC(a, 0, 0, 0, 0,vCoeff_NC))/lom.LoM_NC(a, 0, 0, 0, 0,vCoeff_NC)
    # Coastal (C) lines in blue
    # Plot Coastal (C) in blue
    plt.plot(a_vec, price_drop_c_vec, color='C0', linewidth=2)
    plt.plot(a_vec, price_drop_c_vec_disr, color='C0', linestyle="--", linewidth=2)
    
    # Plot Non-Coastal (NC) in orange
    plt.plot(a_vec, price_drop_nc_vec, color='C1', linewidth=2)
    plt.plot(a_vec, price_drop_nc_vec_disr, color='C1', linestyle="--", linewidth=2)
    
    # Axis labels
    plt.xlabel(r"Belief weight $\alpha$", fontsize=12)
    plt.ylabel("Price drop upon flooding, %", fontsize=12)
    #plt.title("Price Drop Decomposition", fontsize=14)
    
    # Custom legend with black lines for style only
    custom_lines = [
        Line2D([0], [0], color='k', linewidth=2, linestyle='-'),   # Total
        Line2D([0], [0], color='k', linewidth=2, linestyle='--')   # Disruption
    ]
    plt.legend(custom_lines, ['Total', 'Disruption'])
    
    # Grid and layout
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    
    plt.show()
    #vCoeff_in_NC= vCoeff_NC.copy()
    #print('price C lom', lom.LoM_C(grids.vAlpha, 0, 0, 0, 0,vCoeff_C))
    #print('price NC lom', lom.LoM_NC(grids.vAlpha, 0, 0, 0, 0,vCoeff_NC))    # # print(vCoeff_C)
    # print(vCoeff_NC)
    #t0 = time.time()
    # w_c, w_nc, w_renter, vt_buy_c,vt_stay_c, vt_buy_nc,vt_stay_nc, vt_stay_renter, c_c, c_nc, c_renter, h_c,h_nc, l_c, l_nc, h_renter, b_c_stay, b_nc_stay, b_c_buy, b_nc_buy, b_renter = household_problem_epsilons.solve(grids, par, dPi_L, dPi_S, iNj, mMarkov, vCoeff_C, vCoeff_NC)
    # mDist0_c, mDist0_nc, mDist0_renter, rental_stock0, coastal_beq0, noncoastal_beq0, savings_beq0 = sim.stat_dist_finder(grids, par, mMarkov, dPi_S, dPi_L, iNj, vt_stay_nc, vt_stay_c, vt_stay_renter, b_c_stay, b_renter, b_nc_stay,vCoeff_C,vCoeff_NC,0.4)
    
    # paths =1
    # iT = 6
    # datasets = np.zeros((iT, 8, paths))
    # initial_alpha = 0.4
    # func = 'find'
    # method = 'secant'
    
    # for k in range(paths): 
    #     vSimulated_eps, vSimulated_alpha, vEps_k = sim.create_path(par, iT, dPi_S, dPi_L, initial_alpha)
    #     # print(vSimulated_eps)
    #     datasets[:,2,k]  = vSimulated_eps
    #     datasets[1:,3,k] = vSimulated_eps[:-1]
    #     datasets[2:,4,k] = vSimulated_eps[:-2]
    #     datasets[3:,5,k] = vSimulated_eps[:-3]
    #     datasets[:,6,k]  = vEps_k
    #     datasets[:,7,k]  = vSimulated_alpha
    #     # for each path, calculate market clearing prices
    #     for t in range(iT):
    #         # datasets[t,2,k] = int(datasets[t,2,k])
    #         # datasets[t,3,k] = int(datasets[t,3,k])
    #         # datasets[t,4,k] = int(datasets[t,4,k])
    #         # datasets[t,5,k] = int(datasets[t,5,k])
            
    #         #Update guess and bounds
    #         guess_c = lom.LoM_C(datasets[t,7,k], 0, 0, 0, 0,vCoeff_C)
    #         guess_nc =  lom.LoM_NC(datasets[t,7,k], 0, 0, 0, 0,vCoeff_NC)
    #         bound_c_l= 0.1
    #         bound_nc_l= 0.1 
            
    #         bound_c_l_bis=guess_c-0.25
    #         bound_c_r_bis=guess_c+0.25
    #         bound_nc_l_bis=guess_nc-0.25
    #         bound_nc_r_bis=guess_nc+0.25             
        
    #         datasets[t,0,k], datasets[t,1,k], it, succes = equil.house_prices_algorithm(func, method, grids, par, guess_c, guess_nc, bound_c_l, bound_nc_l, bound_c_l_bis, bound_nc_l_bis, bound_c_r_bis, bound_nc_r_bis, mMarkov, dPi_S, dPi_L, iNj, mDist0_c, mDist0_nc, mDist0_renter, vt_stay_nc, vt_stay_c, vt_stay_renter, h_renter, b_c_stay, b_renter, b_nc_stay, rental_stock0, coastal_beq0, noncoastal_beq0, savings_beq0,vCoeff_C, vCoeff_NC,datasets[t,7,k], int(datasets[t,2,k]), int(datasets[t,3,k]),int(datasets[t,4,k]),int(datasets[t,5,k]),datasets[t,6,k])
    #         mDist1_c, mDist1_nc, mDist1_renter, stock_demand_rental1, coastal_beq1, noncoastal_beq1, savings_beq1 = sim.update_dist_continuous(datasets[t,7,k], int(datasets[t,2,k]), int(datasets[t,3,k]),int(datasets[t,4,k]),int(datasets[t,5,k]),datasets[t,6,k], grids, par, mMarkov, dPi_S, dPi_L, iNj, mDist0_c, mDist0_nc, mDist0_renter, datasets[t,0,k], datasets[t,1,k], vt_stay_nc, vt_stay_c, vt_stay_renter, b_c_stay, b_renter, b_nc_stay, coastal_beq0, noncoastal_beq0, savings_beq0,vCoeff_C,vCoeff_NC)
            
    #         mDist0_c  = (mDist1_c)
    #         mDist0_nc = (mDist1_nc)
    #         mDist0_renter = (mDist1_renter)
    #         rental_stock0= (stock_demand_rental1)
    #         coastal_beq0 = (coastal_beq1)
    #         noncoastal_beq0  = (noncoastal_beq1)
    #         savings_beq0 = (savings_beq1)
    #         excess_demand_C_flow, excess_demand_NC_flow, net_demand_C, net_demand_NC, investment_C, investment_NC, stock_demand_rental, rental_stock = sim.excess_demand_continuous(func, datasets[t,7,k], int(datasets[t,2,k]), int(datasets[t,3,k]),int(datasets[t,4,k]),int(datasets[t,5,k]),datasets[t,6,k], grids, par, mMarkov, dPi_S, dPi_L, iNj, mDist1_c, mDist1_nc, mDist1_renter, datasets[t,0,k], datasets[t,1,k], vt_stay_nc, vt_stay_c, vt_stay_renter, h_renter, b_c_stay, b_renter, b_nc_stay,rental_stock0, coastal_beq1, noncoastal_beq1, savings_beq1,vCoeff_C,vCoeff_NC)
    #         print('epsilon, alpha, dP_C, dP_NC:', datasets[t,2,k], datasets[t,7,k], datasets[t,0,k], datasets[t,1,k], np.sum(mDist0_c) + np.sum(mDist0_nc) + np.sum(mDist0_renter))
    #         print('excess flow demand C and NC:', excess_demand_C_flow, excess_demand_NC_flow)
    #         print('net demand C, net demand NC', net_demand_C, net_demand_NC )
            
    #         dP_C_lom, dP_NC_lom =  datasets[t,0,k], datasets[t,1,k]
            
            
    #         if t == 2:
    #             for i in np.arange(-0.001, 0.0010, 0.0001):
    #                 dP_C=dP_C_lom+i
    #                 dP_NC=dP_NC_lom
    #                 excess_demand_C_stock, excess_demand_NC_stock, net_demand_C, net_demand_NC, investment_C, investment_NC, stock_demand_rental, rental_stock=sim.excess_demand_continuous(func, datasets[t,7,k], int(datasets[t,2,k]), int(datasets[t,3,k]),int(datasets[t,4,k]),int(datasets[t,5,k]),datasets[t,6,k], grids, par, mMarkov, dPi_S, dPi_L, iNj, mDist1_c, mDist1_nc, mDist1_renter, dP_C, dP_NC, vt_stay_nc, vt_stay_c, vt_stay_renter, h_renter, b_c_stay, b_renter, b_nc_stay,rental_stock, coastal_beq1, noncoastal_beq1, savings_beq1,vCoeff_C,vCoeff_NC)
                
    #                 print('dP_C, excess deam C, excess dem NC ', dP_C, excess_demand_C_stock, excess_demand_NC_stock)
    #                 print('dP_C, net dem C, net dem NC ',dP_C, net_demand_C, net_demand_NC)
                
    #             for i in np.arange(-0.001, 0.0010, 0.0001):
    #                 dP_C=dP_C_lom
    #                 dP_NC=dP_NC_lom+i
    #                 excess_demand_C_stock, excess_demand_NC_stock, net_demand_C, net_demand_NC, investment_C, investment_NC, stock_demand_rental, rental_stock=sim.excess_demand_continuous(func, datasets[t,7,k], int(datasets[t,2,k]), int(datasets[t,3,k]),int(datasets[t,4,k]),int(datasets[t,5,k]),datasets[t,6,k], grids, par, mMarkov, dPi_S, dPi_L, iNj, mDist1_c, mDist1_nc, mDist1_renter, dP_C, dP_NC, vt_stay_nc, vt_stay_c, vt_stay_renter, h_renter, b_c_stay, b_renter, b_nc_stay,rental_stock, coastal_beq1, noncoastal_beq1, savings_beq1,vCoeff_C,vCoeff_NC)
    #                 print('dP_NC, excess deam C, excess dem NC ', dP_NC,excess_demand_C_stock, excess_demand_NC_stock)
    #                 print('dP_NC, net dem C, net dem NC ', dP_NC,net_demand_C, net_demand_NC)
            
            
    #         if t == 4:
    #             print('we are flooded')
    #             for i in np.arange(-0.001, 0.0010, 0.0001):
    #                 dP_C=dP_C_lom+i
    #                 dP_NC=dP_NC_lom
    #                 excess_demand_C_stock, excess_demand_NC_stock, net_demand_C, net_demand_NC, investment_C, investment_NC, stock_demand_rental, rental_stock=sim.excess_demand_continuous(func, datasets[t,7,k], int(datasets[t,2,k]), int(datasets[t,3,k]),int(datasets[t,4,k]),int(datasets[t,5,k]),datasets[t,6,k], grids, par, mMarkov, dPi_S, dPi_L, iNj, mDist1_c, mDist1_nc, mDist1_renter, dP_C, dP_NC, vt_stay_nc, vt_stay_c, vt_stay_renter, h_renter, b_c_stay, b_renter, b_nc_stay,rental_stock, coastal_beq1, noncoastal_beq1, savings_beq1,vCoeff_C,vCoeff_NC)
                
    #                 print('dP_C, excess deam C, excess dem NC ', dP_C, excess_demand_C_stock, excess_demand_NC_stock)
    #                 print('dP_C, net dem C, net dem NC ',dP_C, net_demand_C, net_demand_NC)
                
    #             for i in np.arange(-0.001, 0.0010, 0.0001):
    #                 dP_C=dP_C_lom
    #                 dP_NC=dP_NC_lom+i
    #                 excess_demand_C_stock, excess_demand_NC_stock, net_demand_C, net_demand_NC, investment_C, investment_NC, stock_demand_rental, rental_stock=sim.excess_demand_continuous(func, datasets[t,7,k], int(datasets[t,2,k]), int(datasets[t,3,k]),int(datasets[t,4,k]),int(datasets[t,5,k]),datasets[t,6,k], grids, par, mMarkov, dPi_S, dPi_L, iNj, mDist1_c, mDist1_nc, mDist1_renter, dP_C, dP_NC, vt_stay_nc, vt_stay_c, vt_stay_renter, h_renter, b_c_stay, b_renter, b_nc_stay,rental_stock, coastal_beq1, noncoastal_beq1, savings_beq1,vCoeff_C,vCoeff_NC)
    #                 print('dP_NC, excess deam C, excess dem NC ', dP_NC,excess_demand_C_stock, excess_demand_NC_stock)
    #                 print('dP_NC, net dem C, net dem NC ', dP_NC,net_demand_C, net_demand_NC)
            
    
    #t1 = time.time()
    #print("computation time for vfi is", t1- t0)

    
    #dP_C_guess = np.zeros((grids.vAlpha.size))
    #dP_NC_guess = np.zeros((grids.vAlpha.size))
    
    #a_index=0
    #alpha = grids.vAlpha[a_index]
    #guess_c = lom.LoM_C(grids.vAlpha, 0, 0, 0, 0,vCoeff_C)[a_index]
    #guess_nc = lom.LoM_NC(grids.vAlpha, 0, 0, 0, 0,vCoeff_NC)[a_index]
    #bound_c_l = lom.LoM_C(grids.vAlpha, 0, 0, 0, 0,vCoeff_C)[a_index] - 0.1
    #bound_c_r = lom.LoM_C(grids.vAlpha, 0, 0, 0, 0,vCoeff_C)[a_index] + 0.1
    #bound_nc_l = lom.LoM_NC(grids.vAlpha, 0, 0, 0, 0,vCoeff_NC)[a_index] - 0.1
    #bound_nc_r = lom.LoM_NC(grids.vAlpha, 0, 0, 0, 0,vCoeff_NC)[a_index] + 0.1
    
    
   # learning=False
    #dP_C_guess, dP_NC_guess, vCoeff_C, vCoeff_NC, iteration=equil.initialise_coefficients(method, grids, par, dPi_L, dPi_S, iNj, mMarkov, vCoeff_C, vCoeff_NC)
    #t0 = time.time()
    #dP_C_vec, dP_NC_vec, vCoeff_C, vCoeff_NC, iteration = equil.find_coefficients(method, grids, par, dPi_L, dPi_S, iNj, mMarkov, vCoeff_C, vCoeff_NC)
    #vt_stay_c, vt_stay_nc, vt_renter,vt_buy_c, vt_buy_nc, b_stay_c, b_stay_nc, b_renter, vt_stay_c_wf, vt_stay_nc_wf, vt_renter_wf, vt_buy_c_wf, vt_buy_nc_wf = household_problem.solve_welfare(grids, par, dPi_L, iNj, mMarkov,vCoeff_C,vCoeff_NC, True)
    ##t1 = time.time()
    #print("computation time without parallelisation is", t1- t0)
    #print(vC_endog)
    #print(results_matrix)
    #t0 = time.time()
    #dP_C_guess, dP_NC_guess, vCoeff_C, vCoeff_NC, iteration=equil.initialise_coefficients(method, grids, par, dPi_L, iNj, mMarkov, vCoeff_C, vCoeff_NC)
    #t1 = time.time()
    #print("computation time without parallelisation is", t1- t0)
    
    #dP_C_guess, dP_NC_guess, vCoeff_C, vCoeff_NC, iteration=equil.initialise_coefficients(method, grids, par, dPi_L, iNj, mMarkov, vCoeff_C, vCoeff_NC)
    
    #for k in range(grids.vSLR.size):
    #    plt.plot(grids.vTime, lom.LoM_C(grids,np.arange(grids.vTime.size), grids.vTime, grids.mPi_S[:,k],vCoeff_C), label=f'{k}')
    #plt.legend(loc = 4)
    #plt.show()
    #plt.clf()  
    
    #for k in range(grids.vSLR.size):
    #    plt.plot(grids.vTime, lom.LoM_NC(grids,np.arange(grids.vTime.size), grids.vTime, grids.mPi_S[:,k],vCoeff_NC), label=f'{k}')
    #plt.legend(loc = 4)
    #plt.show()
    #plt.clf()  

    
    #t0 = time.time()
    #dP_C_vec, dP_NC_vec, vCoeff_C, vCoeff_NC, iteration=equil.find_coefficients(method, grids, par, dPi_L, iNj, mMarkov, vCoeff_C, vCoeff_NC)
    #t1 = time.time()
    #print("computation time without parallelisation is", t1- t0)
    #print(iteration)
    """
    """
    plt.plot(grids.vM, b_stay_c[14, 0, 0, 0,:,1,0,0, 0, 0, 0, 0],  label='coastal, L = 0, H = 1, no flood')
    plt.title( 'b prime, J = -5')
    plt.xlabel('Cash in hand')
    plt.ylabel('Cons')
    plt.legend(loc = 4)
    plt.show()
    plt.clf()
    
    plt.plot(grids.vM, b_stay_nc[14, 1, 0,0,:,1,0,0, 0, 0, 0, 0], label = 'noncoastal, L = 0, H = 1, no flood')        
    plt.title( 'b prime, J = -5')
    plt.xlabel('Cash in hand')
    plt.ylabel('Cons')
    plt.legend(loc = 4)
    plt.show()
    plt.clf()    
    
    plt.plot(grids.vM, b_stay_nc[14, 1, 0,0,:,2,8,4, 0, 0, 0, 0], label = 'noncoastal, j=14, L = 8, H = 2, no flood')        
    plt.title( 'b prime')
    plt.xlabel('Cash in hand')
    plt.ylabel('Cons')
    plt.legend(loc = 4)
    plt.show()
    plt.clf()  
    
    plt.plot(grids.vM, b_stay_nc[13, 1, 0,0,:,2,8,4, 0, 0, 0, 0], label = 'noncoastal, j=13, L = 8, H = 2, no flood')        
    plt.title( 'b prime')
    plt.xlabel('Cash in hand')
    plt.ylabel('Cons')
    plt.legend(loc = 4)
    plt.show()
    plt.clf() 
    
    
    for l_index in range(grids.vL.size):
        plt.plot(
            grids.vM,
            vt_stay_nc[0,0, 0,0,3,:,2,l_index,1],
            label=f'noncoastal, j=0, k=0, g=0, H=2, e=4, l_index={l_index}'
        )
    
    plt.title('vt')
    plt.xlabel('Cash in hand')
    plt.ylabel('Cons')
    plt.legend(loc=4)
    plt.show()
    plt.clf()
    
    for l_index in range(grids.vL.size):
        plt.plot(
            grids.vM,
            vt_stay_c[0,0, 0,0,3,:,2,l_index,1],
            label=f'coastal, j=0, k=0, g=0, H=2, e=4, l_index={l_index}'
        )
    
        
    plt.title('vt')
    plt.xlabel('Cash in hand')
    plt.ylabel('Cons')
    plt.legend(loc=4)
    plt.show()
    plt.clf()
    
    
    for t_index in range(grids.vTime.size):
        plt.plot(
            grids.vM,
            vt_stay_c[t_index,0, 0,0,3,:,2,0,1],
            #label=f'coastal,t_index={t_index}, j=0, k=0, g=0, H=2, e=4, l_index=0'
        )
    
        
    plt.title('vt')
    plt.xlabel('Cash in hand')
    plt.ylabel('Cons')
    #plt.legend(loc=4)
    plt.show()
    plt.clf()
    
    
    for g_index in range(grids.vG.size):
        plt.plot(
            grids.vX[10:],
            vt_renter[0,0, 0,g_index,3,10:,1],
            label=f'rent, j=0, k=1, g_index={g_index}, e=1'
        )
        
        
        plt.plot(
            grids.vX[10:],
            vt_buy_c[0,0, 0,g_index,3,10:,1],
            label=f'buy coastal, j=0, k=1, g_index={g_index}, e=4'
        )
        
       
        plt.plot(
            grids.vX[10:],
            vt_buy_nc[0,0, 0,g_index,3,10:,1],
            label=f'buy non-coastal, j=0, k=1, g_index={g_index}, e=4'
        )
        
        plt.title('vt')
        plt.xlabel('Cash in hand')
        plt.ylabel('Cons')
        plt.legend(loc=4)
        plt.show()
        plt.clf()
        
        
    for slr_index in range(grids.vSLR.size):
        plt.plot(
            grids.vX[10:],
            vt_renter[10,0, 0,2,slr_index,10:,1],
            label=f'rent, j=0, k=0, g_index={g_index}, e=1'
        )
        
        
        plt.plot(
            grids.vX[10:],
            vt_buy_c[10,0, 0,2,slr_index,10:,1],
            label=f'buy coastal, j=0, k=0, g_index={g_index}, e=4'
        )
        
       
        plt.plot(
            grids.vX[10:],
            vt_buy_nc[10,0, 0,2,slr_index,10:,1],
            label=f'buy non-coastal, j=0, k=0, g_index={g_index}, e=4'
        )
        
        plt.title('vt')
        plt.xlabel('Cash in hand')
        plt.ylabel('Cons')
        plt.legend(loc=4)
        plt.show()
        plt.clf()
        
    for slr_index in range(grids.vSLR.size):
        plt.plot(
            grids.vX[10:],
            vt_renter[10,0, 1,2,slr_index,10:,1],
            label=f'rent, j=0, k=1, g_index={g_index}, e=1'
        )
        
        
        plt.plot(
            grids.vX[10:],
            vt_buy_c[10,0, 1,2,slr_index,10:,1],
            label=f'buy coastal, j=0, k=1, g_index={g_index}, e=4'
        )
        
       
        plt.plot(
            grids.vX[10:],
            vt_buy_nc[10,0, 1,2,slr_index,10:,1],
            label=f'buy non-coastal, j=0, k=1, g_index={g_index}, e=4'
        )
        
        plt.title('vt')
        plt.xlabel('Cash in hand')
        plt.ylabel('Cons')
        plt.legend(loc=4)
        plt.show()
        plt.clf()

    
    
    
    plt.plot(grids.vX, b_renter[14, 0, 0,0,:,0, 0, 0, 0, 0], label = 'renter, no flood')
    plt.title( 'b prime, J = -5')
    plt.xlabel('Cash in hand')
    plt.ylabel('Cons')
    plt.legend(loc = 4)
    plt.show()
    plt.clf() 
    
    plt.plot(grids.vM, b_stay_c[3, 0, 0, 0,:,1,0,0, 0, 0, 0, 0],  label='coastal, L = 0, H = 1, no flood')
    plt.title( 'b prime, J = 0')
    plt.xlabel('Cash in hand')
    plt.ylabel('Cons')
    plt.legend(loc = 4)
    plt.show()
    plt.clf()
    
    plt.plot(grids.vM, b_stay_nc[3, 0, 0,0,:,1,0,0, 0, 0, 0, 0], label = 'noncoastal, L = 0, H = 1, no flood')        
    plt.title( 'b prime, J = 0')
    plt.xlabel('Cash in hand')
    plt.ylabel('Cons')
    plt.legend(loc = 4)
    plt.show()
    plt.clf()    
    
    
    plt.plot(grids.vX, b_renter[3, 0, 0,0,:,0, 0, 0, 0, 0], label = 'renter, no flood')
    plt.title( 'b prime, J = 0')
    plt.xlabel('Cash in hand')
    plt.ylabel('Cons')
    plt.legend(loc = 4)
    plt.show()
    plt.clf() 
    
    for g_index in range(grids.vG.size):
        plt.plot(grids.vM, vt_stay_c[0,0, 0, g_index, 0,:,1,0,1],  label='realist welfare')
        plt.plot(grids.vM, vt_stay_c_wf[0, 0, 1, g_index, 0,:,1,0,1],  label='optimist welfare')
        plt.title( 'Coastal')
        plt.xlabel('Cash in hand')
        plt.ylabel('Welfare')
        plt.legend(loc = 4)
        plt.show()
        plt.clf()
        
        plt.plot(grids.vM, vt_stay_nc[0, 0, 0, g_index, 0,:,1,0,1], label = 'realist welfare')    
        plt.plot(grids.vM, vt_stay_nc_wf[0, 0, 1, g_index, 0,:,1,0,1], label = 'optimist welfare')        
        plt.title('noncoastal')
        plt.xlabel('Cash in hand')
        plt.ylabel('Welfare')
        plt.legend(loc = 4)
        plt.show()
        plt.clf()   
        
        plt.plot(grids.vX, vt_renter[0, 0, 0, g_index, 0,:,1], label = 'realist welfare')
        plt.plot(grids.vX, vt_renter_wf[0, 0, 1, g_index, 0,:,1], label = 'optimist welfare')
        plt.title( 'renter')
        plt.xlabel('Cash in hand')
        plt.ylabel('Welfare')
        plt.legend(loc = 4)
        plt.show()
        plt.clf() 
        
        
    for g_index in range(grids.vG.size):
        plt.plot(grids.vM, vt_stay_c[5,0, 0, g_index, 0,:,1,0,1],  label='realist welfare')
        plt.plot(grids.vM, vt_stay_c_wf[5, 0, 1, g_index, 0,:,1,0,1],  label='optimist welfare')
        plt.title( 'Coastal')
        plt.xlabel('Cash in hand')
        plt.ylabel('Welfare')
        plt.legend(loc = 4)
        plt.show()
        plt.clf()
        
        plt.plot(grids.vM, vt_stay_nc[5, 0, 0, g_index, 0,:,1,0,1], label = 'realist welfare')    
        plt.plot(grids.vM, vt_stay_nc_wf[5, 0, 1, g_index, 0,:,1,0,1], label = 'optimist welfare')        
        plt.title('noncoastal')
        plt.xlabel('Cash in hand')
        plt.ylabel('Welfare')
        plt.legend(loc = 4)
        plt.show()
        plt.clf()   
        
        plt.plot(grids.vX, vt_renter[5, 0, 0, g_index, 0,:,1], label = 'realist welfare')
        plt.plot(grids.vX, vt_renter_wf[5, 0, 1, g_index, 0,:,1], label = 'optimist welfare')
        plt.title( 'renter')
        plt.xlabel('Cash in hand')
        plt.ylabel('Welfare')
        plt.legend(loc = 4)
        plt.show()
        plt.clf() 
        
        
        
    for g_index in range(grids.vG.size):
        plt.plot(grids.vM, vt_stay_c[10,0, 0, g_index, 0,:,1,0,1],  label='realist welfare')
        plt.plot(grids.vM, vt_stay_c_wf[10, 0, 1, g_index, 0,:,1,0,1],  label='optimist welfare')
        plt.title( 'Coastal')
        plt.xlabel('Cash in hand')
        plt.ylabel('Welfare')
        plt.legend(loc = 4)
        plt.show()
        plt.clf()
        
        plt.plot(grids.vM, vt_stay_nc[10, 0, 0, g_index, 0,:,1,0,1], label = 'realist welfare')    
        plt.plot(grids.vM, vt_stay_nc_wf[10, 0, 1, g_index, 0,:,1,0,1], label = 'optimist welfare')        
        plt.title('noncoastal')
        plt.xlabel('Cash in hand')
        plt.ylabel('Welfare')
        plt.legend(loc = 4)
        plt.show()
        plt.clf()   
        
        plt.plot(grids.vX, vt_renter[10, 0, 0, g_index, 0,:,1], label = 'realist welfare')
        plt.plot(grids.vX, vt_renter_wf[10, 0, 1, g_index, 0,:,1], label = 'optimist welfare')
        plt.title( 'renter')
        plt.xlabel('Cash in hand')
        plt.ylabel('Welfare')
        plt.legend(loc = 4)
        plt.show()
        plt.clf() 
        
        
    for g_index in range(grids.vG.size):
        plt.plot(grids.vM, vt_stay_c[15,0, 0, g_index, 0,:,1,0,1],  label='realist welfare')
        plt.plot(grids.vM, vt_stay_c_wf[15, 0, 1, g_index, 0,:,1,0,1],  label='optimist welfare')
        plt.title( 'Coastal')
        plt.xlabel('Cash in hand')
        plt.ylabel('Welfare')
        plt.legend(loc = 4)
        plt.show()
        plt.clf()
        
        plt.plot(grids.vM, vt_stay_nc[15, 0, 0, g_index, 0,:,1,0,1], label = 'realist welfare')    
        plt.plot(grids.vM, vt_stay_nc_wf[15, 0, 1, g_index, 0,:,1,0,1], label = 'optimist welfare')        
        plt.title('noncoastal')
        plt.xlabel('Cash in hand')
        plt.ylabel('Welfare')
        plt.legend(loc = 4)
        plt.show()
        plt.clf()   
        
        plt.plot(grids.vX, vt_renter[15, 0, 0, g_index, 0,:,1], label = 'realist welfare')
        plt.plot(grids.vX, vt_renter_wf[15, 0, 1, g_index, 0,:,1], label = 'optimist welfare')
        plt.title( 'renter')
        plt.xlabel('Cash in hand')
        plt.ylabel('Welfare')
        plt.legend(loc = 4)
        plt.show()
        plt.clf() 
        
    for g_index in range(grids.vG.size):
        plt.plot(grids.vM, vt_stay_c[-1,0, 0, g_index, 0,:,1,0,1],  label='realist welfare')
        plt.plot(grids.vM, vt_stay_c_wf[-1, 0, 1, g_index, 0,:,1,0,1],  label='optimist welfare')
        plt.title( 'Coastal')
        plt.xlabel('Cash in hand')
        plt.ylabel('Welfare')
        plt.legend(loc = 4)
        plt.show()
        plt.clf()
        
        plt.plot(grids.vM, vt_stay_nc[-1, 0, 0, g_index, 0,:,1,0,1], label = 'realist welfare')    
        plt.plot(grids.vM, vt_stay_nc_wf[-1, 0, 1, g_index, 0,:,1,0,1], label = 'optimist welfare')        
        plt.title('noncoastal')
        plt.xlabel('Cash in hand')
        plt.ylabel('Welfare')
        plt.legend(loc = 4)
        plt.show()
        plt.clf()   
        
        plt.plot(grids.vX, vt_renter[-1, 0, 0, g_index, 0,:,1], label = 'realist welfare')
        plt.plot(grids.vX, vt_renter_wf[-1, 0, 1, g_index, 0,:,1], label = 'optimist welfare')
        plt.title( 'renter')
        plt.xlabel('Cash in hand')
        plt.ylabel('Welfare')
        plt.legend(loc = 4)
        plt.show()
        plt.clf() 
   
    t0 = time.time()
    mDist1_c, mDist1_nc, mDist1_renter, rental_stock, coastal_beq, noncoastal_beq, savings_beq=sim.stat_dist_finder(grids, par, mMarkov, dPi_L, iNj, vt_stay_nc[0,], vt_stay_c[0,], vt_renter[0,], b_stay_c[0,], b_renter[0,], b_stay_nc[0,],vCoeff_C,vCoeff_NC)
    t1 = time.time()
    
    
    #df = pd.DataFrame(results_matrix)
    #df.to_excel("latest_it.xlsx")
    #w_c, w_nc, w_renter, vt_buy_c,vt_stay_c, vt_buy_nc,vt_stay_nc, vt_stay_renter, c_c, c_nc, c_renter, h_c,h_nc, l_c, l_nc, h_renter, b_c_stay, b_nc_stay, b_c_buy, b_nc_buy, b_renter = household_problem_epsilons.solve(grids, par, dPi_L, dPi_S, iNj, mMarkov,vCoeff_C,vCoeff_NC)
   
    for l_index in range(grids.vL.size):
        plt.plot(grids.vM, vt_stay_c[-5, 0, 0, 4,:,1,l_index,1, 0, 0, 0, 0],  label=f'coastal, L = 0, H = 1, l = {l_index}')
        #plt.plot(grids.vM, vt_stay_c[-5, 0, 0, 4,:,1,0,1, 1, 0, 0, 0], label = 'coastal, L = 0, H = 1, flood')        
        #plt.plot(grids.vM, vt_stay_nc[-5, 0, 0, 4,:,1,0,1, 1, 0, 0, 0], label = 'noncoastal, L = 0, H = 1, flood')
        plt.title( 'Value stayers, J = -5')
        plt.xlabel('Cash in hand')
        plt.ylabel('Cons')
        plt.legend(loc = 4)
    plt.show()
    plt.clf()         

        
    
    for l_index in range(grids.vL.size):
        plt.plot(grids.vM, vt_stay_nc[-5, 0, 0, 4,:,1,l_index,1, 0, 0, 0, 0],  label=f'noncoastal, L = 0, H = 1, l = {l_index}')
        #plt.plot(grids.vM, vt_stay_c[-5, 0, 0, 4,:,1,0,1, 1, 0, 0, 0], label = 'coastal, L = 0, H = 1, flood')        
        #plt.plot(grids.vM, vt_stay_nc[-5, 0, 0, 4,:,1,0,1, 1, 0, 0, 0], label = 'noncoastal, L = 0, H = 1, flood')
        plt.title( 'Value stayers, J = -5')
        plt.xlabel('Cash in hand')
        plt.ylabel('Cons')
        plt.legend(loc = 4)
    plt.show()
    plt.clf()         

    
    for l_index in range(grids.vL.size):
        plt.plot(grids.vM, vt_stay_c[0, 0, 0, 4,:,1,l_index,1, 0, 0, 0, 0],  label=f'coastal, L = 0, H = 1, l = {l_index}')
        #plt.plot(grids.vM, vt_stay_c[-5, 0, 0, 4,:,1,0,1, 1, 0, 0, 0], label = 'coastal, L = 0, H = 1, flood')        
        #plt.plot(grids.vM, vt_stay_nc[-5, 0, 0, 4,:,1,0,1, 1, 0, 0, 0], label = 'noncoastal, L = 0, H = 1, flood')
        plt.title( 'Value stayers, J = 0')
        plt.xlabel('Cash in hand')
        plt.ylabel('Cons')
        plt.legend(loc = 4)      
       
    plt.show()
    plt.clf()      
    
    for l_index in range(grids.vL.size):
        plt.plot(grids.vM, vt_stay_nc[0, 0, 0, 4,:,1,l_index,1, 0, 0, 0, 0],  label=f'noncoastal, L = 0, H = 1, l = {l_index}')
        #plt.plot(grids.vM, vt_stay_c[-5, 0, 0, 4,:,1,0,1, 1, 0, 0, 0], label = 'coastal, L = 0, H = 1, flood')        
        #plt.plot(grids.vM, vt_stay_nc[-5, 0, 0, 4,:,1,0,1, 1, 0, 0, 0], label = 'noncoastal, L = 0, H = 1, flood')
        plt.title( 'Value stayers, J = 0')
        plt.xlabel('Cash in hand')
        plt.ylabel('Cons')
        plt.legend(loc = 4)
    plt.show()
    plt.clf()      
    
    
  
        
    
   
    
    
    
    #print(vCoeff_C)
    #print(vCoeff_NC)
    # w_c, w_nc, w_renter, vt_buy_c,vt_stay_c, vt_buy_nc,vt_stay_nc, vt_stay_renter, c_c, c_nc, c_renter, h_c,h_nc, l_c, l_nc, h_renter, b_c_stay, b_nc_stay, b_c_buy, b_nc_buy, b_renter = household_problem_epsilons.solve(grids, par, dPi_L, dPi_S, iNj, mMarkov, vCoeff_C, vCoeff_NC)

    # for a_index in range(grids.vAlpha.size):
    #     alpha=grids.vAlpha[a_index]
    #     dP_C=lom.LoM_C(alpha, 0, 0, 0, 0,vCoeff_C)
    #     dP_NC=lom.LoM_NC(alpha, 0, 0, 0, 0,vCoeff_NC)    
    #     func = 'intialise'
    #     mDist1_c, mDist1_nc, mDist1_renter, rental_stock, coastal_beq, noncoastal_beq, savings_beq=sim.stat_dist_finder(grids, par, mMarkov, dPi_S, dPi_L, iNj, vt_stay_nc, vt_stay_c, vt_stay_renter, b_c_stay, b_renter, b_nc_stay,vCoeff_C,vCoeff_NC,alpha)
    #     t0 = time.time()
    #     excess_demand_C_stock, excess_demand_NC_stock, net_demand_C, net_demand_NC, investment_C, investment_NC, stock_demand_rental, rental_stock=sim.excess_demand_continuous(func, alpha, 0,0,0,0,0, grids, par, mMarkov, dPi_S, dPi_L, iNj, mDist1_c, mDist1_nc, mDist1_renter, dP_C, dP_NC, vt_stay_nc, vt_stay_c, vt_stay_renter, h_renter, b_c_stay, b_renter, b_nc_stay,rental_stock, coastal_beq, noncoastal_beq, savings_beq,vCoeff_C,vCoeff_NC)
    #     t1 = time.time()
    #     print("computation time for excess demand with alpha", grids.vAlpha[a_index],"is", t1- t0)
    #     print("stock excess demands:",excess_demand_C_stock, excess_demand_NC_stock)
    #     func = 'find'
    #     excess_demand_C_flow, excess_demand_NC_flow, net_demand_C, net_demand_NC, investment_C, investment_NC, stock_demand_rental, rental_stock=sim.excess_demand_continuous(func, alpha, 0,0,0,0,0, grids, par, mMarkov, dPi_S, dPi_L, iNj, mDist1_c, mDist1_nc, mDist1_renter, dP_C, dP_NC, vt_stay_nc, vt_stay_c, vt_stay_renter, h_renter, b_c_stay, b_renter, b_nc_stay,rental_stock, coastal_beq, noncoastal_beq, savings_beq,vCoeff_C,vCoeff_NC)
    #     print("flow excess demands:",excess_demand_C_flow, excess_demand_NC_flow)
    #     print('net demand C, bequests C', net_demand_C, coastal_beq)
    #     print('net demand NC, bequests NC', net_demand_NC, noncoastal_beq)
    #     print('stock_demand_rental, rental_stock', stock_demand_rental, rental_stock)
        #print(net_demand_C,coastal_beq)
        #print(net_demand_NC,noncoastal_beq)
        #print(stock_demand_rental, rental_stock)
        #func = 'find'
        #t0 = time.time()
        #excess_demand_C_stock, excess_demand_NC_stock, net_demand_C, net_demand_NC, investment_C, investment_NC, stock_demand_rental, rental_stock=sim.excess_demand_continuous(func, alpha, 0,0,0,0,0, grids, par, mMarkov, dPi_S, dPi_L, iNj, mDist1_c, mDist1_nc, mDist1_renter, dP_C, dP_NC, vt_stay_nc, vt_stay_c, vt_stay_renter, h_renter, b_c_stay, b_renter, b_nc_stay,rental_stock, coastal_beq, noncoastal_beq, savings_beq,vCoeff_C,vCoeff_NC)
        #t1 = time.time()
        #print("computation time for excess demand with alpha", grids.vAlpha[a_index],"is", t1- t0)
        #print("excess demands:",excess_demand_C_stock, excess_demand_NC_stock)
        #print(net_demand_C,coastal_beq)
        #print(net_demand_NC,noncoastal_beq)
        #print(stock_demand_rental, rental_stock)
    
    vt_stay_c, vt_stay_nc, vt_renter, b_stay_c, b_stay_nc, b_renter=household_problem.solve(grids, par, dPi_L, iNj, mMarkov,vCoeff_C,vCoeff_NC)
    t0 = time.time()
    mDist1_c, mDist1_nc, mDist1_renter, rental_stock, coastal_beq, noncoastal_beq, savings_beq=sim.stat_dist_finder(grids, par, mMarkov, dPi_L, iNj,  vt_stay_c[0,], vt_stay_nc[0,], vt_renter[0,], b_stay_c[0,], b_stay_nc[0,], b_renter[0,], vCoeff_C,vCoeff_NC)
    t1 = time.time()
    print("computation time for distribution is", t1- t0)
    print("stocks:",rental_stock, coastal_beq, noncoastal_beq)
    t0 = time.time()
    method = 'secant'
   
    sum_mDist1_c = np.sum(mDist1_c, axis=6)
    sum_mDist1_c = np.sum(sum_mDist1_c, axis=4)
    sum_mDist1_c = np.sum(sum_mDist1_c, axis=3)
    sum_mDist1_c = np.sum(sum_mDist1_c, axis=2)
    sum_mDist1_c = np.sum(sum_mDist1_c, axis=1)
   
    
    sum_mDist1_nc = np.sum(mDist1_nc, axis=6)
    sum_mDist1_nc = np.sum(sum_mDist1_nc, axis=4)
    sum_mDist1_nc = np.sum(sum_mDist1_nc, axis=3)
    sum_mDist1_nc = np.sum(sum_mDist1_nc, axis=2)
    sum_mDist1_nc = np.sum(sum_mDist1_nc, axis=1)

    
    sum_renters=np.sum(mDist1_renter, axis=4)
    sum_renters=np.sum(sum_renters, axis=3)
    sum_renters=np.sum(sum_renters, axis=2)
    sum_renters=np.sum(sum_renters, axis=1)

    #print("Alpha=",alpha)
    #print(sum_mDist1_c/sumsum_mDist1_c,sum_mDist1_nc/sumsum_mDist1_nc)
    #print(sumsum_mDist1_c,sumsum_mDist1_nc)
   
    
    plt.plot(grids.vL_sim, np.sum(sum_mDist1_c, axis = 0),label='coastal')
    plt.plot(grids.vL_sim, np.sum(sum_mDist1_nc, axis = 0),label='noncoastal')
    plt.title('mortgage mass, all ages')
    plt.legend()
    plt.show()
    plt.clf()
    
    for j in range(par.iNj):
        plt.plot(grids.vL_sim, sum_mDist1_c[j,:],label='coastal')
        plt.plot(grids.vL_sim, sum_mDist1_nc[j,:],label='noncoastal')   
        plt.title(f'Probability Mass Function by Age Group (j = {j})')
        plt.xlabel('Loan to equity ratio')
        plt.ylabel('Probability Mass')
    
        # Add legend and annotation for clarity
        plt.legend()
        plt.text(0.05, 0.95,
        'Note: Each point shows discrete probability mass.\n'
        'Heights sum to 1 across all points.',
        transform=plt.gca().transAxes,
        fontsize=8, va='top', ha='left')           
        # Show and clear plot for next iteration
        plt.show()
        plt.clf()
    
    #sum_mDist1_nc = np.sum(sum_mDist1_nc, axis=1)
    #sum_mDist1_nc = np.sum(sum_mDist1_nc, axis=0)
        
    #sum_mDist1_c = np.sum(sum_mDist1_c, axis=1)
    #sum_mDist1_c = np.sum(sum_mDist1_c, axis=0)
    
    sum_mDist1_c = np.sum(mDist1_c, axis=6)
    sum_mDist1_c = np.sum(sum_mDist1_c, axis=5)
    sum_mDist1_c = np.sum(sum_mDist1_c, axis=4)
    sum_mDist1_c = np.sum(sum_mDist1_c, axis=2)
    sum_mDist1_c = np.sum(sum_mDist1_c, axis=1)
 
    
    sum_mDist1_nc = np.sum(mDist1_nc, axis=6)
    sum_mDist1_nc = np.sum(sum_mDist1_nc, axis=5)
    sum_mDist1_nc = np.sum(sum_mDist1_nc, axis=4)
    sum_mDist1_nc = np.sum(sum_mDist1_nc, axis=2)
    sum_mDist1_nc = np.sum(sum_mDist1_nc, axis=1)
    
    sum_renters=np.sum(mDist1_renter, axis=4)
    sum_renters=np.sum(sum_renters, axis=2)
    sum_renters=np.sum(sum_renters, axis=1)
  

    
    
    
    plt.plot(grids.vM_sim, np.sum(sum_mDist1_c, axis = 0),label='coastal')
    plt.plot(grids.vM_sim, np.sum(sum_mDist1_nc, axis = 0),label='noncoastal')
    plt.plot(grids.vX_sim, np.sum(sum_renters, axis = 0),label='renters')
    plt.title('cih mass, all ages')
    plt.legend()
    plt.show()
    plt.clf()
    

    
    for j in range(par.iNj):
        plt.plot(grids.vM_sim, sum_mDist1_c[j,:]*iNj,label='coastal')
        plt.plot(grids.vM_sim, sum_mDist1_nc[j,:]*iNj,label='noncoastal')
        plt.plot(grids.vX_sim, sum_renters[j,:]*iNj,label='renters')
        plt.title(f'Probability Mass Function by Age Group (j = {j})')
        plt.xlabel('Cash in hand value')
        plt.ylabel('Probability Mass')
    
        # Add legend and annotation for clarity
        plt.legend()
        plt.text(0.05, 0.95,
        'Note: Each point shows discrete probability mass.\n'
        'Heights sum to 1 across all points.',
        transform=plt.gca().transAxes,
        fontsize=8, va='top', ha='left')           
        # Show and clear plot for next iteration
        plt.show()
        plt.clf()

    #print("Total mass:", sum_mDist1_c+sum_mDist1_nc+sum_renters)
    
    start=time.perf_counter() 
    func='find'
    Simulated_eps=0
    Simulated_Leps=0
    Simulated_L2eps=0
    Simulated_L3eps=0
    Simulated_Lkeps=0
    Simulated_alpha=grids.vAlpha[a_index] 
    dP_C=lom.LoM_C(alpha,0,0,0,0,vCoeff_C)
    dP_NC=lom.LoM_NC(alpha,0,0,0,0,vCoeff_NC)
    excess_demand_C_stock, excess_demand_NC_stock, net_demand_C, net_demand_NC, investment_C, investment_NC, stock_demand_rental, rental_stock=sim.excess_demand_continuous(func, Simulated_alpha, Simulated_eps,Simulated_Leps,Simulated_L2eps,Simulated_L3eps,Simulated_Lkeps, grids, par, mMarkov, dPi_S, dPi_L, iNj, mDist1_c, mDist1_nc, mDist1_renter, dP_C, dP_NC, vt_stay_nc, vt_stay_c, vt_renter, b_stay_c, b_renter, b_stay_nc,rental_stock, coastal_beq, noncoastal_beq, savings_beq,vCoeff_C,vCoeff_NC, learning)
    print("Stock checks:")
    print(net_demand_C, coastal_beq)
    print(net_demand_NC, noncoastal_beq)
    print(stock_demand_rental, rental_stock)
    print("Excess demands:",excess_demand_C_stock, excess_demand_NC_stock)
    end=time.perf_counter() 
    print("Excess demand time:", end-start)
    
    moments=mom.calc_moments(par, grids, dP_C, dP_NC, mDist1_c, mDist1_nc,mDist1_renter, alpha, dPi_S,dPi_L,  vCoeff_C, vCoeff_NC)
    print(moments)
    dataset, beta_C, beta_NC, r_sqrd_C, r_sqrd_NC, demand_error_C, demand_error_NC=err.prediction_errors(method, grids, par, dPi_L, dPi_S, iNj, mMarkov, vCoeff_C, vCoeff_NC)
    df = pd.DataFrame(dataset)
    df.to_excel("new_error_stat.xslx")
    
    
    #start=time.perf_counter()
    #dP_C_guess, dP_NC_guess, vCoeff_C, vCoeff_NC, iteration=equil.initialise_coefficients(method, grids, par, dPi_L, dPi_S, iNj, mMarkov, vCoeff_C, vCoeff_NC)
    #end=time.perf_counter() 
    #print("Iteration time is:", end-start)
    #print(iteration)
    #print(vCoeff_C, vCoeff_NC)
    
    dPi_L_vec=np.array([0.1])
    vCoeff_C_vec=np.zeros((dPi_L_vec.size,vCoeff_C.size))
    vCoeff_NC_vec=np.zeros((dPi_L_vec.size,vCoeff_NC.size))
    initialise_time_vec=np.zeros((dPi_L_vec.size)) 
    find_time_vec=np.zeros((dPi_L_vec.size)) 
    init_iteration_vec=np.zeros((dPi_L_vec.size)) 
    find_iteration_vec=np.zeros((dPi_L_vec.size)) 
    for i in range(dPi_L_vec.size):
        dPi_L=dPi_L_vec[i]
        start=time.perf_counter() 
        dP_C_guess, dP_NC_guess, vCoeff_C_vec[i,:], vCoeff_NC_vec[i,:], init_iteration_vec[i]=equil.initialise_coefficients(method, grids, par, dPi_L, dPi_S, iNj, mMarkov, vCoeff_C, vCoeff_NC)
        print("The initialised coefficients are:",vCoeff_C, vCoeff_NC)
        end=time.perf_counter() 
       
        initialise_time_vec[i]=end-start
        print("Initialise time is:", end-start)
    

  
        start=time.perf_counter() 
        dP_C_vec, dP_NC_vec, vCoeff_C_vec[i,:], vCoeff_NC_vec[i,:], find_iteration_vec[i], mc_time, dist_time=equil.find_coefficients(method, grids, par, dPi_L, dPi_S, iNj, mMarkov, vCoeff_C, vCoeff_NC)
        print("The final coefficients are:",vCoeff_C, vCoeff_NC)
    
        end=time.perf_counter() 
        find_time_vec[i]=end-start
        print("Finding time is:", end-start)
        
    df = pd.DataFrame(vCoeff_C_vec)
    df.to_excel("coeffc_find_it.xlsx")
    
    df = pd.DataFrame(vCoeff_NC_vec)
    df.to_excel("coeffnc_find_it.xlsx") 
    print(vCoeff_C_vec,vCoeff_NC_vec)
   
    print(initialise_time_vec,find_time_vec,init_iteration_vec,find_iteration_vec) 
    
     
    
    
            
    
    #a_index=2
    #alpha=grids.vAlpha[a_index]      
    #t0 = time.time()
    #mDist1_c, mDist1_nc, mDist1_renter, rental_stock, coastal_beq, noncoastal_beq, savings_beq = simulation_old.stat_dist_finder(grids, par, mMarkov, dPi_S, dPi_L, iNj, vt_stay_nc, vt_stay_c, vt_stay_renter, b_c_stay, b_renter, b_nc_stay,vCoeff_C,vCoeff_NC,alpha)
    #t1 = time.time()
    #print("computation time for distribution with alpha", grids.vAlpha[a_index],"is", t1- t0)
    #t0 = time.time()
    #method = 'secant'
    
    

    
    
    
   
    
    #func='initialise'
    #Simulated_alpha=alpha
    #Simulated_eps=0
    #Simulated_Leps=0
    #Simulated_L2eps=0
    #Simulated_L3eps=0
    #Simulated_Lkeps=0
    #dP_C=lom.LoM_C(alpha, 0, 0, 0, 0,vCoeff_C)
    #dP_NC=lom.LoM_NC(alpha, 0, 0, 0, 0,vCoeff_NC)
    
    #excess_demand_C_stock, excess_demand_NC_stock, net_demand_C, net_demand_NC, investment_C, investment_NC, stock_demand_rental, rental_stock=simulation_old.excess_demand_continuous(func, Simulated_alpha, Simulated_eps,Simulated_Leps,Simulated_L2eps,Simulated_L3eps,Simulated_Lkeps, grids, par, mMarkov, dPi_S, dPi_L, iNj, mDist1_c, mDist1_nc, mDist1_renter, dP_C, dP_NC, vt_stay_nc, vt_stay_c, vt_stay_renter, b_renter,rental_stock, coastal_beq, noncoastal_beq, savings_beq,vCoeff_C,vCoeff_NC)
    #print(net_demand_C,coastal_beq)
    #print(net_demand_NC,noncoastal_beq)
    #print(stock_demand_rental, rental_stock)
    
    # print("computation time for distribution with alpha", grids.vAlpha[a_index],"is", t1- t0)
    # t0 = time.time()
    # dP_C_guess[a_index], dP_NC_guess[a_index], it = equil.house_prices_brentq(grids, par, guess_c, guess_nc, bound_c_l, bound_c_r, bound_nc_l, bound_nc_r, mMarkov, dPi_S, dPi_L, iNj, mDist1_c, mDist1_nc, mDist1_renter, vt_stay_nc, vt_stay_c, vt_stay_renter, h_renter, b_c_stay, b_renter, b_nc_stay, rental_stock, coastal_beq, noncoastal_beq, savings_beq,vCoeff_C, vCoeff_NC,alpha)
    # t1 = time.time()
    # print("computation time for brentq with alpha", grids.vAlpha[a_index],"is", t1- t0)

    # plt.plot(grids.vX, l_nc[-1, 0, 0, 4,:,1, 0, 0, 0, 0], label = 'L, noncoastal')
    # plt.plot(grids.vX, l_c[-1, 0, 0, 4,:,1, 0, 0, 0, 0], label = 'L coastal')
    # plt.title( 'Policy mortgages, j = -1')
    # plt.xlabel('Cash in hand')
    # plt.ylabel('Cons')
    # plt.legend(loc = 4)
    # plt.show()
    # plt.clf()
    
    # plt.plot(grids.vX, l_nc[-2, 0, 0, 4,:,1, 0, 0, 0, 0], label = 'L, noncoastal')
    # plt.plot(grids.vX, l_c[-2, 0, 0, 4,:,1, 0, 0, 0, 0], label = 'L coastal')
    # plt.title( 'Policy mortgages, j = -2')
    # plt.xlabel('Cash in hand')
    # plt.ylabel('Cons')
    # plt.legend(loc = 4)
    # plt.show()
    # plt.clf()
    
    # plt.plot(grids.vX, h_nc[-1, 0, 0, 4,:,1, 0, 0, 0, 0], label = 'H, noncoastal')
    # plt.plot(grids.vX, h_c[-1, 0, 0, 4,:,1, 0, 0, 0, 0], label = 'H coastal')
    # plt.title( 'Policy housinh, j = -1')
    # plt.xlabel('Cash in hand')
    # plt.ylabel('Cons')
    # plt.legend(loc = 4)
    # plt.show()
    # plt.clf()
    
    # plt.plot(grids.vX, h_nc[-2, 0, 0, 4,:,1, 0, 0, 0, 0], label = 'H, noncoastal')
    # plt.plot(grids.vX, h_c[-2, 0, 0, 4,:,1, 0, 0, 0, 0], label = 'H coastal')
    # plt.title( 'Policy housinh, j = -2')
    # plt.xlabel('Cash in hand')
    # plt.ylabel('Cons')
    # plt.legend(loc = 4)
    # plt.show()
    # plt.clf()
    
    # plt.plot(grids.vX[20:], vt_buy_c[-1, 0, 0, 4,20:,1, 0, 0, 0, 0], label = 'noncoastal')
    # plt.plot(grids.vX[20:], vt_buy_nc[-1, 0, 0, 4,20:,1, 0, 0, 0, 0], label = 'coastal')
    # plt.title( 'Value buyers, j = -1')
    # plt.xlabel('Cash in hand')
    # plt.ylabel('Cons')
    # plt.legend(loc = 4)
    # plt.show()
    # plt.clf()
    
    # plt.plot(grids.vX[20:], vt_buy_c[-2, 0, 0, 4,20:,1, 0, 0, 0, 0], label = 'noncoastal')
    # plt.plot(grids.vX[20:], vt_buy_nc[-2, 0, 0, 4,20:,1, 0, 0, 0, 0], label = 'coastal')
    # plt.title( 'Value buyers, j = -2')
    # plt.xlabel('Cash in hand')
    # plt.ylabel('Cons')
    # plt.legend(loc = 4)
    # plt.show()
    # plt.clf()
    
    
    # plt.plot(grids.vM, vt_stay_c[-1, 0, 0, 4,:,0,0,1, 0, 0, 0, 0], label = 'coastal, L = 0, H = 0')
    # plt.plot(grids.vM, vt_stay_c[-1, 0, 0, 4,:,1,0,1, 0, 0, 0, 0], label = 'coastal, L = 0, H = 1')
    # plt.plot(grids.vM, vt_stay_c[-1, 0, 0, 4,:,2,0,1, 0, 0, 0, 0], label = 'coastal, L = 0, H = 2')
    # plt.title( 'Value stayers, J = -1')
    # plt.xlabel('Cash in hand')
    # plt.ylabel('Cons')
    # plt.legend(loc = 4)
    # plt.show()
    # plt.clf()
    
    
   
    
    #for h_index in range(grids.vH.size):
        #plt.plot(grids.vM,q_nc[13,0,0,0,:,h_index,0,0, 0, 0, 0] )        
        #plt.ylabel('MU_c')
        #plt.xlabel('Cash in hand')
        #plt.show()
        #print(q_nc[13,0,0,0,:,h_index,0,0, 0, 0, 0])
    
    # Simulated_alpha=grids.vAlpha[4]
    # alpha=grids.vAlpha[4]
    # dP_C_nodev = lom.LoM_C(Simulated_alpha, Simulated_eps,Simulated_Leps,Simulated_L2eps,Simulated_Lkeps,vCoeff_C)
    # dP_NC = lom.LoM_NC(Simulated_alpha, Simulated_eps,Simulated_Leps,Simulated_L2eps,Simulated_Lkeps,vCoeff_NC)
    # guess_c = 0.5
    # guess_nc = 0.5
    # bound_c_l = 0.2
    # bound_c_r = 1.0
    # bound_nc_l = 0.2
    # bound_nc_r = 1.0
    # mDist1_c, mDist1_nc, mDist1_renter, rental_stock, coastal_beq, noncoastal_beq, savings_beq=sim.stat_dist_finder(grids, par, mMarkov, dPi_S, dPi_L, iNj, vt_stay_nc, vt_stay_c, vt_stay_renter, b_c_stay, b_renter, b_nc_stay,vCoeff_C,vCoeff_NC,alpha)
    # sum_mDist1_c = np.sum(mDist1_c, axis=6)
    # sum_mDist1_c = np.sum(sum_mDist1_c, axis=4)
    # sum_mDist1_c = np.sum(sum_mDist1_c, axis=3)
    # sum_mDist1_c = np.sum(sum_mDist1_c, axis=2)
    # sum_mDist1_c = np.sum(sum_mDist1_c, axis=1)
    # # sum_mDist1_c = np.sum(sum_mDist1_c, axis=0)
    
    # sum_mDist1_nc = np.sum(mDist1_nc, axis=6)
    # sum_mDist1_nc = np.sum(sum_mDist1_nc, axis=4)
    # sum_mDist1_nc = np.sum(sum_mDist1_nc, axis=3)
    # sum_mDist1_nc = np.sum(sum_mDist1_nc, axis=2)
    # sum_mDist1_nc = np.sum(sum_mDist1_nc, axis=1)
    # # sum_mDist1_nc = np.sum(sum_mDist1_nc, axis=0)
    
    # plt.plot(grids.vL_sim, np.sum(sum_mDist1_c, axis = 0),label='coastal')
    # plt.plot(grids.vL_sim, np.sum(sum_mDist1_nc, axis = 0),label='noncoastal')
    # plt.title('mortgage mass, all ages')
    # plt.legend()
    # plt.show()
    # plt.clf()
    
    # for j in range(par.iNj):
    #     plt.plot(grids.vL_sim, sum_mDist1_c[j,:],label='coastal')
    #     plt.plot(grids.vL_sim, sum_mDist1_nc[j,:],label='noncoastal')
    #     plt.title('mortgage mass, per age')
    #     plt.legend()
    #     plt.show()
    #     plt.clf()
    
    # sum_mDist1_c = np.sum(mDist1_c, axis=6)
    # sum_mDist1_c = np.sum(sum_mDist1_c, axis=5)
    # sum_mDist1_c = np.sum(sum_mDist1_c, axis=4)
    # sum_mDist1_c = np.sum(sum_mDist1_c, axis=2)
    # sum_mDist1_c = np.sum(sum_mDist1_c, axis=1)
    # # sum_mDist1_c = np.sum(sum_mDist1_c, axis=0)
    
    # sum_mDist1_nc = np.sum(mDist1_nc, axis=6)
    # sum_mDist1_nc = np.sum(sum_mDist1_nc, axis=5)
    # sum_mDist1_nc = np.sum(sum_mDist1_nc, axis=4)
    # sum_mDist1_nc = np.sum(sum_mDist1_nc, axis=2)
    # sum_mDist1_nc = np.sum(sum_mDist1_nc, axis=1)
    # # sum_mDist1_nc = np.sum(sum_mDist1_nc, axis=0)
    
    # plt.plot(grids.vM_sim, np.sum(sum_mDist1_c, axis = 0),label='coastal')
    # plt.plot(grids.vM_sim, np.sum(sum_mDist1_nc, axis = 0),label='noncoastal')
    # plt.title('mortgage mass, all ages')
    # plt.legend()
    # plt.show()
    # plt.clf()
    
    # for j in range(par.iNj):
    #     plt.plot(grids.vM_sim, sum_mDist1_c[j,:],label='coastal')
    #     plt.plot(grids.vM_sim, sum_mDist1_nc[j,:],label='noncoastal')
    #     plt.title('cah mass, age = 0')
    #     plt.legend()
    #     plt.show()
    #     plt.clf()
    
    # dP_C_guess, dP_NC_guess, it = equil.house_prices_brentq(grids, par, guess_c, guess_nc, bound_c_l, bound_c_r, bound_nc_l, bound_nc_r, mMarkov, dPi_S, dPi_L, iNj, mDist1_c, mDist1_nc, mDist1_renter, vt_stay_nc, vt_stay_c, vt_stay_renter, h_renter, b_c_stay, b_renter, b_nc_stay, rental_stock, coastal_beq, noncoastal_beq, savings_beq,vCoeff_C, vCoeff_NC,alpha)
    
    # price_dev = np.linspace(-0.001,0.001,21)
    # excess_flow_demand_C_vec = np.zeros((price_dev.size))
    # depreciation_C_vec= np.zeros((price_dev.size))
    # investment_C_vec= np.zeros((price_dev.size))
    # net_demand_C_vec= np.zeros((price_dev.size))
    # excess_demand_C_vec= np.zeros((price_dev.size))
    
    # for i in range(price_dev.size):
    #     dP_C = dP_C_guess + price_dev[i]
    #     excess_demand_C_vec[i], excess_demand_NC, net_demand_C_vec[i], net_demand_NC, investment_C_vec[i], investment_NC, stock_demand_rental, rental_stock= sim.excess_demand_continuous(Simulated_alpha, Simulated_eps,Simulated_Leps,Simulated_L2eps,Simulated_L3eps,Simulated_Lkeps, grids, par, mMarkov, dPi_S, dPi_L, iNj, mDist1_c, mDist1_nc, mDist1_renter, dP_C, dP_NC, vt_stay_nc, vt_stay_c, vt_stay_renter, h_renter, b_c_stay, b_renter, b_nc_stay,rental_stock, coastal_beq, noncoastal_beq, savings_beq,vCoeff_C,vCoeff_NC)
    #     if price_dev[i]==0:
    #         print(net_demand_C_vec[i],coastal_beq)
    #         print(net_demand_NC,noncoastal_beq)
    #         print(stock_demand_rental, rental_stock)
            
    # plt.plot(price_dev, excess_demand_C_vec)
    # plt.title('excess demand C')
    # plt.show()
    # plt.clf()
    
    # plt.plot(price_dev, net_demand_C_vec)
    # plt.title('net demand C')
    # plt.show()
    # plt.clf()
    
    # plt.plot(price_dev, investment_C_vec)
    # plt.title('investment C')
    # plt.show()
    # plt.clf()
    
    # plt.plot(price_dev, depreciation_C_vec)
    # plt.title('depreciation C')
    # plt.show()
    # plt.clf()
    
    # plt.plot(price_dev, excess_flow_demand_C_vec)
    # plt.title('excess flow demand C')
    # plt.show()
    # plt.clf()
        
    # plt.plot(grids.vX, vt_stay_renter[0, 0, 0, 4,:,1, 0, 0, 0,0], label = 'j = 0')
    # plt.plot(grids.vX, vt_stay_renter[1, 0, 0, 4,:,1, 0, 0, 0,0], label = 'j = 1')
    # plt.title( 'Value, rent')
    # plt.xlabel('Cash in hand')
    # plt.ylabel('Cons')
    # plt.legend(loc = 4)
    # plt.show()
    # plt.clf()
    
    # plt.plot(grids.vX, h_renter[0, 0, 0, 4,:,1, 0, 0, 0,0], label = 'j = 0')
    # plt.plot(grids.vX, h_renter[1, 0, 0, 4,:,1, 0, 0, 0,0], label = 'j = 1')
    # plt.title( 'Policy, rent')
    # plt.xlabel('Cash in hand')
    # plt.ylabel('Cons')
    # plt.legend(loc = 4)
    # plt.show()
    # plt.clf()
        
    # plt.plot(grids.vM, h_c[4, 0, 0, 4,:,1, 0, 0, 0, 0], label = 'Housing')
    # plt.plot(grids.vM, l_c[4, 0, 0, 4,:,1, 0, 0, 0, 0], label = 'Mortgage')
    # plt.title( 'Policy, coastal')
    # plt.xlabel('Cash in hand')
    # plt.ylabel('Cons')
    # plt.legend(loc = 4)
    # plt.show()
    # plt.clf()
    
    # plt.plot(grids.vM, h_nc[4, 0, 0, 4,:,1, 0, 0, 0, 0], label = 'Housing')
    # plt.plot(grids.vM, l_nc[4, 0, 0, 4,:,1, 0, 0, 0, 0], label = 'Mortgage')
    # plt.title( 'Policy, non coastal')
    # plt.xlabel('Cash in hand')
    # plt.ylabel('Cons')
    # plt.legend(loc = 4)
    # plt.show()
    # plt.clf()
    
    # for l_index in range(grids.vL.size):
    #     plt.plot(grids.vM, vt_stay_c[4, 0, 0, 4,:,0,l_index,1, 0, 0, 0, 0], label = 'h=0')
    #     plt.plot(grids.vM, vt_stay_c[4, 0, 0, 4,:,1,l_index,1, 0, 0, 0, 0], label = 'h=1')
    #     plt.plot(grids.vM, vt_stay_c[4, 0, 0, 4,:,2,l_index,1, 0, 0, 0, 0], label = 'h=2')
    #     plt.title( 'Value, coastal')
    #     plt.xlabel('Cash in hand')
    #     plt.ylabel('Cons')
    #     plt.legend(loc = 4)
    #     plt.show()
    #     plt.clf()
    

    #choice=np.zeros((3,grids.vM.size))
    #choice2=np.zeros((3,grids.vM.size))
    #print(q_nc[13,0,0,0,:,2,0,0, 0, 0, 0])
    #print(c_nc[14,0,0,0,:,2,0,0, 0, 0, 0])
    #print(c_nc[13,0,0,0,:,2,0,0, 0, 0, 0])
   # for m_index in range(0,grids.vM.size):
       # choice[0,m_index]=interp.interp_1d(grids.vM,vt_stay_nc[14,0,0,0,:,2,0,0, 0, 0, 0],grids.vM[m_index]+(-par.dDelta)*0.9*grids.vH[2]+grids.vE[0])-interp.interp_1d(grids.vX,vt_stay_renter[14,0,0,0,:,0,0, 0, 0, 0],grids.vM[m_index]+(1-par.dDelta-par.dKappa_sell)*0.9*grids.vH[2]+grids.vE[0])
       # choice[1,m_index]=interp.interp_1d(grids.vM,vt_stay_nc[14,0,0,0,:,2,1,0, 0, 0, 0],grids.vM[m_index]+(-par.dDelta)*0.9*grids.vH[2]+grids.vE[1])-interp.interp_1d(grids.vX,vt_stay_renter[14,0,0,0,:,1,0, 0, 0, 0],grids.vM[m_index]+(1-par.dDelta-par.dKappa_sell)*0.9*grids.vH[2]+grids.vE[1])
      #  choice[2,m_index]=interp.interp_1d(grids.vM,vt_stay_nc[14,0,0,0,:,2,2,0, 0, 0, 0],grids.vM[m_index]+(-par.dDelta)*0.9*grids.vH[2]+grids.vE[2])-interp.interp_1d(grids.vX,vt_stay_renter[14,0,0,0,:,2,0, 0, 0, 0],grids.vM[m_index]+(1-par.dDelta-par.dKappa_sell)*0.9*grids.vH[2]+grids.vE[2])

    #for m_index in range(0,grids.vM.size):
       # choice2[0,m_index]=interp.interp_1d(grids.vM,vt_stay_nc[14,0,0,0,:,2,0,0, 0, 0, 0],grids.vM[m_index]+(-par.dDelta)*0.9*grids.vH[2]+grids.vE[0])-interp.interp_1d(grids.vX,vt_buy_nc[14,0,0,0,:,0,0, 0, 0, 0],grids.vM[m_index]+(1-par.dDelta-par.dKappa_sell)*0.9*grids.vH[2]+grids.vE[0])
       # choice2[1,m_index]=interp.interp_1d(grids.vM,vt_stay_nc[14,0,0,0,:,2,1,0, 0, 0, 0],grids.vM[m_index]+(-par.dDelta)*0.9*grids.vH[2]+grids.vE[1])-interp.interp_1d(grids.vX,vt_buy_nc[14,0,0,0,:,1,0, 0, 0, 0],grids.vM[m_index]+(1-par.dDelta-par.dKappa_sell)*0.9*grids.vH[2]+grids.vE[1])
       # choice2[2,m_index]=interp.interp_1d(grids.vM,vt_stay_nc[14,0,0,0,:,2,2,0, 0, 0, 0],grids.vM[m_index]+(-par.dDelta)*0.9*grids.vH[2]+grids.vE[2])-interp.interp_1d(grids.vX,vt_buy_nc[14,0,0,0,:,2,0, 0, 0, 0],grids.vM[m_index]+(1-par.dDelta-par.dKappa_sell)*0.9*grids.vH[2]+grids.vE[2])

         
    #print(choice)
    #print(choice2)
    #print(np.exp(grids.vE))
 
    
    #plt.plot(grids.vM,-1/vt_stay_c[0,0,-1,0,:,0,0,0, 0, 0, 0], label='Stayer c,j=0,g=-1,vt')
    #plt.plot(grids.vM,-1/vt_stay_nc[0,0,-1,0,:,0,0,0, 0, 0, 0], label='Stayer nc,j=0,g=-1,vt')
    #plt.plot(w_renter[0,0,0,0,:,0,0, 0, 0, 0], label='Renter')
    #plt.legend()
    #plt.show()
    
    #plt.plot(grids.vM,-1/vt_stay_c[0,0,-1,0,:,1,0,0, 0, 0, 0], label='Stayer c,j=0,g=-1,vt')
    #plt.plot(grids.vM,-1/vt_stay_nc[0,0,-1,0,:,1,0,0, 0, 0, 0], label='Stayer nc,j=0,g=-1,vt')
    #plt.plot(w_renter[0,0,0,0,:,0,0, 0, 0, 0], label='Renter')
    #plt.legend()
    #plt.show()
    
    #plt.plot(grids.vM,-1/vt_stay_c[0,0,-1,0,:,2,0,0, 0, 0, 0], label='Stayer c,j=0,g=-1,vt')
    #plt.plot(grids.vM,-1/vt_stay_nc[0,0,-1,0,:,2,0,0, 0, 0, 0], label='Stayer nc,j=0,g=-1,vt')
    #plt.plot(w_renter[0,0,0,0,:,0,0, 0, 0, 0], label='Renter')
    #plt.legend()
    #plt.show()
    
    #plt.plot(grids.vX[28:],-1/vt_buy_c[0,0,-1,0,28:,0,0, 0, 0, 0], label='Buyer c,j=0,g=-1,vt')
    #plt.plot(grids.vX[28:],-1/vt_buy_nc[0,0,-1,0,28:,0,0, 0, 0, 0], label='Buyer nc,j=0,g=-1,vt')
    #plt.plot(w_renter[0,0,0,0,:,0,0, 0, 0, 0], label='Renter')
    ##plt.legend()
    #plt.show()
    
    #plt.plot(grids.vX[28:],h_buyc[0,0,-1,0,28:,0,0, 0, 0, 0], label='House bought c,j=0,g=-1,vt')
    #plt.plot(grids.vX[28:],h_buync[0,0,-1,0,28:,0,0, 0, 0, 0], label='House bought nc,j=0,g=-1,vt')
    #plt.plot(w_renter[0,0,0,0,:,0,0, 0, 0, 0], label='Renter')
    #plt.legend()
    #plt.show()
    
    
    #plt.plot(grids.vB,w_c[0,0,-1,0,:,2,0,0, 0, 0, 0], label='Stayer c,j=0,g=-1')
    #plt.plot(grids.vB,w_nc[0,0,-1,0,:,2,0,0, 0, 0, 0], label='Stayer nc,j=0,g=-1')
    #plt.plot(w_renter[0,0,0,0,:,0,0, 0, 0, 0], label='Renter')
    #plt.legend()
    #plt.show()

    #plt.plot(grids.vM,-1/vt_stay_c[5,0,-1,0,:,2,0,0, 0, 0, 0], label='Stayer c,j=5,g=-1,vt')
    #plt.plot(grids.vM,-1/vt_stay_nc[5,0,-1,0,:,2,0,0, 0, 0, 0], label='Stayer nc,j=5,g=-1,vt')
    #plt.plot(w_renter[5,0,0,0,:,0,0, 0, 0, 0], label='Renter')
    #plt.legend()
    #plt.show()
    
    #plt.plot(grids.vM,c_c[5,0,-1,0,:,2,0,0, 0, 0, 0], label='Stayer c,j=5,g=-1')
    #plt.plot(grids.vM,c_nc[5,0,-1,0,:,2,0,0, 0, 0, 0], label='Stayer nc,j=5,g=-1')
    #plt.plot(w_renter[5,0,0,0,:,0,0, 0, 0, 0], label='Renter')
    #plt.legend()
    #plt.show()
    
    
    #plt.plot(grids.vM,-1/vt_stay_c[10,0,-1,0,:,2,0,0, 0, 0, 0], label='Stayer c,j=10,g=-1,vt')
    #plt.plot(grids.vM,-1/vt_stay_nc[10,0,-1,0,:,2,0,0, 0, 0, 0], label='Stayer nc,j=10,g=-1,vt')
    #plt.plot(w_renter[10,0,0,0,:,0,0, 0, 0, 0], label='Renter')
    #plt.legend()
    #plt.show() 
    
    #plt.plot(grids.vX[28:],-1/vt_buy_c[10,0,-1,0,28:,0,0, 0, 0, 0], label='Buyer c,j=10,g=-1,vt')
    #plt.plot(grids.vX[28:],-1/vt_buy_nc[10,0,-1,0,28:,0,0, 0, 0, 0], label='Buyer nc,j=10,g=-1,vt')
    #plt.plot(w_renter[10,0,0,0,:,0,0, 0, 0, 0], label='Renter')
    #plt.legend()
    #plt.show() 
    
    #plt.plot(grids.vX[28:],h_buyc[10,0,-1,0,28:,0,0, 0, 0, 0], label='House bought c,j=10,g=-1,vt')
    #plt.plot(grids.vX[28:],h_buync[10,0,-1,0,28:,0,0, 0, 0, 0], label='House bought nc,j=10,g=-1,vt')
    #plt.plot(w_renter[10,0,0,0,:,0,0, 0, 0, 0], label='Renter')
    #plt.legend()
    #plt.show() 
    
    #plt.plot(grids.vM,-1/vt_stay_c[14,0,-1,0,:,2,0,0, 0, 0, 0], label='Stayer c,j=10,g=-1,vt')
    #plt.plot(grids.vM,-1/vt_stay_nc[14,0,-1,0,:,2,0,0, 0, 0, 0], label='Stayer nc,j=10,g=-1,vt')
    #plt.plot(w_renter[10,0,0,0,:,0,0, 0, 0, 0], label='Renter')
    #plt.legend()
    #plt.show() 
    
     
    #plt.plot(grids.vX[28:],-1/vt_buy_c[14,0,-1,0,28:,0,0, 0, 0, 0], label='Buyer c,j=10,g=-1,vt')
    #plt.plot(grids.vX[28:],-1/vt_buy_nc[14,0,-1,0,28:,0,0, 0, 0, 0], label='Buyer nc,j=10,g=-1,vt')
    #plt.plot(w_renter[10,0,0,0,:,0,0, 0, 0, 0], label='Renter')
    #plt.legend()
    #plt.show() 
    
    #plt.plot(grids.vX[28:],h_buyc[14,0,-1,0,28:,0,0, 0, 0, 0], label='House bought c,j=10,g=-1,vt')
    #plt.plot(grids.vX[28:],h_buync[14,0,-1,0,28:,0,0, 0, 0, 0], label='House bought nc,j=10,g=-1,vt')
    #plt.plot(w_renter[10,0,0,0,:,0,0, 0, 0, 0], label='Renter')
    #plt.legend()
    #plt.show() 
    
    #plt.plot(grids.vM,c_c[11,0,-1,0,:,2,0,0, 0, 0, 0], label='Stayer c,j=13,g=-1')
    #plt.plot(grids.vM,c_nc[11,0,-1,0,:,2,0,0, 0, 0, 0], label='Stayer nc,j=13,g=-1')
    #plt.plot(w_renter[10,0,0,0,:,0,0, 0, 0, 0], label='Renter')
    #plt.legend()
    #plt.show()
    
    #plt.plot(grids.vM,c_c[12,0,-1,0,:,2,0,0, 0, 0, 0], label='Stayer c,j=13,g=-1')
    #plt.plot(grids.vM,c_nc[12,0,-1,0,:,2,0,0, 0, 0, 0], label='Stayer nc,j=13,g=-1')
    #plt.plot(w_renter[10,0,0,0,:,0,0, 0, 0, 0], label='Renter')
    #plt.legend()
    #plt.show()
        
    #plt.plot(grids.vM,c_c[13,0,-1,0,:,2,0,0, 0, 0, 0], label='Stayer c,j=13,g=-1')
    #plt.plot(grids.vM,c_nc[13,0,-1,0,:,2,0,0, 0, 0, 0], label='Stayer nc,j=13,g=-1')
    #plt.plot(w_renter[10,0,0,0,:,0,0, 0, 0, 0], label='Renter')
    #plt.legend()
    #plt.show()
    
    #plt.plot(grids.vM,c_c[14,0,-1,0,:,2,0,0, 0, 0, 0], label='Stayer c,j=14,g=-1')
    #plt.plot(grids.vM,c_nc[14,0,-1,0,:,2,0,0, 0, 0, 0], label='Stayer nc,j=14,g=-1')
    #plt.plot(w_renter[10,0,0,0,:,0,0, 0, 0, 0], label='Renter')
    #plt.legend()
    #plt.show()
   

    #print(w_c[0,0,0,0,:,2,0,0, 0, 0, 0],w_c[0,0,1,0,:,2,0,0, 0, 0, 0],w_c[0,0,2,0,:,2,0,0, 0, 0, 0])
    #plt.plot(grids.vM,-1/qt_nc_stay[13,0,0,0,:,2,0,0, 0, 0, 0], label='Stayer MUc')
    #plt.legend()
    #plt.show()
    #print(interp.interp_1d(grids.vX,-1/qt_stay_renter[14,0,0,0,:,0,0, 0, 0, 0],(1-par.dDelta-par.dKappa_sell)*0.9*grids.vH[2]+grids.vE[0]))

    #print(-1/qt_stay_renter[0,0,0,0,:,0,0, 0, 0, 0])

    # dP_C=np.zeros((grids.vAlpha.size))
    # dP_NC=np.zeros((grids.vAlpha.size))
    # coastal_stock=np.zeros((grids.vAlpha.size))   
    # noncoastal_stock=np.zeros((grids.vAlpha.size)) 
    # vrental_stock=np.zeros((grids.vAlpha.size))
    # net_demand_C=np.zeros((grids.vAlpha.size)) 
    # net_demand_NC=np.zeros((grids.vAlpha.size))
    # excess_demand_C=np.zeros((grids.vAlpha.size)) 
    # excess_demand_NC=np.zeros((grids.vAlpha.size))
   
    # for alpha_index in range(grids.vAlpha.size):
    #     Simulated_alpha=grids.vAlpha[alpha_index]
    #     dP_C[alpha_index] = lom.LoM_C(Simulated_alpha, Simulated_eps,Simulated_Leps,Simulated_L2eps,Simulated_Lkeps,vCoeff_C)
    #     dP_NC[alpha_index] = lom.LoM_NC(Simulated_alpha, Simulated_eps,Simulated_Leps,Simulated_L2eps,Simulated_Lkeps,vCoeff_NC)
    # plt.plot(grids.vAlpha,dP_C,label='Coastal house price')
    # plt.plot(grids.vAlpha,dP_NC,label='Non-coastal house price')    
    # plt.xlabel('Belief parameter alpha')
    # plt.ylabel('Long-run equilibrium price')
    # plt.legend()
    # plt.show()
    

    # for alpha_index in range(grids.vAlpha.size):
    #     Simulated_alpha=grids.vAlpha[alpha_index]
    #     alpha=grids.vAlpha[alpha_index]
    #     dP_C[alpha_index] = lom.LoM_C(Simulated_alpha, Simulated_eps,Simulated_Leps,Simulated_L2eps,Simulated_Lkeps,vCoeff_C)
    #     dP_NC[alpha_index] = lom.LoM_NC(Simulated_alpha, Simulated_eps,Simulated_Leps,Simulated_L2eps,Simulated_Lkeps,vCoeff_NC)
    #     print(dP_C[alpha_index],dP_NC[alpha_index])
    #     mDist1_c, mDist1_nc, mDist1_renter, rental_stock, coastal_beq, noncoastal_beq, savings_beq=sim.stat_dist_finder(grids, par, mMarkov, dPi_S, dPi_L, iNj, vt_stay_nc, vt_stay_c, vt_stay_renter, h_c, h_nc, h_renter, b_c_stay, b_renter, b_nc_stay,vCoeff_C,vCoeff_NC,alpha)
    #     excess_demand_C[alpha_index], excess_demand_NC[alpha_index], net_demand_C[alpha_index], net_demand_NC[alpha_index], coastal_beq, noncoastal_beq, investment_C, investment_NC, depreciation_C, depreciation_NC, stock_demand_rental, rental_stock=sim.excess_demand_continuous(Simulated_alpha, Simulated_eps,Simulated_Leps,Simulated_L2eps,Simulated_Lkeps, grids, par, mMarkov, dPi_S, dPi_L, iNj, mDist1_c, mDist1_nc, mDist1_renter, dP_C[alpha_index], dP_NC[alpha_index], vt_stay_nc, vt_stay_c, vt_stay_renter, h_renter, b_c_stay, b_renter, b_nc_stay,rental_stock, coastal_beq, noncoastal_beq, savings_beq,vCoeff_C,vCoeff_NC)         
    #     print(excess_demand_C[alpha_index], excess_demand_NC[alpha_index])
    #     print(net_demand_C, net_demand_NC)
    #     print(coastal_beq, noncoastal_beq)
    #     sum_mDist1_nc = np.sum(mDist1_nc, axis=5)
    #     sum_mDist1_nc = np.sum(sum_mDist1_nc, axis=3)
    #     sum_mDist1_nc = np.sum(sum_mDist1_nc, axis=2) 
    #     sum_mDist1_nc = np.sum(sum_mDist1_nc, axis=1)
    #     sum_mDist1_nc = np.sum(sum_mDist1_nc, axis=0) 
    #     noncoastal_stock[alpha_index]=np.sum(sum_mDist1_nc*grids.vH)+noncoastal_beq
    #     sum_mDist1_c = np.sum(mDist1_c, axis=5)
    #     sum_mDist1_c = np.sum(sum_mDist1_c, axis=3)
    #     sum_mDist1_c = np.sum(sum_mDist1_c, axis=2) 
    #     sum_mDist1_c = np.sum(sum_mDist1_c, axis=1)
    #     sum_mDist1_c = np.sum(sum_mDist1_c, axis=0) 
    #     coastal_stock[alpha_index]=np.sum(sum_mDist1_c*grids.vH)+coastal_beq
    #     vrental_stock[alpha_index]=rental_stock
       
    # plt.plot(grids.vAlpha,coastal_stock,label='Coastal housing')
    # plt.plot(grids.vAlpha,noncoastal_stock,label='Non-coastal housing') 
    # plt.xlabel('Belief parameter alpha')
    # plt.ylabel('Long-run equilibrium stock')
    # plt.title('larger b')
    # plt.legend()
    # plt.show()
    
    # plt.plot(grids.vAlpha,excess_demand_C*(1/par.dDelta),label='Coastal housing')
    # plt.plot(grids.vAlpha,excess_demand_NC*(1/par.dDelta),label='Non-coastal housing') 
    # plt.xlabel('Belief parameter alpha')
    # plt.ylabel('Excess stock demand')
    # plt.title('larger b')
    # plt.legend()
    # plt.show()
    
    # plt.plot(grids.vAlpha,dP_C,label='Coastal housing')
    # plt.plot(grids.vAlpha,dP_NC,label='Non-coastal housing') 
    # plt.xlabel('Belief parameter alpha')
    # plt.ylabel('Price')
    # plt.title('larger b')
    # plt.legend()
    # plt.show()
 
    #print(choice)
    #print(choice2)
    #print(np.exp(grids.vE))
    #plt.plot(vt_stay_nc[14,0,0,0,:,2,0,0, 0, 0, 0], label='Stayer')
    #plt.plot(vt_stay_renter[14,0,0,0,:,0,0, 0, 0, 0], label='Renter')
    #plt.legend()
    #plt.show()

    #plt.plot(grids.vM,-1/qt_nc_stay[14,0,0,0,:,2,0,0, 0, 0, 0], label='Stayer MUc')
    #plt.legend()
    #plt.show()
    #print(interp.interp_1d(grids.vX,q_renter[14,0,0,0,:,0,0, 0, 0, 0],(1-par.dDelta-par.dKappa_sell)*0.9*grids.vH[2]+grids.vE[0]))

    #print(-1/qt_stay_renter[14,0,0,0,:,0,0, 0, 0, 0])

    
    
    #dP_C_guess, dP_NC_guess, vCoeff_C, vCoeff_NC, iteration=equil.initialise_coefficients(grids, par, dPi_L, dPi_S, iNj, mMarkov, vCoeff_C, vCoeff_NC)  #sum_mDist1_c = np.sum(mDist1_c, axis=5)
    #sum_mDist1_c = np.sum(sum_mDist1_c, axis=4)
    #sum_mDist1_c = np.sum(sum_mDist1_c, axis=2) 
    #sum_mDist1_c = np.sum(sum_mDist1_c, axis=1)
    #sum_mDist1_c = np.sum(sum_mDist1_c, axis=0)
    #print(sum_mDist1_c)
    #sum_mDist1_nc = np.sum(mDist1_nc, axis=5)
    #sum_mDist1_nc = np.sum(sum_mDist1_nc, axis=4)
    #sum_mDist1_nc = np.sum(sum_mDist1_nc, axis=2) 
    #sum_mDist1_nc = np.sum(sum_mDist1_nc, axis=1)
    #sum_mDist1_nc = np.sum(sum_mDist1_nc, axis=0)
    #print(sum_mDist1_nc)
    #sum_mDist1_renter = np.sum(mDist1_renter, axis=4)
    #sum_mDist1_renter = np.sum(sum_mDist1_renter, axis=2) 
    #sum_mDist1_renter = np.sum(sum_mDist1_renter, axis=1)
    #sum_mDist1_renter = np.sum(sum_mDist1_renter, axis=0)
    #print(sum_mDist1_renter)
    #dP_C_guess[a_index], dP_NC_guess[a_index], it = equil.house_prices_brentq(grids, par, guess_c, guess_nc, bound_c_l, bound_c_r, bound_nc_l, bound_nc_r, mMarkov, dPi_S, dPi_L, iNj, mDist1_c, mDist1_nc, mDist1_renter, vt_stay_nc, vt_stay_c, vt_stay_renter, h_renter, b_c_stay, b_renter, b_nc_stay, rental_stock, coastal_beq, noncoastal_beq, savings_beq,vCoeff_in_C, vCoeff_in_NC,alpha_guess)
    #excess_demand_C, excess_demand_NC, net_demand_C, net_demand_NC, coastal_beq, noncoastal_beq, investment_C, investment_NC, depreciation_C, depreciation_NC, stock_demand_rental, rental_stock=sim.excess_demand_continuous(alpha_guess, Simulated_eps,Simulated_Leps,Simulated_L2eps,Simulated_Lkeps, grids, par, mMarkov, dPi_S, dPi_L, iNj, mDist1_c, mDist1_nc, mDist1_renter, dP_C_guess[a_index], dP_NC_guess[a_index], vt_stay_nc, vt_stay_c, vt_stay_renter, h_renter, b_c_stay, b_renter, b_nc_stay,rental_stock, coastal_beq, noncoastal_beq, savings_beq,vCoeff_C,vCoeff_NC)
    #print(excess_demand_C, excess_demand_NC)
    #print(net_demand_C, net_demand_NC, coastal_beq, noncoastal_beq, investment_C, investment_NC, depreciation_C, depreciation_NC, stock_demand_rental, rental_stock)
    #print('price C mc',dP_C_guess[a_index] )
    #print('price NC mc',dP_NC_guess[a_index] )
    
    
    #for h_index in range(grids.vH.size):
        ##plt.plot(grids.vX,q_nc[14,0,0,0,:,h_index,0,0, 0, 0, 0] )
        #plt.ylabel('MU_c')
        #plt.xlabel('Cash in hand')
        #plt.show()
    
  
    
    #for h_index in range(grids.vH.size):  
        #plt.plot(grids.vX,w_c[0,0,0,0,:,h_index,0,0, 0, 0, 0])
        #plt.ylabel('Continuation value')
        #plt.xlabel('Cash in hand')
        #plt.show()
        #plt.plot(grids.vX,q_c[0,0,0,0,:,h_index,0,0, 0, 0, 0])
        #plt.ylabel('Continuation value')
        #plt.xlabel('Cash in hand')
        #plt.show()
    
    #print(w_nc[13,0,0,0,:,0,0, 0, 0, 0])
    #print(q_nc[13,0,0,0,:,0,0, 0, 0, 0])

#plt.plot(grids.vM_sim,sum_mDist1_c[j,:])
#plt.ylabel('Mass')
#plt.xlabel('Cash in hand')
#plt.show()
    
    
    #a_index=2
    #alpha_guess=grids.vAlpha[a_index]
    #dP_C=lom.LoM_C(alpha_guess, 0, 0, 0, 0,vCoeff_C)
    #dP_NC=lom.LoM_NC(alpha_guess, 0, 0, 0, 0,vCoeff_NC)
    #mDist1_c, mDist1_nc, mDist1_renter, rental_stock, coastal_beq, noncoastal_beq, savings_beq = sim.stat_dist_finder(grids, par, mMarkov, dPi_S, dPi_L, iNj, vt_stay_nc, vt_stay_c, vt_stay_renter, h_c, h_nc, h_renter, b_c_stay, b_renter, b_nc_stay,vCoeff_in_C,vCoeff_in_NC,alpha_guess)
    
    #excess_demand_C, excess_demand_NC, net_demand_C, net_demand_NC, coastal_beq, noncoastal_beq, investment_C, investment_NC, depreciation_C, depreciation_NC, stock_demand_rental, rental_stock=sim.excess_demand_continuous(alpha_guess, Simulated_eps,Simulated_Leps,Simulated_L2eps,Simulated_Lkeps, grids, par, mMarkov, dPi_S, dPi_L, iNj, mDist1_c, mDist1_nc, mDist1_renter, dP_C, dP_NC, vt_stay_nc, vt_stay_c, vt_stay_renter, h_renter, b_c_stay, b_renter, b_nc_stay,rental_stock, coastal_beq, noncoastal_beq, savings_beq,vCoeff_C,vCoeff_NC)
    #print(net_demand_C, coastal_beq)
    #print(net_demand_NC, noncoastal_beq)
    #print(stock_demand_rental, rental_stock)
    #sum_mDist1_c = np.sum(mDist1_c, axis=5)
    #sum_mDist1_c = np.sum(sum_mDist1_c, axis=4)
    #sum_mDist1_c = np.sum(sum_mDist1_c, axis=3) 
    #sum_mDist1_c = np.sum(sum_mDist1_c, axis=2)
    #sum_mDist1_c = np.sum(sum_mDist1_c, axis=1)
    #print(sum_mDist1_c)
    #sum_mDist1_nc = np.sum(mDist1_nc, axis=5)
    #sum_mDist1_nc = np.sum(sum_mDist1_nc, axis=4)
    #sum_mDist1_nc = np.sum(sum_mDist1_nc, axis=3) 
    #sum_mDist1_nc = np.sum(sum_mDist1_nc, axis=2)
    #sum_mDist1_nc = np.sum(sum_mDist1_nc, axis=1)
    #print(sum_mDist1_nc)
    #sum_mDist1_renter = np.sum(mDist1_renter, axis=4)
    #sum_mDist1_renter = np.sum(sum_mDist1_renter, axis=3) 
    #sum_mDist1_renter = np.sum(sum_mDist1_renter, axis=2)
    #sum_mDist1_renter = np.sum(sum_mDist1_renter, axis=1)
    #print(sum_mDist1_renter)
           
                    #mass_rent,mass_buyc,mass_buync = sim.continuous_decide_renter(grids,vt_buy_c, vt_buy_nc, vt_stay_renter_sim,mass)        
                    #if (np.sum(mass_rent)+np.sum(mass_buyc)+np.sum(mass_buync)-np.sum(mass))>1e-10:
                        #print(mass)
                        #print(mass_rent, mass_buyc, mass_buync)
                    #assert not (np.sum(mass_rent)+np.sum(mass_buyc)+np.sum(mass_buync)-np.sum(mass))>1e-10
                    #for x_index_sim in range(grids.vX_sim.size):
                        #stock_demand_rental+=mass_rent[x_index_sim]*h_renter[x_index_sim]                        
                        #mDist1_c, mDist1_nc, mDist1_renter, coastal_beq, noncoastal_beq, savings_beq = sim.simulate_renter_mass(grids, par,iNj, mMarkov, mDist1_c, mDist1_nc, mDist1_renter, mass_rent[x_index_sim], mass_buyc[x_index_sim], mass_buync[x_index_sim], h_index,e_index, k_index, g_index, j,h, h_pol_C_index[x_index_sim], h_pol_NC_index[x_index_sim], b_buy_c[x_index_sim], b_buy_nc[x_index_sim], B_pol[x_index_sim], coastal_beq,noncoastal_beq, savings_beq)

    #net_demand_C_UD = demand_C - supply_C
    #net_demand_NC_UD = demand_NC - supply_NC

        
            
   
    



    
    #w_c, w_nc, w_renter, vt_buy_c,vt_stay_c, vt_buy_nc,vt_stay_nc, vt_stay_renter, c_c, c_nc, c_renter, h_c,h_nc, h_renter, b_c_stay, b_nc_stay, b_c_buy, b_nc_buy, b_renter=equilibrium_debug.initialise_valuef(grids, par, dPi_L, dPi_S, iNj, mMarkov, vCoeff_C, vCoeff_NC)
    #alpha=0.0
    #dP_C=20
    #dP_NC=20
    #mDist1_c, mDist1_nc, mDist1_renter, rental_stock, coastal_beq, noncoastal_beq, savings_beq, it_counter, total_bequest=sim.stat_dist_finder(grids, par, mMarkov, dPi_S, dPi_L, iNj, vt_stay_nc, vt_stay_c, vt_stay_renter, h_c, h_c, h_renter, b_c_stay, b_renter, b_nc_stay,vCoeff_C,vCoeff_NC,alpha)
    #excess_demand_C, excess_demand_NC, net_demand_C, net_demand_NC, coastal_beq, noncoastal_beq, investment_C, investment_NC, depreciation_C, depreciation_NC, stock_demand_rental, rental_stock=sim.excess_demand_continuous(alpha, 0,0,0,0, grids, par, mMarkov, dPi_S, dPi_L, iNj, mDist1_c, mDist1_nc, mDist1_renter, dP_C, dP_NC, vt_stay_nc, vt_stay_c, vt_stay_renter, h_renter, b_c_stay, b_renter, b_nc_stay,rental_stock, coastal_beq, noncoastal_beq, savings_beq,vCoeff_C,vCoeff_NC)
    #print(excess_demand_C, excess_demand_NC)
    #print(net_demand_C, net_demand_NC)
    #print(coastal_beq, noncoastal_beq,rental_stock)
    #print(investment_C, investment_NC, depreciation_C, depreciation_NC)
    
    #print(rental_stock, coastal_beq, noncoastal_beq)
    #sum_mDist1_c = np.sum(mDist1_c, axis=5)
    #sum_mDist1_c = np.sum(sum_mDist1_c, axis=4)
    #sum_mDist1_c = np.sum(sum_mDist1_c, axis=3) 
    #sum_mDist1_c = np.sum(sum_mDist1_c, axis=2)
    #sum_mDist1_c = np.sum(sum_mDist1_c, axis=1)
    #print(sum_mDist1_c)
    #sum_mDist1_nc = np.sum(mDist1_nc, axis=5)
    #sum_mDist1_nc = np.sum(sum_mDist1_nc, axis=4)
    #sum_mDist1_nc = np.sum(sum_mDist1_nc, axis=3) 
    #sum_mDist1_nc = np.sum(sum_mDist1_nc, axis=2)
    #sum_mDist1_nc = np.sum(sum_mDist1_nc, axis=1)
    #print(sum_mDist1_nc)
    #sum_mDist1_renter = np.sum(mDist1_renter, axis=4)
    #sum_mDist1_renter = np.sum(sum_mDist1_renter, axis=3) 
    #sum_mDist1_renter = np.sum(sum_mDist1_renter, axis=2)
    #sum_mDist1_renter = np.sum(sum_mDist1_renter, axis=1)
    #print(sum_mDist1_renter)
    
    #alpha=0.0
    #mDist1_c, mDist1_nc, mDist1_renter, rental_stock, coastal_beq, noncoastal_beq, savings_beq, it_counter, total_bequest=sim.stat_dist_finder(grids, par, mMarkov, dPi_S, dPi_L, iNj, vt_stay_nc, vt_stay_c, vt_stay_renter, h_c, h_c, h_renter, b_c_stay, b_renter, b_nc_stay,vCoeff_C,vCoeff_NC,alpha)
   
    #dP_C=0.896035072959272
    #dP_NC=0.8944162450150355
    
    #for i in np.arange(-0.001, 0.0010, 0.0001):         
     #   dP_NC=0.8944162450150355-i
      #  excess_demand_C, excess_demand_NC, net_demand_C, net_demand_NC, coastal_beq, noncoastal_beq, investment_C, investment_NC, depreciation_C, depreciation_NC, stock_demand_rental, rental_stock=sim.excess_demand_continuous(alpha, 0,0,0,0, grids, par, mMarkov, dPi_S, dPi_L, iNj, mDist1_c, mDist1_nc, mDist1_renter, dP_C, dP_NC, vt_stay_nc, vt_stay_c, vt_stay_renter, h_renter, b_c_stay, b_renter, b_nc_stay,rental_stock, coastal_beq, noncoastal_beq, savings_beq,vCoeff_C,vCoeff_NC)
       # print(excess_demand_C, excess_demand_NC)
        #print(net_demand_C, net_demand_NC, stock_demand_rental)
      
        
    #excess_demand_C_old, excess_demand_NC_old, net_demand_C_old, net_demand_NC_old, coastal_beq, noncoastal_beq, investment_C, investment_NC, depreciation_C, depreciation_NC, stock_demand_rental_old, rental_stock_old, demand_C_old, demand_NC_old=sim.excess_demand(alpha, 0,0,0,0, grids, par, mMarkov, dPi_S, dPi_L, iNj, mDist1_c, mDist1_nc, mDist1_renter, dP_C, dP_NC, vt_stay_nc, vt_stay_c, vt_stay_renter, h_renter, b_c_stay, b_renter, b_nc_stay,rental_stock, coastal_beq, noncoastal_beq, savings_beq,vCoeff_C,vCoeff_NC)
    #print(net_demand_C, net_demand_NC, coastal_beq, noncoastal_beq)
    #print(net_demand_C_old, net_demand_NC_old, coastal_beq, noncoastal_beq)
    #print(excess_demand_C, excess_demand_NC)
    #print(excess_demand_C_old, excess_demand_NC_old)
    #print(constrained_mass)
   # for i in np.arange(-0.001, 0.0010, 0.0001):
    #    dP_C=lom.LoM_C(alpha, 0, 0, 0, 0,vCoeff_C)+i
    #    dP_NC=lom.LoM_NC(alpha, 0, 0, 0, 0,vCoeff_NC)
     #   excess_demand_C, excess_demand_NC, net_demand_C, net_demand_NC, coastal_beq, noncoastal_beq, investment_C, investment_NC, depreciation_C, depreciation_NC, stock_demand_rental, rental_stock, demand_C, demand_NC=sim.excess_demand_continuous(alpha, 0,0,0,0, grids, par, mMarkov, dPi_S, dPi_L, iNj, mDist1_c, mDist1_nc, mDist1_renter, dP_C, dP_NC, vt_stay_nc, vt_stay_c, vt_stay_renter, h_renter, b_c_stay, b_renter, b_nc_stay,rental_stock, coastal_beq, noncoastal_beq, savings_beq,vCoeff_C,vCoeff_NC)
      #  excess_demand_C_old, excess_demand_NC_old, net_demand_C_old, net_demand_NC_old, coastal_beq, noncoastal_beq, investment_C, investment_NC, depreciation_C, depreciation_NC, stock_demand_rental_old, rental_stock_old, demand_C_old, demand_NC_old=sim.excess_demand(alpha, 0,0,0,0, grids, par, mMarkov, dPi_S, dPi_L, iNj, mDist1_c, mDist1_nc, mDist1_renter, dP_C, dP_NC, vt_stay_nc, vt_stay_c, vt_stay_renter, h_renter, b_c_stay, b_renter, b_nc_stay,rental_stock, coastal_beq, noncoastal_beq, savings_beq,vCoeff_C,vCoeff_NC)
       # print(excess_demand_C, excess_demand_NC)
        #print(excess_demand_C_old, excess_demand_NC_old)
    
    
    #sum_mDist1_c = np.sum(mDist1_c, axis=5)
    #sum_mDist1_c = np.sum(sum_mDist1_c, axis=4)
    #sum_mDist1_c = np.sum(sum_mDist1_c, axis=2)
    #sum_mDist1_c = np.sum(sum_mDist1_c, axis=1)
  
 
    #for j in range(iNj-1):
        #plt.plot(grids.vM_sim,sum_mDist1_c[j,:])
        #plt.ylabel('Mass')
        #plt.xlabel('Cash in hand')
        #plt.show()
        
    #sum_mDist1_renter = np.sum(mDist1_renter, axis=4)
    #sum_mDist1_renter = np.sum(sum_mDist1_renter, axis=2)
    #sum_mDist1_renter = np.sum(sum_mDist1_renter, axis=1)
  
 
    #for j in range(iNj-1):
        #plt.plot(grids.vX_sim,sum_mDist1_renter[j,:])
        #plt.ylabel('Mass')
        #plt.xlabel('Cash in hand')
        #plt.show()
        
     
    
    #for i in np.arange(-0.001, 0.0010, 0.0001):
     #   dP_C=lom.LoM_C(alpha, 0, 0, 0, 0,vCoeff_C)+i
      #  dP_NC=lom.LoM_NC(alpha, 0, 0, 0, 0,vCoeff_NC)
       # excess_demand_C, excess_demand_NC, net_demand_C, net_demand_NC, coastal_beq, noncoastal_beq, investment_C, investment_NC, depreciation_C, depreciation_NC, stock_demand_rental, rental_stock, demand_C, demand_NC=sim.excess_demand_continuous(alpha, 0,0,0,0, grids, par, mMarkov, dPi_S, dPi_L, iNj, mDist1_c, mDist1_nc, mDist1_renter, dP_C, dP_NC, vt_stay_nc, vt_stay_c, vt_stay_renter, h_renter, b_c_stay, b_renter, b_nc_stay,rental_stock, coastal_beq, noncoastal_beq, savings_beq,vCoeff_C,vCoeff_NC)
        #excess_demand_C_old, excess_demand_NC_old, net_demand_C_old, net_demand_NC_old, coastal_beq, noncoastal_beq, investment_C, investment_NC, depreciation_C, depreciation_NC, stock_demand_rental_old, rental_stock_old, demand_C_old, demand_NC_old=sim.excess_demand(alpha, 0,0,0,0, grids, par, mMarkov, dPi_S, dPi_L, iNj, mDist1_c, mDist1_nc, mDist1_renter, dP_C, dP_NC, vt_stay_nc, vt_stay_c, vt_stay_renter, h_renter, b_c_stay, b_renter, b_nc_stay,rental_stock, coastal_beq, noncoastal_beq, savings_beq,vCoeff_C,vCoeff_NC)
        #print(stock_demand_rental, rental_stock, demand_C, demand_NC)
        #print(stock_demand_rental_old, rental_stock_old, demand_C_old, demand_NC_old)
        
    #for alpha_index in range(grids.vAlpha.size):
     #   alpha=grids.vAlpha[alpha_index]
   #     dP_C=lom.LoM_C(alpha, 0, 0, 0, 0,vCoeff_C)
   #     dP_NC=lom.LoM_NC(alpha, 0, 0, 0, 0,vCoeff_NC) 
    #    mDist1_c, mDist1_nc, mDist1_renter, rental_stock, coastal_beq, noncoastal_beq, savings_beq, it_counter, total_bequest=sim.stat_dist_finder(grids, par, mMarkov, dPi_S, dPi_L, iNj, vt_stay_nc, vt_stay_c, vt_stay_renter, h_c, h_c, h_renter, b_c_stay, b_renter, b_nc_stay,vCoeff_C,vCoeff_NC,alpha)
    #    dP_C, dP_NC, it, excess_demand_C_check, excess_demand_NC_check=equil.house_prices_brentq(grids, par, dP_C, dP_NC, 0.8, 1.3, 0.8, 1.3, mMarkov, dPi_S, dPi_L, iNj, mDist1_c, mDist1_nc, mDist1_renter, vt_stay_nc, vt_stay_c, vt_stay_renter, h_renter, b_c_stay, b_renter, b_nc_stay, rental_stock, coastal_beq, noncoastal_beq, savings_beq,vCoeff_C, vCoeff_NC,alpha)
   #     print("This is the excess demand from the algorithm:", excess_demand_C_check, excess_demand_NC_check)
   #     excess_demand_C, excess_demand_NC, net_demand_C, net_demand_NC, coastal_beq, noncoastal_beq, investment_C, investment_NC, depreciation_C, depreciation_NC, stock_demand_rental, rental_stock=sim.excess_demand(alpha, Simulated_eps,Simulated_Leps,Simulated_L2eps,Simulated_Lkeps, grids, par, mMarkov, dPi_S, dPi_L, iNj, mDist1_c, mDist1_nc, mDist1_renter, dP_C, dP_NC, vt_stay_nc, vt_stay_c, vt_stay_renter, h_renter, b_c_stay, b_renter, b_nc_stay,rental_stock, coastal_beq, noncoastal_beq, savings_beq,vCoeff_C,vCoeff_NC)
   #     print(excess_demand_C, excess_demand_NC)
        
    #alpha=0.5
    #mDist1_c, mDist1_nc, mDist1_renter, rental_stock, coastal_beq, noncoastal_beq, savings_beq, it_counter, total_bequest=sim.stat_dist_finder(grids, par, mMarkov, dPi_S, dPi_L, iNj, vt_stay_nc, vt_stay_c, vt_stay_renter, h_c, h_c, h_renter, b_c_stay, b_renter, b_nc_stay,vCoeff_C,vCoeff_NC,alpha)
    #excess_demand_C_cvary=np.zeros((np.arange(-0.01, 0.011, 0.001).size))
    #excess_demand_NC_cvary=np.zeros((np.arange(-0.01, 0.011, 0.001).size))
    #dP_C_cvary=np.zeros((np.arange(-0.01, 0.011, 0.001).size))
    #print("Alpha=0.5")      
    #it_count=0
    #for i in np.arange(-0.01, 0.011, 0.001):
     #   dP_C=lom.LoM_C(alpha, 0, 0, 0, 0,vCoeff_C)+i
      #  dP_NC=lom.LoM_NC(alpha, 0, 0, 0, 0,vCoeff_NC)    
       # excess_demand_C, excess_demand_NC, net_demand_C, net_demand_NC, coastal_beq, noncoastal_beq, investment_C, investment_NC, depreciation_C, depreciation_NC, stock_demand_rental, rental_stock=sim.excess_demand(alpha, Simulated_eps,Simulated_Leps,Simulated_L2eps,Simulated_Lkeps, grids, par, mMarkov, dPi_S, dPi_L, iNj, mDist1_c, mDist1_nc, mDist1_renter, dP_C, dP_NC, vt_stay_nc, vt_stay_c, vt_stay_renter, h_renter, b_c_stay, b_renter, b_nc_stay,rental_stock, coastal_beq, noncoastal_beq, savings_beq,vCoeff_C,vCoeff_NC)
        #excess_demand_C_cvary[it_count]=excess_demand_C
        #excess_demand_NC_cvary[it_count]=excess_demand_NC
        #dP_C_cvary[it_count]=dP_C-lom.LoM_C(alpha, 0, 0, 0, 0,vCoeff_C)
        #it_count+=1
        #print(net_demand_C,coastal_beq)
        #print(net_demand_NC,noncoastal_beq)
        #print(stock_demand_rental,rental_stock)
        #print(depreciation_C,investment_C)
        #print(depreciation_NC,investment_NC)
    #plt.figure(figsize=(8, 5))
    #plt.plot(dP_C_cvary, excess_demand_C_cvary, label='Excess demand for coastal housing')
    #plt.plot(dP_C_cvary, excess_demand_NC_cvary, label='Excess demand for non-coastal housing')

    ##plt.xlabel('Deviation from LoM price, $P_C$')
    #plt.ylabel('Excess demand')
    #plt.title(r'Excess housing demand for $\alpha=0.5$')  # Latex-style rendering of alpha
    #plt.legend()
    #plt.grid(True)
    #plt.tight_layout()
    #plt.show()
    
    #alpha=1
    #mDist1_c, mDist1_nc, mDist1_renter, rental_stock, coastal_beq, noncoastal_beq, savings_beq, it_counter, total_bequest=sim.stat_dist_finder(grids, par, mMarkov, dPi_S, dPi_L, iNj, vt_stay_nc, vt_stay_c, vt_stay_renter, h_c, h_c, h_renter, b_c_stay, b_renter, b_nc_stay,vCoeff_C,vCoeff_NC,alpha)
    #excess_demand_C_cvary=np.zeros((np.arange(-0.01, 0.011, 0.001).size))
    #excess_demand_NC_cvary=np.zeros((np.arange(-0.01, 0.011, 0.001).size))
    #dP_C_cvary=np.zeros((np.arange(-0.01, 0.011, 0.001).size))
    #print("Alpha=1")      
    #it_count=0
    #for i in np.arange(-0.01, 0.011, 0.001):
     #   dP_C=lom.LoM_C(alpha, 0, 0, 0, 0,vCoeff_C)+i
      #  dP_NC=lom.LoM_NC(alpha, 0, 0, 0, 0,vCoeff_NC)    
       # excess_demand_C, excess_demand_NC, net_demand_C, net_demand_NC, coastal_beq, noncoastal_beq, investment_C, investment_NC, depreciation_C, depreciation_NC, stock_demand_rental, rental_stock=sim.excess_demand(alpha, Simulated_eps,Simulated_Leps,Simulated_L2eps,Simulated_Lkeps, grids, par, mMarkov, dPi_S, dPi_L, iNj, mDist1_c, mDist1_nc, mDist1_renter, dP_C, dP_NC, vt_stay_nc, vt_stay_c, vt_stay_renter, h_renter, b_c_stay, b_renter, b_nc_stay,rental_stock, coastal_beq, noncoastal_beq, savings_beq,vCoeff_C,vCoeff_NC)
        #excess_demand_C_cvary[it_count]=excess_demand_C
       # excess_demand_NC_cvary[it_count]=excess_demand_NC
       # dP_C_cvary[it_count]=dP_C-lom.LoM_C(alpha, 0, 0, 0, 0,vCoeff_C)
       # it_count+=1
       # print(net_demand_C,coastal_beq)
       # print(net_demand_NC,noncoastal_beq)
       # print(stock_demand_rental,rental_stock)
       # print(depreciation_C,investment_C)
       # print(depreciation_NC,investment_NC)
   # plt.figure(figsize=(8, 5))
   # plt.plot(dP_C_cvary, excess_demand_C_cvary, label='Excess demand for coastal housing')
   # plt.plot(dP_C_cvary, excess_demand_NC_cvary, label='Excess demand for non-coastal housing')

    #plt.xlabel('Deviation from LoM price, $P_C$')
    #plt.ylabel('Excess demand')
    #plt.title(r'Excess housing demand for $\alpha=1$')  # Latex-style rendering of alpha
    #plt.legend()
    #plt.grid(True)
    #plt.tight_layout()
    #plt.show()
    
    #alpha=grids.vAlpha[2]
    #mDist1_c, mDist1_nc, mDist1_renter, rental_stock, coastal_beq, noncoastal_beq, savings_beq, it_counter, total_bequest=sim.stat_dist_finder(grids, par, mMarkov, dPi_S, dPi_L, iNj, vt_stay_nc, vt_stay_c, vt_stay_renter, h_c, h_c, h_renter, b_c_stay, b_renter, b_nc_stay,vCoeff_C,vCoeff_NC,alpha)
    #excess_demand_C_cvary=np.zeros((np.arange(-0.01, 0.011, 0.001).size))
    #excess_demand_NC_cvary=np.zeros((np.arange(-0.01, 0.011, 0.001).size))
    #dP_C_cvary=np.zeros((np.arange(-0.01, 0.011, 0.001).size))
          
    #it_count=0
    #print("Alpha=0")
    #for i in np.arange(-0.001, 0.0010, 0.0001):
        #dP_C=lom.LoM_C(alpha, 0, 0, 0, 0,vCoeff_C)+i
        #dP_NC=lom.LoM_NC(alpha, 0, 0, 0, 0,vCoeff_NC)    
        #excess_demand_C, excess_demand_NC, net_demand_C, net_demand_NC, coastal_beq, noncoastal_beq, investment_C, investment_NC, depreciation_C, depreciation_NC, stock_demand_rental, rental_stock=sim.excess_demand(alpha, Simulated_eps,Simulated_Leps,Simulated_L2eps,Simulated_Lkeps, grids, par, mMarkov, dPi_S, dPi_L, iNj, mDist1_c, mDist1_nc, mDist1_renter, dP_C, dP_NC, vt_stay_nc, vt_stay_c, vt_stay_renter, h_renter, b_c_stay, b_renter, b_nc_stay,rental_stock, coastal_beq, noncoastal_beq, savings_beq,vCoeff_C,vCoeff_NC)
        #excess_demand_C_cvary[it_count]=excess_demand_C
        #excess_demand_NC_cvary[it_count]=excess_demand_NC
       ## dP_C_cvary[it_count]=dP_C-lom.LoM_C(alpha, 0, 0, 0, 0,vCoeff_C)
       # print(excess_demand_C_cvary[it_count],excess_demand_NC_cvary[it_count])
      # it_count+=1       
       # print(net_demand_C,coastal_beq)
        #print(net_demand_NC,noncoastal_beq)
       # print(stock_demand_rental,rental_stock)
       # print(depreciation_C,investment_C)
       # print(depreciation_NC,investment_NC)

    #plt.figure(figsize=(8, 5))
    #plt.plot(dP_C_cvary, excess_demand_C_cvary, label='Excess demand for coastal housing')
    #plt.plot(dP_C_cvary, excess_demand_NC_cvary, label='Excess demand for non-coastal housing')

    #plt.xlabel('Deviation from LoM price, $P_C$')
    #plt.ylabel('Excess demand')
    #plt.title(r'Excess housing demand for $\alpha=0$')  # Latex-style rendering of alpha
    #plt.legend()
    #plt.grid(True)
    #plt.tight_layout()
    #plt.show()
    
    
    

    
    #dP_C_guess, dP_NC_guess, vCoeff_C, vCoeff_NC, iteration=equil.initialise_coefficients(grids, par, dPi_L, dPi_S, iNj, mMarkov, vCoeff_C, vCoeff_NC)
    #print(dP_C_guess, dP_NC_guess, vCoeff_C, vCoeff_NC, iteration)
              
    # dP_C_guess[0],dP_NC_guess[0],it=equil.initialise_debug0(grids, par, dPi_L, dPi_S, iNj, mMarkov, vCoeff_C, vCoeff_NC,w_c, w_nc, w_renter, vt_buy_c,vt_stay_c, vt_buy_nc,vt_stay_nc, vt_stay_renter, c_c, c_nc, c_renter, h_c,h_nc, h_renter, b_c_stay, b_nc_stay, b_c_buy, b_nc_buy, b_renter)
   # print(dP_C_guess[0],dP_NC_guess[0],it)
   
    #dP_C_guess[1],dP_NC_guess[1],it=equil.initialise_debug1(grids, par, dPi_L, dPi_S, iNj, mMarkov, vCoeff_C, vCoeff_NC,w_c, w_nc, w_renter, vt_buy_c,vt_stay_c, vt_buy_nc,vt_stay_nc, vt_stay_renter, c_c, c_nc, c_renter, h_c,h_nc, h_renter, b_c_stay, b_nc_stay, b_c_buy, b_nc_buy, b_renter)
    #print(dP_C_guess[1],dP_NC_guess[1],it)
    
    #dP_C_guess[2],dP_NC_guess[2],it=equil.initialise_debug2(grids, par, dPi_L, dPi_S, iNj, mMarkov, vCoeff_C, vCoeff_NC,w_c, w_nc, w_renter, vt_buy_c,vt_stay_c, vt_buy_nc,vt_stay_nc, vt_stay_renter, c_c, c_nc, c_renter, h_c,h_nc, h_renter, b_c_stay, b_nc_stay, b_c_buy, b_nc_buy, b_renter)
    #print(dP_C_guess[2],dP_NC_guess[2],it)
    
    #dP_C_guess[3],dP_NC_guess[3],it=equil.initialise_debug3(grids, par, dPi_L, dPi_S, iNj, mMarkov, vCoeff_C, vCoeff_NC,w_c, w_nc, w_renter, vt_buy_c,vt_stay_c, vt_buy_nc,vt_stay_nc, vt_stay_renter, c_c, c_nc, c_renter, h_c,h_nc, h_renter, b_c_stay, b_nc_stay, b_c_buy, b_nc_buy, b_renter)
    #print(dP_C_guess[3],dP_NC_guess[3],it)
    
   # dP_C_guess[4],dP_NC_guess[4],it=equil.initialise_debug4(grids, par, dPi_L, dPi_S, iNj, mMarkov, vCoeff_C, vCoeff_NC,w_c, w_nc, w_renter, vt_buy_c,vt_stay_c, vt_buy_nc,vt_stay_nc, vt_stay_renter, c_c, c_nc, c_renter, h_c,h_nc, h_renter, b_c_stay, b_nc_stay, b_c_buy, b_nc_buy, b_renter)
   # print(dP_C_guess[4],dP_NC_guess[4],it)
    
  #  dP_C_guess[5],dP_NC_guess[5],it=equilibrium_debug.initialise_debug5(grids, par, dPi_L, dPi_S, iNj, mMarkov, vCoeff_C, vCoeff_NC,w_c, w_nc, w_renter, vt_buy_c,vt_stay_c, vt_buy_nc,vt_stay_nc, vt_stay_renter, c_c, c_nc, c_renter, h_c,h_nc, h_renter, b_c_stay, b_nc_stay, b_c_buy, b_nc_buy, b_renter)
   # print(dP_C_guess[5],dP_NC_guess[5],it)
    
    #dP_C_guess[6],dP_NC_guess[6],it=equil.initialise_debug6(grids, par, dPi_L, dPi_S, iNj, mMarkov, vCoeff_C, vCoeff_NC,w_c, w_nc, w_renter, vt_buy_c,vt_stay_c, vt_buy_nc,vt_stay_nc, vt_stay_renter, c_c, c_nc, c_renter, h_c,h_nc, h_renter, b_c_stay, b_nc_stay, b_c_buy, b_nc_buy, b_renter)
    #print(dP_C_guess[6],dP_NC_guess[6],it)
    
    #dP_C_guess[7],dP_NC_guess[7],it=equil.initialise_debug7(grids, par, dPi_L, dPi_S, iNj, mMarkov, vCoeff_C, vCoeff_NC,w_c, w_nc, w_renter, vt_buy_c,vt_stay_c, vt_buy_nc,vt_stay_nc, vt_stay_renter, c_c, c_nc, c_renter, h_c,h_nc, h_renter, b_c_stay, b_nc_stay, b_c_buy, b_nc_buy, b_renter)
    #print(dP_C_guess[7],dP_NC_guess[7],it)
    
    #dP_C_guess[8],dP_NC_guess[8],it=equil.initialise_debug8(grids, par, dPi_L, dPi_S, iNj, mMarkov, vCoeff_C, vCoeff_NC,w_c, w_nc, w_renter, vt_buy_c,vt_stay_c, vt_buy_nc,vt_stay_nc, vt_stay_renter, c_c, c_nc, c_renter, h_c,h_nc, h_renter, b_c_stay, b_nc_stay, b_c_buy, b_nc_buy, b_renter)
    #print(dP_C_guess[8],dP_NC_guess[8],it)
    
    #dP_C_guess[9],dP_NC_guess[9],it=equil.initialise_debug9(grids, par, dPi_L, dPi_S, iNj, mMarkov, vCoeff_C, vCoeff_NC,w_c, w_nc, w_renter, vt_buy_c,vt_stay_c, vt_buy_nc,vt_stay_nc, vt_stay_renter, c_c, c_nc, c_renter, h_c,h_nc, h_renter, b_c_stay, b_nc_stay, b_c_buy, b_nc_buy, b_renter)
    #print(dP_C_guess[9],dP_NC_guess[9],it)
    
    #print(dP_C_guess,dP_NC_guess,it)
    
    #vCoeff_out_C, vCoeff_out_NC=equil.initialise_coefficients_debug(grids, dP_C_guess, dP_NC_guess)
    #print(vCoeff_out_C, vCoeff_out_NC)
    #w_c, w_nc, w_renter, vt_buy_c,vt_stay_c, vt_buy_nc,vt_stay_nc, vt_stay_renter, c_c, c_nc, c_renter, h_c,h_nc, h_renter, b_c_stay, b_nc_stay, b_c_buy, b_nc_buy, b_renter = household_problem_epsilons.solve(grids, par, dPi_L, dPi_S, iNj, mMarkov,vCoeff_C,vCoeff_NC)
    #mDist1_c, mDist1_nc, mDist1_renter, rental_stock, coastal_beq, noncoastal_beq, savings_beq, it_counter, total_bequest = sim.stat_dist_finder(grids, par, mMarkov, dPi_S, dPi_L, iNj, vt_stay_nc, vt_stay_c, vt_stay_renter, h_c, h_nc, h_renter, b_c_stay, b_renter, b_nc_stay,vCoeff_C,vCoeff_NC,0)
    #guess_c = 1
    #guess_nc = 1
    #bound_c_l = 0.001
    #bound_c_r = 2
    #bound_nc_l = 0.001
    #bound_nc_r = 2
    #print("Made it to brentq")
    #dP_C_guess, dP_NC_guess, it = equil.house_prices_brentq(grids, par, guess_c, guess_nc, bound_c_l, bound_c_r, bound_nc_l, bound_nc_r, mMarkov, dPi_S, dPi_L, iNj, mDist1_c, mDist1_nc, mDist1_renter, vt_stay_nc, vt_stay_c, vt_stay_renter, h_renter, b_c_stay, b_renter, b_nc_stay, rental_stock, coastal_beq, noncoastal_beq, savings_beq,vCoeff_C, vCoeff_NC,0)
    #w_c, w_nc, w_renter, vt_buy_c,vt_stay_c, vt_buy_nc,vt_stay_nc, vt_stay_renter, c_c, c_nc, c_renter, h_c,h_nc, h_renter, b_c_stay, b_nc_stay, b_c_buy, b_nc_buy, b_renter = household_problem_epsilons.solve(grids, par, dPi_L, dPi_S, iNj, mMarkov,vCoeff_C,vCoeff_NC)
    #alpha=0
    #mDist1_c, mDist1_nc, mDist1_renter, drental_stock, dcoastal_beq, dnoncoastal_beq, dsavings_beq, it_counter, bequests = sim.stat_dist_finder(grids, par, mMarkov, dPi_S, dPi_L, iNj, vt_stay_nc, vt_stay_c, vt_stay_renter, h_c, h_nc, h_renter, b_c_stay, b_renter, b_nc_stay,vCoeff_C,vCoeff_NC,alpha)
    #eps=0
    #Leps=0
    #L2eps=0
    #Lkeps=0
    #for priceshift in range(0,5):
    #    dP_C=lom.LoM_C(alpha,eps,Leps,L2eps,Lkeps,vCoeff_C)-priceshift*0.02
    #    dP_NC=lom.LoM_NC(alpha,eps,Leps,L2eps,Lkeps,vCoeff_NC)-priceshift*0.02
     #t0 = time.time()
      #       t1 = time.time()
    #     # print(dP_C, dP_NC)
    #     print('find p in', t1-t0, 'with', it, 'iterations')
     #   excess_demand_C, excess_demand_NC, net_demand_C, net_demand_NC, coastal_beq, noncoastal_beq, investment_C, investment_NC, depreciation_C, depreciation_NC, stock_demand_rental, rental_stock = sim.excess_demand(alpha,eps,Leps,L2eps,Lkeps, grids, par, mMarkov, dPi_S, dPi_L, iNj, mDist1_c, mDist1_nc, mDist1_renter, dP_C, dP_NC, vt_stay_nc, vt_stay_c, vt_stay_renter, h_renter, b_c_stay, b_renter, b_nc_stay, drental_stock, dcoastal_beq, dnoncoastal_beq, dsavings_beq,vCoeff_C,vCoeff_NC) 
     #   print(excess_demand_C, excess_demand_NC, net_demand_C, net_demand_NC, coastal_beq, noncoastal_beq, investment_C, investment_NC, depreciation_C, depreciation_NC, stock_demand_rental, rental_stock)
       
    # print(dP_C, dP_NC)
    # beta_C = equil.ols_numba(x_matrix, dP_C)
    # beta_NC = equil.ols_numba(x_matrix, dP_NC)
    
    # vCoeff_C[0] = beta_C[0]
    # vCoeff_C[1] = beta_C[1]
    # vCoeff_C[6] = beta_C[2]
    # vCoeff_C[7] = beta_C[3]
    # vCoeff_NC[0] = beta_NC[0]
    # vCoeff_NC[1] = beta_NC[1]
    # vCoeff_NC[6] = beta_NC[2]
    # vCoeff_NC[7] = beta_NC[3]
    
    # print('Coefficients C', vCoeff_C)
    # print('Coefficients NC', vCoeff_NC)
                        

    #output = equil.initial_guess(grids, par, dPi_L, dPi_S, iNj, mMarkov,vCoeff)
    # w_c, w_nc, w_renter, vt_buy_c,vt_stay_c, vt_buy_nc,vt_stay_nc, vt_stay_renter, c_c, c_nc, c_renter, h_c,h_nc, h_renter, b_c_stay, b_nc_stay, b_c_buy, b_nc_buy, b_renter = household_problem_epsilons.solve(grids, par, dPi_L, dPi_S, iNj, mMarkov,vCoeff_C,vCoeff_NC)
    # mDist0_c, mDist0_nc, mDist0_renter, rental_stock, coastal_beq, noncoastal_beq, savings_beq, it_counter, bequests = sim.stat_dist_finder(grids, par, mMarkov, dPi_S, dPi_L, iNj, vt_stay_nc, vt_stay_c, vt_stay_renter, h_c, h_nc, h_renter, b_c_stay, b_renter, b_nc_stay,vCoeff_C,vCoeff_NC,alpha)
    # lower_check=0.9
    # upper_check=1.25
    # stepsize=0.05
    # price_check, nrchecks=equil.generate_pricecheck(lower_check, upper_check, stepsize)
    

    # alpha=0
    # w_c, w_nc, w_renter, vt_buy_c,vt_stay_c, vt_buy_nc,vt_stay_nc, vt_stay_renter, c_c, c_nc, c_renter, h_c,h_nc, h_renter, b_c_stay, b_nc_stay, b_c_buy, b_nc_buy, b_renter = household_problem_epsilons.solve(grids, par, dPi_L, dPi_S, iNj, mMarkov,vCoeff_C,vCoeff_NC)
    # mDist0_c, mDist0_nc, mDist0_renter, rental_stock, coastal_beq, noncoastal_beq, savings_beq, it_counter, total_bequest, errorterm_c, errorterm_nc,net_demand_C_UD,net_demand_NC_UD= sim.stat_dist_finder(grids, par, mMarkov, dPi_S, dPi_L, iNj, vt_stay_nc, vt_stay_c, vt_stay_renter, h_c, h_nc, h_renter, b_c_stay, b_renter, b_nc_stay,vCoeff_C,vCoeff_NC,alpha)
    # excess_demand_C, excess_demand_NC, net_demand_C, net_demand_NC, coastal_beq, noncoastal_beq, net_investment_C, net_investment_NC, stock_demand_rental, rental_stock, bequests_coastal_ED, bequests_noncoastal_ED=sim.excess_demand(alpha,eps,Leps,L2eps,Lkeps, grids, par, mMarkov, dPi_S, dPi_L, iNj, mDist0_c, mDist0_nc, mDist0_renter, dP_C, dP_NC, vt_stay_nc, vt_stay_c, vt_stay_renter, h_renter, b_c_stay, b_renter, b_nc_stay, rental_stock, coastal_beq, noncoastal_beq, savings_beq,vCoeff_C,vCoeff_NC) 
    # print(excess_demand_C, excess_demand_NC, net_demand_C, net_demand_NC, coastal_beq, noncoastal_beq, bequests_coastal_ED, bequests_noncoastal_ED, stock_demand_rental, rental_stock)    
    # print(net_demand_C_UD,net_demand_NC_UD)

    
    # print('With P_c', round(dP_C,3), 'and P_nc', round(dP_NC,3),'excess demand for C', round(excess_demand_C,4), 'and non C', round(excess_demand_NC,4))
    # print('With P_c', round(dP_C,3), 'and P_nc', round(dP_NC,3),'net demand for C', round(net_demand_C,4), ' investment for C', round(net_investment_C,4), 'bequests c',round(coastal_beq,4))
    # print('With P_c', round(dP_C,3), 'and P_nc', round(dP_NC,3),'net demand for NC', round(net_demand_NC,4), ' investment for NC', round(net_investment_NC,4), 'bequests nc',round(noncoastal_beq,4))
 
    # t1 = time.time()
    
    # print(stock_tot)
    
    
    # print(excess_demand_c_opt)
    # print(net_demand_c)
    
    # plt.plot(c_prices,excess_demand_c_opt[:,0] , label = 'Alpha = 0')
    # plt.plot(c_prices,excess_demand_c_opt[:,1] , label = 'Alpha = 0.5')
    # plt.plot(c_prices,excess_demand_c_opt[:,2] , label = 'Alpha = 1')
    # plt.title( 'Net demand for Coastal housing with a fixed non-coastal price, optimists')
    # plt.xlabel('Price C')
    # plt.legend(loc = 1)
    # plt.show()
    # plt.clf()
    
    # plt.plot(c_prices,excess_demand_c_real[:,0] , label = 'Alpha = 0')
    # plt.plot(c_prices,excess_demand_c_real[:,1] , label = 'Alpha = 0.5')
    # plt.plot(c_prices,excess_demand_c_real[:,2] , label = 'Alpha = 1')
    # plt.title( 'Net demand for Coastal housing with a fixed non-coastal price, realists')
    # plt.xlabel('Price C')
    # plt.legend(loc = 1)
    # plt.show()
    # plt.clf()
    
    # plt.plot(c_prices,excess_demand_nc_opt[:,0] , label = 'Alpha = 0')
    # plt.plot(c_prices,excess_demand_nc_opt[:,1] , label = 'Alpha = 0.5')
    # plt.plot(c_prices,excess_demand_nc_opt[:,2] , label = 'Alpha = 1')
    # plt.title( 'Net demand for non-Coastal housing with a fixed non-coastal price, optimists')
    # plt.xlabel('Price C')
    # plt.legend(loc = 4)
    # plt.show()
    # plt.clf()
    
    # plt.plot(c_prices,excess_demand_nc_real[:,0] , label = 'Alpha = 0')
    # plt.plot(c_prices,excess_demand`_nc_real[:,1] , label = 'Alpha = 0.5')
    # plt.plot(c_prices,excess_demand_nc_real[:,2] , label = 'Alpha = 1')
    # plt.title( 'Net demand for non-Coastal housing with a fixed non-coastal price, realists')
    # plt.xlabel('Price C')
    # plt.legend(loc = 4)
    # plt.show()
    # plt.clf()
    
    # plt.plot(c_prices,net_demand_c[:,0] , label = 'Alpha = 0')
    # plt.plot(c_prices,net_demand_c[:,1] , label = 'Alpha = 0.5')
    # plt.plot(c_prices,net_demand_c[:,2] , label = 'Alpha = 1')
    # plt.title( 'Excess demand for Coastal housing with a fixed non-coastal price')
    # plt.xlabel('Price C')
    # plt.legend(loc = 1)
    # plt.show()
    # plt.clf()
    
    # plt.plot(c_prices,net_demand_nc[:,0] , label = 'Alpha = 0')
    # plt.plot(c_prices,net_demand_nc[:,1] , label = 'Alpha = 0.5')
    # plt.plot(c_prices,net_demand_nc[:,2] , label = 'Alpha = 1')
    # plt.title( 'Excess demand for non-Coastal housing with a fixed non-coastal price')
    # plt.xlabel('Price C')
    # plt.legend(loc = 4)
    # plt.show()
    # plt.clf()
    
    # w_c, w_nc, w_renter, vt_buy_c,vt_stay_c, vt_buy_nc,vt_stay_nc, vt_stay_renter, c_c, c_nc, c_renter, h_c,h_nc, h_renter, b_c_stay, b_nc_stay, b_c_buy, b_nc_buy, b_renter = household_problem_epsilons.solve(grids, par, dPi_L, dPi_S, iNj, mMarkov, vCoeff_C, vCoeff_NC)
    
    # renter = np.zeros(grids.vAlpha.size)
    # coastal = np.zeros(grids.vAlpha.size)
    # noncoastal = np.zeros(grids.vAlpha.size)
    # total= np.zeros(grids.vAlpha.size)
    # vCoeff_C=np.array([0.95,-0.1,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
    # for alpha_idx in range(grids.vAlpha.size):
    #     alpha_guess = grids.vAlpha[alpha_idx]
    #     mDist1_c, mDist1_nc, mDist1_renter, rental_stock, coastal_beq, noncoastal_beq, savings_beq, it_counter, total_bequest = sim.stat_dist_finder(grids, par, mMarkov, dPi_S, dPi_L, iNj, vt_stay_nc, vt_stay_c, vt_stay_renter, h_c, h_nc, h_renter, b_c_stay, b_renter, b_nc_stay,vCoeff_C,vCoeff_NC,alpha_guess)
    #     renter[alpha_idx] = np.sum(mDist1_renter[:,:,:,:,:])
    #     coastal[alpha_idx] = np.sum(mDist1_c[:,:,:,:,:])
    #     noncoastal[alpha_idx] = np.sum(mDist1_nc[:,:,:,:,:])
    
    #     total[alpha_idx] = renter[alpha_idx] + coastal[alpha_idx] + noncoastal[alpha_idx]
    #     renter[alpha_idx] /= total[alpha_idx]
    #     coastal[alpha_idx] /= total[alpha_idx]
    #     noncoastal[alpha_idx] /= total[alpha_idx]
    
    # t0 = time.time()
    # rootfinding_output = equil.initial_guess(grids, par, dPi_L, dPi_S, iNj, mMarkov,vCoeff_C, vCoeff_NC)
    # print(rootfinding_output)
    # t1 = time.time()
    # print('Computation time to find market-clearing price is', t1-t0, ".")
    
    # vCoeff_C=np.array([1.0,-0.001,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]) #Initial coefficient guess
    # vCoeff_NC=np.array([1.0,0.001,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
    # dP_C_guess, dP_NC_guess, vCoeff_C, vCoeff_NC, it = equil.initialise_coefficients(grids, par, dPi_L, dPi_S, iNj, mMarkov, vCoeff_C, vCoeff_NC)
    # t1 = time.time()
    # print('Computation time to find market-clearing price is', t1-t0, ".")
    # print(dP_C_guess, dP_NC_guess, vCoeff_C, vCoeff_NC, it)

    
    # w_c, w_nc, w_renter, vt_buy_c,vt_stay_c, vt_buy_nc,vt_stay_nc, vt_stay_renter, c_c, c_nc, c_renter, h_c,h_nc, h_renter, b_c_stay, b_nc_stay, b_c_buy, b_nc_buy, b_renter = household_problem_epsilons.solve(grids, par, dPi_L, dPi_S, iNj, mMarkov,vCoeff)
    # mDist1_c, mDist1_nc, mDist1_renter, rental_stock, coastal_beq, noncoastal_beq, savings_beq, it_counter, bequests = sim.stat_dist_finder(grids, par, mMarkov, dPi_S, dPi_L, iNj, vt_stay_nc, vt_stay_c, vt_stay_renter, h_c, h_nc, h_renter, b_c_stay, b_renter, b_nc_stay,vCoeff)
    # alpha=0
    # eps=0
    # Leps=0
    # L2eps=0
    # Lkeps=0
    
    # dP_NC = 1.16682129
    # dP_C = np.linspace(0.8,1.3,15)
    # total_imbalance = np.zeros((dP_C.size))
    # excess_demand_C = np.zeros((dP_C.size))
    # excess_demand_NC = np.zeros((dP_C.size))
    # demand_C= np.zeros((dP_C.size)) 
    # supply_C = np.zeros((dP_C.size)) 
    # demand_NC= np.zeros((dP_C.size)) 
    # supply_NC = np.zeros((dP_C.size)) 
    # vnoncoastal_beq= np.zeros((dP_C.size)) 
    # vcoastal_beq= np.zeros((dP_C.size)) 
    # depreciation_C= np.zeros((dP_C.size)) 
    # depreciation_NC= np.zeros((dP_C.size)) 
    # investment_C= np.zeros((dP_C.size)) 
    # investment_NC= np.zeros((dP_C.size)) 

    
    # for i in range(dP_C.size):
    #     price = np.array([dP_C[i], dP_NC])
    #     total_imbalance[i], excess_demand_C[i], excess_demand_NC[i], demand_C[i], supply_C[i], demand_NC[i], supply_NC[i], vnoncoastal_beq[i], vcoastal_beq[i], depreciation_C[i], depreciation_NC[i], investment_C[i], investment_NC[i] = equil.market_imbalance(price,alpha,eps,Leps,L2eps,Lkeps, grids, par, mMarkov, dPi_S, dPi_L, iNj, mDist1_c, mDist1_nc, mDist1_renter, vt_stay_nc, vt_stay_c, vt_stay_renter, h_renter, b_c_stay, b_renter, b_nc_stay, rental_stock, coastal_beq, noncoastal_beq, savings_beq,vCoeff)
    
    # plt.plot(dP_C,excess_demand_C , label = 'excess demand C')
    # plt.plot(dP_C,excess_demand_NC , label = 'excess demand NC')
    # plt.plot(dP_C,total_imbalance , label = 'Total')
    # plt.title( 'excess demand')
    # plt.xlabel('dP_C')
    # plt.legend(loc = 4)
    # plt.show()
    # plt.clf()
    
    
    # plt.plot(dP_C,demand_C , label = 'excess demand C')
    # plt.plot(dP_C,supply_C , label = 'Supply C')
    # plt.plot(dP_C,vcoastal_beq , label = 'Bequests C')
    # plt.plot(dP_C, depreciation_C, label = 'depreciation C')
    # plt.plot(dP_C, investment_C, label = 'Investment C')
    # plt.title( 'Coastal market')
    # plt.xlabel('dP_C')
    # plt.legend(loc = 4)
    # plt.show()
    # plt.clf()
    
 """


###########################################################

### start main
if __name__ == "__main__":
    #import cProfile
    #import pstats
    #profiler = cProfile.Profile()
    #profiler.enable()

    main()

    #profiler.disable()
    #stats = pstats.Stats(profiler).sort_stats("cumtime")
    #stats.print_stats()   # show top 20 slowest functions
