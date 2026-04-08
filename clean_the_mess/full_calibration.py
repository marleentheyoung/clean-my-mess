"""
Calibration_full.py
"""

import numpy as np
import time
import equilibrium as equil
import misc_functions as misc
import tauchen as tauch
import grid_creation as grid_creation
import household_problem_epsilons_nolearning as household_problem
import simulation as sim
import moments as find_moments
import LoM_epsilons as lom
import nlopt






def f(x,grad):
    # dbeta, dNu, db_bar, dOmega, domega_g, dPsi, dPhi
    dBeta = x[0]
    dNu = x[1]
    db_bar =x[2]
    dOmega =x[3]
    omega_g =x[4]
    dPsi =x[5]
    dPhi = x[6]
    
    vCoeff_C_initial = np.array([0.6732231829966053 , 0., 0., 0., 0.])
    vCoeff_NC_initial = np.array([0.7470390048443882, 0., 0., 0.])
    bequest_guess= np.zeros((3))
    dPi_L = 0.01


    time_increment=2
    vPi_S_median=np.array([0.0194,    0.0198,    0.0202,    0.0206,    0.0210,    0.0214,    0.0218,    0.0222,    0.0226,    0.0230,    0.0234,    0.0239,    0.0243,
                           0.0248,    0.0254,    0.0259,    0.0265,    0.0273,    0.0280,   0.0289,    0.0300,    0.0310,    0.0321,    0.0333,    0.0347,    0.0361,
    0.0376,    0.0392,    0.0410,    0.0427,    0.0444,    0.0461,    0.0478,    0.0495,    0.0513,    0.0530,    0.0547,    0.0565,    0.0583,
    0.0601,    0.0619,    0.0637,    0.0654,    0.0672,    0.0690,    0.0708,    0.0726,    0.0744,    0.0762,    0.0780,    0.0798,    0.0816])
    vPi_S_median=1-(1-vPi_S_median)**time_increment    
    
    
    par_dict = {"time_increment": time_increment,
              "iNj": 30,
              "j_ret": 23,
              "dBeta": dBeta**time_increment, 
              "dDelta": 1-(1-0.015)**time_increment, 
              "dDelta_deprec_rental": 1-(1-0.015)**time_increment,
              "dDelta_default": 0,
              "r": 1.03**time_increment-1, 
              #"dDelta_rental": 1.038**time_increment-1, #From KMV - r plus psi=0.008
              "dPsi": dPsi,
              "r_m": 1.04**time_increment-1, 
              'vPi_S_median': vPi_S_median,
              'dPi_S_initial': vPi_S_median[0],
              "dKappa_sell": 0.07,
              "dKappa_buy": 0,
              "dXi_foreclosure": 0.8,
              "dNu": dNu,
              "dZeta": 0.01, 
              "dZeta_fixed": 1/26, 
              "lambda_pti":0.25,
              "max_ltv": 0.95,
              "damage_states": 3,
              "dLambda": 0.8,
              "dGamma": 1/1.25,
              "dSigma": 2,
              "b_bar":db_bar,
              "dPhi":  dPhi,
              "nonlingrid": 1,
              "nonlingrid_big": 1,  
              #"iNb_left_tail": 20,
              #"iNb_left": 50,
              #"iNb_right": 10,
              "iNb":60,
              "iBmin": 0, 
              "iBmax": min(20, 20*((1-0.964)/(1-0.97348525))),
              "dZ":0.8,
              "h_max": 5.15,
              "dXi_min":1,
              "dXi_max": 1 + omega_g,
              "iXin":4,
              "alpha_0": 0.4,
              "dRho": 0.97, 
              "dSigmaeps": 0.20, 
              "dSigmaeps_trans": 0.05, 
              "iNumStates":5, #THIS NUMBER MUST BE UNEVEN
              "iNumTrans":3, 
              "iM":1,
              'vAgeEquiv':np.ones(50),
              'dNC_frac': 0.5,
              'dC_frac': 0.5,
              'dTheta': 1.5/2.5, 
              'dL': 0.311,
              'dOmega': dOmega,
              'sd_income_initial': .3228617,
              'beta0_nowealth': -.9910767, #=Constant in logit regression nowealth=1 on log income in lowest age bracket
              'beta1_nowealth': -.3417672, #=Coeff on log income (in terms of sds away from the median) in logit regression nowealth=1 on log income in lowest age bracket
              'var_poswealth': 1.557037, #=Variance log wealth conditional on wealth>0 in lowest age bracket
              'beta0_age': 2.0017337, #=Constant of fitted line age --> mean log income 
              'beta1_age': .78795906  , #=Coeff of age for fitted line age --> mean log income 
              'beta2_age': -.02717023, #=Coeff of age^2 for fitted line age --> mean log income  
              'beta3_age': .00042535, #=Coeff of age^3 for fitted line age --> mean log income 
              'beta4_age':  -2.505727467e-06, #=Coeff of age^4 for fitted line age --> mean log income 
              'corr_poswealth_income': 0.1984, #=Correlation between log income (in terms of sds away from the median) and log wealth conditional on wealth>0 in lowest age bracket
              'tau_0': 0.75,
              'tau_1': 0.151

              }

    
    par = misc.construct_jitclass(par_dict)
    mMarkov, vE = tauch.tauchen(par.dRho, par.dSigmaeps, par.iNumStates, par.iM, par.time_increment)
    grids, mMarkov=grid_creation.create(par)        
    method='secant'
    dP_C_guess, dP_NC_guess, vCoeff_C_new, vCoeff_NC_new, mDist1_c, mDist1_nc, mDist1_renter, rental_stock_C, rental_stock_NC, coastal_beq, noncoastal_beq, savings_beq, no_beq, iteration=equil.initialise_coefficients_initial(par, grids, method, dPi_L, par.iNj, mMarkov, vCoeff_C_initial, vCoeff_NC_initial, bequest_guess)      
    vt_stay_c, vt_stay_nc, vt_renter, b_stay_c, b_stay_nc, b_renter = household_problem.solve_initial(grids, par, dPi_L, par.iNj, mMarkov, vCoeff_C_new[0],vCoeff_NC_new[0])
    mDist1_c, mDist1_nc, mDist1_renter, rental_stock_C, rental_stock_NC, coastal_beq, noncoastal_beq, savings_beq, vcoastal_beq, vnoncoastal_beq, vsavings_beq, no_beq=sim.stat_dist_finder(False, grids, par, mMarkov, dPi_L, par.iNj, vt_stay_c[0,], vt_stay_nc[0,], vt_renter[0,], b_stay_c[0,], b_stay_nc[0,], b_renter[0,], vCoeff_C_new,vCoeff_NC_new, np.zeros((3)))
    dP_C_lom=lom.LoM_C(grids,0, vCoeff_C_new)
    dP_NC_lom=lom.LoM_NC(grids,0, vCoeff_NC_new)
    
    # MODEL MOMENTS
    HO_C_share, HO_NC_share, R_C_share, R_NC_share, HO_C_share_before35, HO_NC_share_before35, HO_C_share_death, HO_NC_share_death, total_NW_HO_C, total_NW_HO_NC, total_NW_R, total_NW_HO, total_NW_age_15, total_NW_age_27, total_NW_all_ages, median_NW_age_15, median_NW_age_27, median_NW_all_ages, thirtythree_percentile_NW_age_27, sixtyseven_percentile_NW_age_27, thirtythree_percentile_NW_age_30, sixtyseven_percentile_NW_age_30, tenth_percentile_housing, median_housing, ninetieth_percentile_housing, cumdens_housing_all_ages, NW_housing_share_sorted=find_moments.calc_moments(par, grids, 0, mDist1_c, mDist1_nc,mDist1_renter, grids.vPi_S_median[0], dPi_L, vCoeff_C_new, vCoeff_NC_new)
    total_saving_model = median_NW_all_ages
    NW_decay_model = total_NW_age_27/total_NW_age_15
    bequest_ineq_model = sixtyseven_percentile_NW_age_30/thirtythree_percentile_NW_age_30
    homeownership_model = HO_C_share+HO_NC_share
    price_diff_model = (dP_C_lom-dP_NC_lom)/dP_NC_lom
    homeownership_young_model = HO_C_share_before35 + HO_NC_share_before35
    med_housing_model = median_housing
    
    
    # DATA MOMENTS
    total_saving_data = 1.2
    NW_decay_data = 1.51
    bequest_ineq_data = 3.24
    homeownership = 0.66
    price_diff = -0.114
    homeownership_young = 0.39
    med_housing = 0.5

    
    sq_saving       = ((total_saving_data-total_saving_model)/total_saving_data)**2
    sq_nw           = ((NW_decay_data-NW_decay_model)/NW_decay_data)**2
    sq_ineqnw       = ((bequest_ineq_data-bequest_ineq_model)/bequest_ineq_data)**2
    sq_homeownership = ((homeownership - homeownership_model)/homeownership)**2
    sq_price_diff     = ((price_diff - price_diff_model)/price_diff)**2
    sq_homeownership_young = ((homeownership_young - homeownership_young_model)/homeownership_young)**2
    sq_med_housing    = ((med_housing - med_housing_model)/med_housing)**2
    weights = np.array([1,1,1,1.5,1,1,1])
    
    squaredsum =  weights[0]*sq_saving + weights[1]*sq_nw + weights[2]*sq_ineqnw + weights[3]*sq_homeownership + weights[4]*sq_price_diff +  weights[5]*sq_homeownership_young + weights[6]*sq_med_housing 
    
    return squaredsum

def main():


    t0 = time.time()
     
    # Define bounds for each parameter: # dbeta, dNu, db_bar, dOmega, domega_g, dPsi, dPhi
    lb = [0.94,  10,  0.1, 0,     0.02, 0.003,  0.1]  # Lower bounds
    ub = [0.965, 50,  5,   0.01,  0.06, 0.006,  0.15]   # Upper bounds
    
    # Define optimization problem
    opt = nlopt.opt(nlopt.G_MLSL_LDS, 7)
    opt.set_lower_bounds(lb)
    opt.set_upper_bounds(ub)
    opt.set_min_objective(f)
    opt.set_local_optimizer(nlopt.opt(nlopt.LN_NELDERMEAD, 7))
    opt.set_xtol_rel(1e-6)  # Local convergence criteria
    opt.set_xtol_abs(1e-6)
    
    # Set up the multi-start algorithm
    opt.set_maxeval(75)  # Maximum number of evaluations
    opt.set_population(200)  # Number of starting points
    
    # Optimize
    # x = [dbeta, deta, b_bar, dgamma, s_bar,A_g, A_r, A_kf, A_ec, A_ffc]
    x_opt = opt.optimize([0.95, 32, 2, 0.005, 0.04, 0.004, 0.12])
    t1 = time.time()
    print("Calibration time:", t1-t0, ".")
    # Print results
    print("Optimized parameters:", x_opt)
    
###########################################################
### start main
if __name__ == "__main__":
    main()
