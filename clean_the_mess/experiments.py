import numpy as np
import household_problem as household_problem
import simulation as sim
import lom as lom
import equilibrium as equil
import grid_creation as grid_creation
from numba import njit




@njit
def full_information_shock(grids, par, method, iNj, mMarkov, vCoeff_C_experiment, vCoeff_NC_experiment, price_history, mDist1_c, mDist1_nc, mDist1_renter, rental_stock_C, rental_stock_NC, coastal_beq, noncoastal_beq, savings_beq):
     
    mDist_c_start = np.zeros((par.iNj, 1, grids.vG.size, grids.vM_sim.size, grids.vH.size, grids.vL_sim.size,grids.vE.size))
    mDist_nc_start = np.zeros((par.iNj, 1, grids.vG.size, grids.vM_sim.size, grids.vH.size, grids.vL_sim.size,grids.vE.size))
    mDist_renter_start = np.zeros((par.iNj, 1, grids.vG.size, grids.vX_sim.size, grids.vE.size))
    
    
    mDist_c_start[:,0,:,:,:,:,:] = mDist1_c[:,0,:,:,:,:,:]+mDist1_c[:,1,:,:,:,:,:]
    mDist_nc_start[:,0,:,:,:,:,:] = mDist1_nc[:,0,:,:,:,:,:]+mDist1_nc[:,1,:,:,:,:,:]
    mDist_renter_start[:,0,:,:,:] = mDist1_renter[:,0,:,:,:]+mDist1_renter[:,1,:,:,:]
    
    
    dP_C_initial=price_history[0,-2]
    dP_NC_initial=price_history[1,-2]
    sceptics=False
    dP_C_vec_experiment, dP_NC_vec_experiment, vCoeff_C_experiment, vCoeff_NC_experiment, iteration, vt_stay_c, vt_stay_nc, vt_renter, b_stay_c, b_stay_nc, b_renter, vt_stay_c_wf, vt_stay_nc_wf, vt_renter_wf=equil.find_coefficients(par, grids, method, sceptics, iNj, mMarkov, vCoeff_C_experiment, vCoeff_NC_experiment,dP_C_initial, dP_NC_initial,mDist_c_start, mDist_nc_start, mDist_renter_start, rental_stock_C, rental_stock_NC, coastal_beq, noncoastal_beq, savings_beq)
    
    return dP_C_vec_experiment, dP_NC_vec_experiment, vCoeff_C_experiment, vCoeff_NC_experiment, vt_stay_c, vt_stay_nc, vt_renter, b_stay_c, b_stay_nc, b_renter, vt_stay_c_wf, vt_stay_nc_wf, vt_renter_wf

@njit
def gen_distribution_now(grids, par, func, method, mMarkov, vCoeff_C, vCoeff_NC, vCoeff_C_initial, vCoeff_NC_initial):
    
    initial=True 
    sceptics=True
    vt_stay_c, vt_stay_nc, vt_renter, b_stay_c, b_stay_nc, b_renter = household_problem.solve_ss(grids, par, par.iNj, mMarkov, vCoeff_C_initial[0],vCoeff_NC_initial[0], initial, sceptics, False)                                              
    mDist0_c, mDist0_nc, mDist0_renter, rental_stock_C0, rental_stock_NC0, coastal_beq0, noncoastal_beq0, savings_beq0, _, _, _, _, coastal_mass_J, noncoastal_mass_J, renter_mass_J=sim.stat_dist_finder(sceptics, grids, par, mMarkov, par.iNj, vt_stay_c[0,], vt_stay_nc[0,], vt_renter[0,], b_stay_c[0,], b_stay_nc[0,], b_renter[0,], vCoeff_C_initial,vCoeff_NC_initial, np.zeros((3)), initial)
    
    experiment=True
    price_history, mDist1_c, mDist1_nc, mDist1_renter, stock_demand_rental_C, stock_demand_rental_NC, vcoastal_beq, vnoncoastal_beq, vsavings_beq, _, _, _, _, _, _, coastal_stock, noncoastal_stock, rental_stock=equil.generate_pricepath(grids, par, func, mMarkov, vCoeff_C,vCoeff_NC, vCoeff_C_initial[0], vCoeff_NC_initial[0], mDist0_c, mDist0_nc, mDist0_renter, rental_stock_C0, rental_stock_NC0, coastal_beq0, noncoastal_beq0, savings_beq0, coastal_mass_J, noncoastal_mass_J, renter_mass_J, method, sceptics, experiment)
    return price_history, mDist0_c, mDist0_nc, mDist0_renter, mDist1_c, mDist1_nc, mDist1_renter, stock_demand_rental_C, stock_demand_rental_NC, vcoastal_beq, vnoncoastal_beq, vsavings_beq

def full_information_experiment(par, func, method,  vCoeff_C, vCoeff_NC, vCoeff_C_experiment, vCoeff_NC_experiment, vCoeff_C_initial, vCoeff_NC_initial):

    grids, mMarkov=grid_creation.create(par)
    experiment=True
    price_history, _, _, _,  mDist1_c, mDist1_nc, mDist1_renter, stock_demand_rental_C, stock_demand_rental_NC, vcoastal_beq, vnoncoastal_beq, vsavings_beq=gen_distribution_now(grids, par, func, method, mMarkov, vCoeff_C, vCoeff_NC, vCoeff_C_initial, vCoeff_NC_initial)
    grids_exp, mMarkov=grid_creation.create(par, experiment)
    dP_C_vec_experiment, dP_NC_vec_experiment, vCoeff_C_experiment, vCoeff_NC_experiment, vt_stay_c, vt_stay_nc, vt_renter, b_stay_c, b_stay_nc, b_renter, vt_stay_c_wf, vt_stay_nc_wf, vt_renter_wf=full_information_shock(grids_exp, par, method, par.iNj, mMarkov, vCoeff_C_experiment, vCoeff_NC_experiment, price_history, mDist1_c, mDist1_nc, mDist1_renter, stock_demand_rental_C, stock_demand_rental_NC, vcoastal_beq[-1], vnoncoastal_beq[-1], vsavings_beq[-1])
    return price_history, dP_C_vec_experiment, dP_NC_vec_experiment, vCoeff_C_experiment, vCoeff_NC_experiment, vt_stay_c, vt_stay_nc, vt_renter, b_stay_c, b_stay_nc, b_renter, vt_stay_c_wf, vt_stay_nc_wf, vt_renter_wf