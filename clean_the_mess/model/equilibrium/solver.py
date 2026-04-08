"""
equilibrium.py

Purpose:
    Find the pricing forecasting rule consistent with agents' behaviour
"""
import numpy as np
import numba as nb
from numba import prange
from numba import njit
import model.household.vfi as household_problem
import model.simulation.distribution as sim
import model.lom as lom
import math
import time
import model.utils as misc


@njit
def flatten_third_dim(mat):
    I, J, K = mat.shape
    # final shape will be (I, J*K)
    out = np.empty((I*K, J), dtype=np.float64)
    for i in range(I):
        for k in range(K):
            for j in range(J):
                out[k*I + i, j] = mat[i, j, k] 
    return out

@njit
def generate_pricepath(grids, par, func, mMarkov, vCoeff_in_C,vCoeff_in_NC, dP_C_initial, dP_NC_initial, mDist0_c, mDist0_nc, mDist0_renter, rental_stock_C0, rental_stock_NC0, coastal_beq0, noncoastal_beq0, savings_beq0, coastal_mass_J, noncoastal_mass_J, renter_mass_J, method, sceptics, experiment=False, welfare=False, plot_stocks=False):
    """Simulate the full price path and distribution evolution over the transition.

    Solves the VFI once for the given coefficients, then steps forward through
    time: at each period, clears both housing markets (or uses LoM prices if
    welfare=True), updates the distribution, and records the price history.

    Args:
        grids: Numba jitclass with model grids.
        par: Numba jitclass with model parameters.
        func: If False, find market-clearing prices at each step.
            If True, use LoM-implied prices (skip market clearing).
        mMarkov: 2D array, income Markov transition matrix.
        vCoeff_in_C: 1D array (5,), coastal LoM coefficients.
        vCoeff_in_NC: 1D array (5,), non-coastal LoM coefficients.
        dP_C_initial: Scalar, initial coastal price (= vCoeff_C_initial[0]).
        dP_NC_initial: Scalar, initial non-coastal price (= vCoeff_NC_initial[0]).
        mDist0_c: Initial coastal owner distribution.
        mDist0_nc: Initial non-coastal owner distribution.
        mDist0_renter: Initial renter distribution.
        rental_stock_C0: Initial coastal rental stock.
        rental_stock_NC0: Initial non-coastal rental stock.
        coastal_beq0: Initial coastal bequest aggregate.
        noncoastal_beq0: Initial non-coastal bequest aggregate.
        savings_beq0: Initial savings bequest aggregate.
        coastal_mass_J: Terminal-age coastal owner mass by belief type.
        noncoastal_mass_J: Terminal-age non-coastal owner mass by belief type.
        renter_mass_J: Terminal-age renter mass by belief type.
        method: String, market-clearing algorithm ('secant' or 'bisection').
        sceptics: If True, include both belief types.
        experiment: If True, truncate time horizon to experiment year.
        welfare: If True, skip market clearing and use LoM prices
            (for welfare analysis where prices are already known).
        plot_stocks: If True, track housing stock evolution over time.

    Returns:
        price_history: 2D array (T, 2), market-clearing prices [P_C, P_NC] per period.
        mDist1_c: Final coastal owner distribution.
        mDist1_nc: Final non-coastal owner distribution.
        mDist1_renter: Final renter distribution.
        stock_demand_rental_C1: Final coastal rental stock.
        stock_demand_rental_NC1: Final non-coastal rental stock.
        vcoastal_beq: 1D array, coastal bequests per period.
        vnoncoastal_beq: 1D array, non-coastal bequests per period.
        vsavings_beq: 1D array, savings bequests per period.
        vt_stay_c: Coastal stayer policy values from VFI.
        vt_stay_nc: Non-coastal stayer policy values from VFI.
        vt_renter: Renter policy values from VFI.
        v_owner_c_wf: Welfare values for coastal owners.
        v_owner_nc_wf: Welfare values for non-coastal owners.
        v_nonowner_wf: Welfare values for renters.
        coastal_stock: Housing stock by belief type over time (if plot_stocks).
        noncoastal_stock: Housing stock by belief type over time.
        rental_stock: Rental stock by belief type over time.
    """


    vt_stay_c, vt_stay_nc, vt_renter, b_stay_c, b_stay_nc, b_renter, v_owner_c_wf, v_owner_nc_wf, v_nonowner_wf = household_problem.solve(grids, par, par.iNj, mMarkov,vCoeff_in_C,vCoeff_in_NC, sceptics, welfare)
    dP_C_lag=dP_C_initial
    dP_NC_lag=dP_NC_initial
    
    if experiment==True:
        price_history=np.zeros((int((2026-1998)/par.time_increment)+1,2))
        nperiods=int((2026-1998)/par.time_increment)+1
        
    else:
        price_history=np.zeros((grids.vTime.size,2))
        nperiods=grids.vTime.size
        
    vcoastal_beq=np.zeros((nperiods-1))
    vnoncoastal_beq=np.zeros((nperiods-1))
    vsavings_beq=np.zeros((nperiods-1))
    
    
    if sceptics==True:
        k_dim=2
    else:
        k_dim=1
    
    coastal_stock=np.zeros((grids.vTime.size,k_dim))
    noncoastal_stock=np.zeros((grids.vTime.size,k_dim))
    rental_stock=np.zeros((grids.vTime.size,k_dim))
    
    if plot_stocks==True:
        for k_index in range(k_dim):
            coastal_stock[0,k_index]=np.sum(mDist0_c[:, k_index, :, :, :, :, :])+coastal_mass_J[k_index]
            noncoastal_stock[0,k_index]=np.sum(mDist0_nc[:, k_index, :, :, :, :, :])+noncoastal_mass_J[k_index]
            rental_stock[0,k_index]=np.sum(mDist0_renter[1:, k_index, :, :, :])+renter_mass_J[k_index]

    
    
    for t_index in range(nperiods):  

        if t_index==0:
            guess_c = lom.LoM_C(grids,t_index,vCoeff_in_C)
            guess_nc = lom.LoM_NC(grids,t_index,vCoeff_in_NC)

        else:
            guess_c = lom.LoM_C(grids,t_index, vCoeff_in_C)+(price_history[t_index-1,0]-lom.LoM_C(grids,t_index-1, vCoeff_in_C))
            guess_nc = lom.LoM_NC(grids,t_index, vCoeff_in_NC)+(price_history[t_index-1,1]-lom.LoM_NC(grids,t_index-1, vCoeff_in_NC))
        bound_c_l= 0.1
        bound_nc_l= 0.1 
        
        bound_c_l_bis=guess_c-0.1
        bound_c_r_bis=guess_c+0.1
        bound_nc_l_bis=guess_nc-0.1
        bound_nc_r_bis=guess_nc+0.1            
        
        if not welfare and not plot_stocks:
            price_history[t_index,0], price_history[t_index,1], it, succes = house_prices_algorithm(sceptics, func, method, grids, par, guess_c, guess_nc, bound_c_l, bound_nc_l, bound_c_l_bis, bound_nc_l_bis, bound_c_r_bis, bound_nc_r_bis, mMarkov, par.iNj,  mDist0_c, mDist0_nc, mDist0_renter, vt_stay_c[t_index,],  vt_stay_nc[t_index,], vt_renter[t_index,], b_stay_c[t_index,],b_stay_nc[t_index,],  b_renter[t_index,], t_index, rental_stock_C0, rental_stock_NC0, coastal_beq0, noncoastal_beq0, savings_beq0, vCoeff_in_C, vCoeff_in_NC, dP_C_lag, dP_NC_lag)
        else:
            price_history[t_index,0]=lom.LoM_C(grids,t_index,vCoeff_in_C)
            price_history[t_index,1]=lom.LoM_NC(grids,t_index,vCoeff_in_NC)
        
        #end=time.perf_counter()                                        
        #print("MC time",end-start)
       # mc_time+=end-start
        
        #excess_demand_C_flow, excess_demand_NC_flow, net_demand_C, net_demand_NC, investment_C, investment_NC, stock_demand_rental, rental_stock = sim.excess_demand_continuous(func, price_history[t,7], int(price_history[t,2]), int(price_history[t,3]),int(price_history[t,4]),int(price_history[t,5]),price_history[t,6], grids, par, mMarkov, dPi_S, dPi_L, iNj, mDist0_c, mDist0_nc, mDist0_renter, price_history[t,0], price_history[t,1], vt_stay_nc, vt_stay_c, vt_renter, b_c_stay, b_renter, b_nc_stay,rental_stock0, coastal_beq0, noncoastal_beq0, savings_beq0,vCoeff_in_C,vCoeff_in_NC)
        print("Time step:",t_index)
        if t_index<nperiods-1:
            #start=time.perf_counter()
            mDist1_c, mDist1_nc, mDist1_renter, stock_demand_rental_C1, stock_demand_rental_NC1, vcoastal_beq[t_index], vnoncoastal_beq[t_index], vsavings_beq[t_index], _, coastal_mass_J, noncoastal_mass_J, renter_mass_J = sim.update_dist_continuous(sceptics, False, 0, func, grids, par, t_index, mMarkov, par.iNj, mDist0_c, mDist0_nc, mDist0_renter, price_history[t_index,0], price_history[t_index,1], vt_stay_c[t_index,], vt_stay_nc[t_index,],  vt_renter[t_index,], b_stay_c[t_index,], b_stay_nc[t_index,], b_renter[t_index,],  coastal_beq0, noncoastal_beq0, savings_beq0,vCoeff_in_C,vCoeff_in_NC, dP_C_lag, dP_NC_lag)
            dP_C_lag=price_history[t_index,0]
            dP_NC_lag=price_history[t_index,1]
            #end=time.perf_counter() 
            #print("dist time",end-start)
            #dist_time+=end-start
             
             
            mDist0_c  = (mDist1_c)
            mDist0_nc = (mDist1_nc)
            mDist0_renter = (mDist1_renter)
            rental_stock_C0= (stock_demand_rental_C1)
            rental_stock_NC0= (stock_demand_rental_NC1)
            coastal_beq0 = (vcoastal_beq[t_index])
            noncoastal_beq0  = (vnoncoastal_beq[t_index])
            savings_beq0 = (vsavings_beq[t_index])
            
            if plot_stocks==True:
                for k_index in range(k_dim):
                    coastal_stock[t_index+1,k_index]=np.sum(mDist0_c[:, k_index, :, :, :, :, :])+coastal_mass_J[k_index]
                    noncoastal_stock[t_index+1,k_index]=np.sum(mDist0_nc[:, k_index, :, :, :, :, :])+noncoastal_mass_J[k_index]
                    rental_stock[t_index+1,k_index]=np.sum(mDist0_renter[1:, k_index, :, :, :])+renter_mass_J[k_index]

    
    return price_history, mDist1_c, mDist1_nc, mDist1_renter, stock_demand_rental_C1, stock_demand_rental_NC1, vcoastal_beq, vnoncoastal_beq, vsavings_beq, vt_stay_c, vt_stay_nc, vt_renter, v_owner_c_wf, v_owner_nc_wf, v_nonowner_wf, coastal_stock, noncoastal_stock, rental_stock
    
        
@njit
def find_coefficients(par, grids, method, sceptics, iNj, mMarkov, vCoeff_C, vCoeff_NC,dP_C_initial, dP_NC_initial,mDist0_c, mDist0_nc, mDist0_renter, rental_stock_C0, rental_stock_NC0, coastal_beq0, noncoastal_beq0, savings_beq0):
    """Find Chebyshev LoM coefficients for the transition-path equilibrium.

    Iterates: (1) solve full VFI over transition path, (2) forward-simulate
    clearing markets at each time step, (3) regress market-clearing prices
    on Chebyshev basis via OLS, (4) update coefficients with dampening (rho=0.5).
    Converges when both coastal and non-coastal coefficient vectors change by
    less than 0.001*rho per element. Typically requires 10-15 iterations.

    Args:
        par: Numba jitclass with model parameters.
        grids: Numba jitclass with model grids.
        method: String, market-clearing algorithm ('secant' or 'bisection').
        sceptics: If True, include both belief types.
        iNj: Number of lifecycle periods.
        mMarkov: 2D array, income Markov transition matrix.
        vCoeff_C: 1D array (5,), initial guess for coastal LoM coefficients.
        vCoeff_NC: 1D array (5,), initial guess for non-coastal LoM coefficients.
        dP_C_initial: Scalar, initial steady-state coastal price (= vCoeff_C_initial[0]).
        dP_NC_initial: Scalar, initial steady-state non-coastal price (= vCoeff_NC_initial[0]).
        mDist0_c: Initial coastal owner distribution from steady state.
        mDist0_nc: Initial non-coastal owner distribution.
        mDist0_renter: Initial renter distribution.
        rental_stock_C0: Initial coastal rental stock.
        rental_stock_NC0: Initial non-coastal rental stock.
        coastal_beq0: Initial coastal bequest aggregate.
        noncoastal_beq0: Initial non-coastal bequest aggregate.
        savings_beq0: Initial savings bequest aggregate.

    Returns:
        dP_C_vec: 1D array, market-clearing coastal prices at each time step.
        dP_NC_vec: 1D array, market-clearing non-coastal prices at each time step.
        vCoeff_C: Converged coastal LoM coefficients.
        vCoeff_NC: Converged non-coastal LoM coefficients.
        iteration: Number of iterations used.
        vt_stay_c: Final coastal stayer policy values.
        vt_stay_nc: Final non-coastal stayer policy values.
        vt_renter: Final renter policy values.
        v_owner_c_wf: Welfare values for coastal owners.
        v_owner_nc_wf: Welfare values for non-coastal owners.
        v_nonowner_wf: Welfare values for renters.
    """

    max_it=15
    iteration =0   

    
    func=False

    for it in range(0, max_it):
        
        iteration += 1
        vCoeff_in_C = vCoeff_C.copy()
        vCoeff_in_NC= vCoeff_NC.copy()        
        # for guess of coefficients, find value functions                
                
        # given value functions, find no flooding stationary distribution given initial alpha
        iT = grids.vTime.size
        
        price_history, _, _, _, _, _, _, _, _, vt_stay_c, vt_stay_nc, vt_renter, v_owner_c_wf, v_owner_nc_wf, v_nonowner_wf,_,_,_ =generate_pricepath(grids, par, func, mMarkov, vCoeff_in_C,vCoeff_in_NC, dP_C_initial, dP_NC_initial, mDist0_c, mDist0_nc, mDist0_renter, rental_stock_C0, rental_stock_NC0, coastal_beq0, noncoastal_beq0, savings_beq0, 0,0,0,method,sceptics)
              
        vCoeff_C, vCoeff_NC, rho, dP_C_vec, dP_NC_vec=coeff_updater(par, grids, price_history, vCoeff_in_C, vCoeff_in_NC, iT)
                
        #create x matrix with k*t rows and alll variables such in coefficient vector. Regress prices on agg states to get coefficients
       

        print('Coefficients C', vCoeff_C)
        print('Coefficients NC', vCoeff_NC)
        dP_C_lom=lom.LoM_C(grids,np.arange(grids.vTime.size),vCoeff_C)
        dP_NC_lom=lom.LoM_NC(grids,np.arange(grids.vTime.size),vCoeff_NC)
        print('price C lom: median SLR', dP_C_lom)
        print('price NC lom: median SLR', dP_NC_lom)
        
        if np.all(np.abs(vCoeff_C - vCoeff_in_C)<0.001*rho) and np.all(np.abs(vCoeff_NC - vCoeff_in_NC)<0.001*rho):
            print("Coefficients converged")
            break
        if iteration>=max_it:
            print("Maximum iterations reached")
            break       

        
    return dP_C_vec, dP_NC_vec, vCoeff_C, vCoeff_NC, iteration, vt_stay_c, vt_stay_nc, vt_renter, v_owner_c_wf, v_owner_nc_wf, v_nonowner_wf#, mc_time, dist_time



@njit
def coeff_updater(par, grids, input_data, vCoeff_in_C, vCoeff_in_NC, iT):
  
    rho=0.5
    dP_C_vec = input_data[:,0] 
    dP_NC_vec = input_data[:,1]
    x_matrix = np.ones((iT, vCoeff_in_C.size), dtype = np.float64)
    time_vector = (2*grids.vTime[:]-(grids.vTime[0]+grids.vTime[-1]))/(grids.vTime[-1]-grids.vTime[0])
    x_matrix[:,1]=time_vector
    x_matrix[:,2]=2*time_vector**2-1
    x_matrix[:,3]=4*time_vector**3-3*time_vector
    x_matrix[:,4]=8*time_vector**4-8*time_vector**2+1
    #update coefficients
    beta_C = misc.ols_numba(x_matrix, dP_C_vec)
    beta_NC = misc.ols_numba(x_matrix, dP_NC_vec)
    vCoeff_C = rho*beta_C+(1-rho)*vCoeff_in_C
    vCoeff_NC= rho*beta_NC+(1-rho)*vCoeff_in_NC

    return vCoeff_C, vCoeff_NC, rho, dP_C_vec, dP_NC_vec

@njit
def initialise_coefficients_ss(par, grids, method, iNj, mMarkov, vCoeff_C_ss, vCoeff_NC_ss, initial = True, sceptics=True):
    """Find steady-state equilibrium prices via iterated VFI and market clearing.

    Iterates: (1) solve VFI at guessed constant price, (2) find stationary
    distribution, (3) compute market-clearing price, (4) update price guess
    with dampening. Adaptive step size reduces rho when oscillation is detected.
    Uses func=True internally, meaning VFI sees LoM-implied prices that equal
    the constant coefficient guess.

    Note: Only the first element (vCoeff[0]) is updated -- higher-order
    Chebyshev terms are zero in steady state since prices are constant.

    Args:
        par: Numba jitclass with model parameters.
        grids: Numba jitclass with model grids.
        method: String, market-clearing algorithm ('secant' or 'bisection').
        iNj: Number of lifecycle periods.
        mMarkov: 2D array, income Markov transition matrix.
        vCoeff_C_ss: 1D array (5,), initial guess for coastal coefficients.
            Modified in-place; elements 1-4 are zeroed.
        vCoeff_NC_ss: 1D array (5,), initial guess for non-coastal coefficients.
            Modified in-place; elements 1-4 are zeroed.
        initial: If True, find initial steady state (t=0 flood probs).
            If False, find terminal steady state (t=T flood probs).
        sceptics: If True, include both belief types.

    Returns:
        dP_C_guess: Final market-clearing coastal price.
        dP_NC_guess: Final market-clearing non-coastal price.
        vCoeff_C_ss: Converged coastal coefficient vector.
        vCoeff_NC_ss: Converged non-coastal coefficient vector.
        mDist1_c: Stationary coastal owner distribution.
        mDist1_nc: Stationary non-coastal owner distribution.
        mDist1_renter: Stationary renter distribution.
        rental_stock_C: Coastal rental stock at steady state.
        rental_stock_NC: Non-coastal rental stock at steady state.
        coastal_beq: Coastal bequest aggregate at steady state.
        noncoastal_beq: Non-coastal bequest aggregate.
        savings_beq: Savings bequest aggregate.
        no_beq: Number of agents with no bequest.
        iteration: Number of iterations used.
    """
    func = True
    max_it=25
    iteration =0
    rho=0.4
    vCoeff_C_ss[1:]=0
    vCoeff_NC_ss[1:]=0
    
    vCoeff_C_record=np.zeros((max_it))        
    vCoeff_NC_record=np.zeros((max_it))
    
    counter=0
    bequest_guess=np.zeros((3))
    
    if initial:
        t_index=0 
    else:
        t_index=grids.vTime.size-1
    
    
    for iteration in range(0, max_it):        
        iteration += 1
        counter += 1 #This is to control step size adjustments
        vCoeff_in_C_ss = vCoeff_C_ss.copy()
        vCoeff_in_NC_ss= vCoeff_NC_ss.copy()
        dP_C_lom=vCoeff_in_C_ss[0]
        dP_NC_lom=vCoeff_in_NC_ss[0]
        print('price C lom', dP_C_lom)
        print('price NC lom', dP_NC_lom)
        vt_stay_c, vt_stay_nc, vt_renter, b_stay_c, b_stay_nc, b_renter = household_problem.solve_ss(grids, par, iNj, mMarkov, vCoeff_in_C_ss[0],vCoeff_in_NC_ss[0], initial, sceptics)
       
        
        guess_c = dP_C_lom
        guess_nc =  dP_NC_lom
        bound_c_l=0.1
        bound_nc_l=0.1                     
                          
        bound_c_l_bis = dP_C_lom- 0.25
        bound_c_r_bis = dP_C_lom + 0.25
        bound_nc_l_bis =  dP_NC_lom- 0.25
        bound_nc_r_bis =  dP_NC_lom + 0.25
    
        mDist1_c, mDist1_nc, mDist1_renter, rental_stock_C, rental_stock_NC, coastal_beq, noncoastal_beq, savings_beq, _, _, _, no_beq, _, _, _ = sim.stat_dist_finder(sceptics, grids, par, mMarkov, iNj, vt_stay_c[0,], vt_stay_nc[0,], vt_renter[0,], b_stay_c[0,], b_stay_nc[0,], b_renter[0,], vCoeff_in_C_ss,vCoeff_in_NC_ss, bequest_guess, initial)
        bequest_guess[0]=coastal_beq
        bequest_guess[1]=noncoastal_beq
        bequest_guess[2]=savings_beq
        
        
        dP_C_lag=dP_C_lom
        dP_NC_lag=dP_NC_lom
        dP_C_guess, dP_NC_guess, _, success = house_prices_algorithm(sceptics, func, method, grids, par, guess_c, guess_nc, bound_c_l, bound_nc_l, bound_c_l_bis, bound_nc_l_bis, bound_c_r_bis, bound_nc_r_bis, mMarkov,  iNj,  mDist1_c, mDist1_nc, mDist1_renter, vt_stay_c[0,],  vt_stay_nc[0,], vt_renter[0,], b_stay_c[0,],b_stay_nc[0,],  b_renter[0,], t_index, rental_stock_C, rental_stock_NC, coastal_beq, noncoastal_beq, savings_beq, vCoeff_in_C_ss,vCoeff_in_NC_ss, dP_C_lag, dP_NC_lag)
                                                
        print('price C mc',dP_C_guess)
        print('price NC mc',dP_NC_guess)
            
        if (iteration>=3 and counter>=3) and ((dP_C_guess-vCoeff_C_record[iteration-1])*(vCoeff_C_record[iteration-1]-vCoeff_C_record[iteration-2])<0 and rho*np.abs(dP_C_guess-vCoeff_C_record[iteration-1])>0.5*np.abs(vCoeff_C_record[iteration-1]-vCoeff_C_record[iteration-2])):
            print("Oscillating, reducing step size if possible, rho=",rho)
            rho=max(rho/2, 0.1)
            counter=0
                
        if (iteration>=3 and counter>=3) and ((dP_NC_guess-vCoeff_NC_record[iteration-1])*(vCoeff_NC_record[iteration-1]-vCoeff_NC_record[iteration-2])<0 and rho*np.abs(dP_NC_guess-vCoeff_NC_record[iteration-1])>0.5*np.abs(vCoeff_NC_record[iteration-1]-vCoeff_NC_record[iteration-2])):
            print("Oscillating, reducing step size if possible, rho=",rho)
            rho=max(rho/2, 0.1)
            counter=0
            
        vCoeff_C_ss[0] = rho*dP_C_guess+(1-rho)*vCoeff_in_C_ss[0]
        vCoeff_NC_ss[0] = rho*dP_NC_guess+(1-rho)*vCoeff_in_NC_ss[0]
             
        print('Coefficients C', vCoeff_C_ss)
        print('Coefficients NC', vCoeff_NC_ss)
        
        vCoeff_C_record[iteration]=vCoeff_C_ss[0]
        vCoeff_NC_record[iteration]=vCoeff_NC_ss[0]  


        if np.all(np.abs(vCoeff_C_ss - vCoeff_in_C_ss)<0.0005*rho) and np.all(np.abs(vCoeff_NC_ss - vCoeff_in_NC_ss)<0.0005*rho):
            print("Successful convergence")
            break
        if iteration>=max_it:
            print("Maximum iterations reached")
            break
        

        
    return dP_C_guess, dP_NC_guess, vCoeff_C_ss, vCoeff_NC_ss, mDist1_c, mDist1_nc, mDist1_renter, rental_stock_C, rental_stock_NC, coastal_beq, noncoastal_beq, savings_beq, no_beq, iteration 


@njit
def precompute_market_data(sceptics, func, grids, par, mMarkov, iNj, mDist1_c, mDist1_nc, mDist1_renter, 
                          vt_stay_c, vt_stay_nc,  vt_renter,b_stay_c, b_stay_nc,  b_renter, t_index, 
                          rental_stock_C, rental_stock_NC, coastal_beq, noncoastal_beq, savings_beq, vCoeff_in_C, vCoeff_in_NC, dP_C_lag, dP_NC_lag):
    
    
    # Store all the market data that doesn't depend on prices
    market_data = {
        'sceptics': sceptics,
        'func': func,       
        't_index': t_index,
        'grids': grids,
        'par': par,
        'mMarkov': mMarkov,  
        'iNj': iNj,
        'mDist1_c': mDist1_c,
        'mDist1_nc': mDist1_nc,
        'mDist1_renter': mDist1_renter,        
        'vt_stay_c': vt_stay_c,
        'vt_stay_nc': vt_stay_nc,
        'vt_renter': vt_renter,   
        'b_stay_c': b_stay_c,        
        'b_stay_nc': b_stay_nc,
        'b_renter': b_renter,
        'rental_stock_C': rental_stock_C,
        'rental_stock_NC': rental_stock_NC,
        'coastal_beq': coastal_beq,
        'noncoastal_beq': noncoastal_beq,
        'savings_beq': savings_beq,
        'vCoeff_in_C': vCoeff_in_C,
        'vCoeff_in_NC': vCoeff_in_NC,
         'dP_C_lag': dP_C_lag, 
         'dP_NC_lag': dP_NC_lag
    }
    
    return market_data

@njit
def compute_excess_demand_pair(dP_C, dP_NC, market_data):
    """
    Compute both excess demands simultaneously to reduce function call overhead.
    
    Expected gain: 10-15% reduction by computing both demands in one call
    """
    # Import the excess_demand_continuous function (assuming it's available)
    # This would need to be imported from simulation.py
    excess_demand_C, excess_demand_NC, _, _, _, _, _, _,_,_ = sim.excess_demand_continuous(market_data['sceptics'], market_data['func'], market_data['grids'], market_data['par'],
        market_data['t_index'],        
        market_data['mMarkov'],  market_data['iNj'],
        market_data['mDist1_c'], market_data['mDist1_nc'], market_data['mDist1_renter'],
        dP_C, dP_NC, market_data['vt_stay_c'], market_data['vt_stay_nc'], market_data['vt_renter'],
        market_data['b_stay_c'], market_data['b_stay_nc'], market_data['b_renter'], 
        market_data['rental_stock_C'], market_data['rental_stock_NC'], market_data['coastal_beq'], 
        market_data['noncoastal_beq'], market_data['savings_beq'],
        market_data['vCoeff_in_C'], market_data['vCoeff_in_NC'], market_data['dP_C_lag'], market_data['dP_NC_lag']
    )
    
    return excess_demand_C, excess_demand_NC

@njit
def bisection_root_finding(compute_func, bounds_low, bounds_high, market_data, 
                         price_other, is_coastal=True, tol=1e-5, max_iter=50):
    a, b = bounds_low, bounds_high
    
    # Cache function evaluations
    if is_coastal:
        fa = compute_func(a, price_other, market_data)[0]  # excess_demand_C
        fb = compute_func(b, price_other, market_data)[0]
    else:
        fa = compute_func(price_other, a, market_data)[1]  # excess_demand_NC
        fb = compute_func(price_other, b, market_data)[1]
    
    # Check if root is bracketed
    if fa * fb > 0:
        # If not bracketed, return the point with smaller absolute function value
        if abs(fa) < abs(fb):
            return a
        else:
            return b
    
    # Simple bisection method for more reliability
    for iteration in range(max_iter):
        # Check convergence
        if abs(b - a) < tol:
            return 0.5 * (a + b)
        
        # Check if we found exact root
        if abs(fa) < tol:
            return a
        if abs(fb) < tol:
            return b
        
        # Bisection step
        c = 0.5 * (a + b)
        
        # Evaluate function at midpoint
        if is_coastal:
            fc = compute_func(c, price_other, market_data)[0]
        else:
            fc = compute_func(price_other, c, market_data)[1]
        
        # Check if we found exact root
        if abs(fc) < tol:
            return c
        
        
        
        # Update interval
        if fa * fc < 0:
            # Root is between a and c
            b = c
            fb = fc
        else:
            # Root is between c and b
            a = c
            fa = fc
            
        
    # Return midpoint if max iterations reached
    return 0.5 * (a + b)


@njit
def secant_method_system_2d(compute_excess_demand_pair, dP_C_0, dP_NC_0,dP_C_1, dP_NC_1,dP_C_2, dP_NC_2,bound_c_l, bound_nc_l,market_data,tol=1e-5,tol_wider=1e-3, max_iter=30):
    # Evaluate initial residuals
    excess_C_0, excess_NC_0 = compute_excess_demand_pair(dP_C_0, dP_NC_0, market_data) # f1_0, f2_0 = f1(x0, y0), f2(x0, y0)
    excess_C_1, excess_NC_1 = compute_excess_demand_pair(dP_C_1, dP_NC_1, market_data) #f1_1, f2_1 = f1(x1, y1), f2(x1, y1)
    excess_C_2, excess_NC_2 = compute_excess_demand_pair(dP_C_2, dP_NC_2, market_data) #f1_2, f2_2 = f1(x2, y2), f2(x2, y2)

    # Quick exit if any starting point is already a solution
    if math.sqrt(excess_C_0*excess_C_0 + excess_NC_0*excess_NC_0) < tol:
        return dP_C_0, dP_NC_0, True, 0, excess_C_0, excess_NC_0
    if math.sqrt(excess_C_1*excess_C_1 + excess_NC_1*excess_NC_1) < tol:
        return dP_C_1, dP_NC_1, True, 0, excess_C_1, excess_NC_1
    if math.sqrt(excess_C_2*excess_C_2 + excess_NC_2*excess_NC_2) < tol:
        return dP_C_2, dP_NC_2, True, 0, excess_C_2, excess_NC_2

    for i in range(max_iter):
        # Differences relative to the “current” point (x2, y2)
        dx0, dy0 = dP_C_2 - dP_C_0, dP_NC_2 - dP_NC_0 # dx0, dy0 = x2 - x0, y2 - y0
        dx1, dy1 = dP_C_2 - dP_C_1, dP_NC_2 - dP_NC_1 # dx1, dy1 = x2 - x1, y2 - y1
        
        
        df1_0 = excess_C_2 - excess_C_0
        df1_1 = excess_C_2 - excess_C_1
        df2_0 = excess_NC_2 - excess_NC_0
        df2_1 = excess_NC_2 - excess_NC_1

        det_A = dx0*dy1 - dx1*dy0
        if abs(det_A) < 1e-15:
            print('singular coordinates')
            return dP_C_2, dP_NC_2, False, i+1, 0, 0

        # Approximate Jacobian entries via Cramer’s rule
        J11 = (df1_0*dy1 - df1_1*dy0) / det_A
        J12 = (dx0*df1_1 - dx1*df1_0) / det_A
        J21 = (df2_0*dy1 - df2_1*dy0) / det_A
        J22 = (dx0*df2_1 - dx1*df2_0) / det_A

        det_J = J11*J22 - J12*J21
        
        if abs(det_J) < 1e-15:
            print('singular jacobian')
            return dP_C_2, dP_NC_2, False, i+1, 0, 0

        # Solve J · delta = –F
        delta_x = (-excess_C_2*J22 + excess_NC_2*J12) / det_J
        delta_y = ( excess_C_2*J21 - excess_NC_2*J11) / det_J

        dP_C_next, dP_NC_next = dP_C_2 + delta_x, dP_NC_2 + delta_y
        if dP_C_next<bound_c_l:
            dP_C_next=bound_c_l       
        if dP_NC_next<bound_nc_l:
            dP_NC_next=bound_nc_l
        excess_C_next, excess_NC_next = compute_excess_demand_pair(dP_C_next, dP_NC_next, market_data) #f1(x_next, y_next), f2(x_next, y_next)

        # Step 3k: Check convergence using both function values and coordinate change
        function_norm = math.sqrt(excess_C_next*excess_C_next + excess_NC_next*excess_NC_next)
        coordinate_change = math.sqrt(delta_x*delta_x + delta_y*delta_y)
        
        #Prevent pointless oscillations close to target
        if i>9:
            if function_norm < tol_wider and coordinate_change<tol_wider:
                print("Early exit")
                return dP_C_next, dP_NC_next, True, i + 1, excess_C_next, excess_NC_next
        
        #Solution found
        if function_norm < tol or coordinate_change<tol:
            return dP_C_next, dP_NC_next, True, i + 1, excess_C_next, excess_NC_next

        # Roll forward the three‐point history
        dP_C_0, dP_NC_0, excess_C_0, excess_NC_0 = dP_C_1, dP_NC_1, excess_C_1, excess_NC_1
        dP_C_1, dP_NC_1, excess_C_1, excess_NC_1 = dP_C_2, dP_NC_2, excess_C_2, excess_NC_2
        dP_C_2, dP_NC_2, excess_C_2, excess_NC_2 = dP_C_next, dP_NC_next, excess_C_next, excess_NC_next

    # If we get here, no convergence within max_iter
    return dP_C_2, dP_NC_2, False, max_iter, 0, 0

@njit
def check_convergence(dP_C, dP_NC, dP_C_prev, dP_NC_prev, excess_C, excess_NC, 
                     price_tol=1e-3, error_tol=1e-4):
    price_dist = max(abs(dP_C - dP_C_prev), abs(dP_NC - dP_NC_prev))
    error = max(abs(excess_C), abs(excess_NC))
    
    price_converged = price_dist <= price_tol
    error_converged = error <= error_tol
    
    return price_converged and error_converged, price_dist, error

@njit
def house_prices_algorithm(sceptics, func, method, grids, par, guess_c, guess_nc, bound_c_l, bound_nc_l, bound_c_l_bis, bound_nc_l_bis, bound_c_r_bis, bound_nc_r_bis, mMarkov, iNj,  mDist1_c, mDist1_nc, mDist1_renter, vt_stay_c,  vt_stay_nc, vt_renter, b_stay_c,b_stay_nc,  b_renter, t_index, rental_stock_C, rental_stock_NC, coastal_beq, noncoastal_beq, savings_beq, vCoeff_in_C, vCoeff_in_NC, dP_C_lag, dP_NC_lag):
     
    # Pre-compute market data that doesn't change during iteration
    market_data = precompute_market_data(sceptics, func, grids, par, mMarkov, iNj, mDist1_c, mDist1_nc, mDist1_renter, 
                                  vt_stay_c, vt_stay_nc,  vt_renter,b_stay_c, b_stay_nc,  b_renter, t_index, 
                                  rental_stock_C, rental_stock_NC, coastal_beq, noncoastal_beq, savings_beq, vCoeff_in_C, vCoeff_in_NC, dP_C_lag, dP_NC_lag)
    
    # Initialize
    dP_C = guess_c
    dP_NC = guess_nc
    PRICE_TOL = 1e-3
    ERROR_TOL = 1e-5
    MAX_ITERATIONS = 15
    SECANT_STEP = 0.005

    if method == 'secant':
        # initial guesses — triangle of perturbations around guess
        dP_C_0 = guess_c - SECANT_STEP/2
        dP_NC_0 = guess_nc - SECANT_STEP/3

        dP_C_1 = guess_c + SECANT_STEP/2
        dP_NC_1 = guess_nc - SECANT_STEP/3

        # Apex (top point)
        dP_C_2 = guess_c
        dP_NC_2 = guess_nc + 2*SECANT_STEP/3

        dP_C, dP_NC, succes, iteration, excess_demand_C, excess_demand_NC = secant_method_system_2d(compute_excess_demand_pair, dP_C_0, dP_NC_0,dP_C_1, dP_NC_1,dP_C_2, dP_NC_2,bound_c_l, bound_nc_l,market_data)
        if succes == False:
            print("Secant method failed")
            for iteration in range(MAX_ITERATIONS):
                dP_C_prev = dP_C
                dP_NC_prev = dP_NC
                
                #Usually secant fails close to target
                bound_c_l_bis=dP_C-1e-2
                bound_c_r_bis=dP_C+1e-2
                bound_nc_l_bis=dP_NC-1e-2
                bound_nc_r_bis=dP_NC+1e-2
                
                # Use adaptive root finding for coastal prices
                dP_C = bisection_root_finding(
                    compute_excess_demand_pair, bound_c_l_bis, bound_c_r_bis, market_data, dP_NC, is_coastal=True)
                
                # Use adaptive root finding for non-coastal prices  
                dP_NC = bisection_root_finding(
                    compute_excess_demand_pair, bound_nc_l_bis, bound_nc_r_bis, market_data, dP_C, is_coastal=False)
                
                # Compute final excess demands for convergence check
                excess_C, excess_NC = compute_excess_demand_pair(dP_C, dP_NC, market_data)
                
                # Check convergence with modular function
                converged, price_dist, error = check_convergence(
                    dP_C, dP_NC, dP_C_prev, dP_NC_prev, excess_C, excess_NC, PRICE_TOL, ERROR_TOL)

                # print('Iteration', iteration, 'P_C=',dP_C, ', P_NC =',dP_NC, 'Error_C =' , excess_C, 'Error_NC =', excess_NC)

                if converged:
                    succes = True
                    break

                # Early exit if making no progress
                if iteration > 2 and price_dist < 5e-4:
                    # print('Early exit due to small price changes at iteration {iteration+1}')
                    break

            if iteration >= MAX_ITERATIONS - 1 and error > ERROR_TOL:
                print("Market clearing failed after MAX_ITERATIONS iterations")
    
    elif method == 'bisection':
        for iteration in range(MAX_ITERATIONS):
            dP_C_prev = dP_C
            dP_NC_prev = dP_NC
            
            # Use adaptive root finding for coastal prices
            dP_C = bisection_root_finding(
                compute_excess_demand_pair, bound_c_l_bis, bound_c_r_bis, market_data, dP_NC, is_coastal=True)
            
            # Use adaptive root finding for non-coastal prices  
            dP_NC = bisection_root_finding(
                compute_excess_demand_pair, bound_nc_l_bis, bound_nc_r_bis, market_data, dP_C, is_coastal=False)
            
            # Compute final excess demands for convergence check
            excess_C, excess_NC = compute_excess_demand_pair(dP_C, dP_NC, market_data)
            
            # Check convergence with modular function
            converged, price_dist, error = check_convergence(
                dP_C, dP_NC, dP_C_prev, dP_NC_prev, excess_C, excess_NC, PRICE_TOL, ERROR_TOL)

            # print('Iteration', iteration, 'P_C=',dP_C, ', P_NC =',dP_NC, 'Error_C =' , excess_C, 'Error_NC =', excess_NC)

            if converged:
                succes = True
                break

            # Update bounds for next iteration (adaptive bounds)
            bound_c_l_bis = max(bound_c_l_bis, dP_C - 0.1)
            bound_c_r_bis = min(bound_c_r_bis, dP_C + 0.1)
            bound_nc_l_bis = max(bound_nc_l_bis, dP_NC - 0.1)
            bound_nc_r_bis = min(bound_nc_r_bis, dP_NC + 0.1)

            # Early exit if making no progress
            if iteration > 2 and price_dist < PRICE_TOL:
                # print('Early exit due to small price changes at iteration {iteration+1}')
                break

        if iteration >= MAX_ITERATIONS - 1 and error > ERROR_TOL:
            print("Market clearing failed after MAX_ITERATIONS iterations")
    
    return dP_C, dP_NC, iteration, succes


