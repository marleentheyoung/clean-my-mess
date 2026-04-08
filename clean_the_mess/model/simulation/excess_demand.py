"""
simulation.py

Purpose:
    Simulate the economy for a given law of motion
"""
import numpy as np
from numba import njit
import model.interp as interp
import model.lom as lom
import model.utils as misc
import model.simulation.buyer_sim as buy_sim
import numba as nb
import model.utility as ut
import mortgage_choice_simulation as mortgage_sim
import model.simulation.mortgage_sim_exc as mortgage_sim_exc
import model.simulation.initial_joint as initial_joint_sim
import model.tauchen as tauch

NEG_INF = -1e12


   
@njit
def excess_demand_continuous(sceptics, initialise, grids, par, t_index, mMarkov, iNj, mDist0_c, mDist0_nc, mDist0_renter, dP_C, dP_NC,  vt_stay_c, vt_stay_nc, vt_renter, b_stay_c,  b_stay_nc, b_renter, rental_stock_C, rental_stock_NC, coastal_beq, noncoastal_beq, savings_beq,vCoeff_in_C,vCoeff_in_NC, dP_C_lag, dP_NC_lag):
    
    #print("Price in:", dP_C, dP_NC)
    demand_C=0
    demand_NC=0
    stayer_demand_C=0
    stayer_demand_NC=0
    supply_NC = 0
    supply_C = 0
    stock_demand_rental_C = 0.0
    stock_demand_rental_NC = 0.0
    depreciation_C = 0
    depreciation_NC = 0
    default_stock_C=0
    default_stock_NC=0
    dPi_S=grids.vPi_S_median[t_index]   
    
    if sceptics==False:
        k_dim=1 
    else:
        k_dim=2
          
    dP_C_growth_lom=lom.LoM_C(grids,t_index,vCoeff_in_C)-lom.LoM_C(grids,max(t_index-1,0), vCoeff_in_C)
    dP_NC_growth_lom=lom.LoM_NC(grids,t_index, vCoeff_in_NC)-lom.LoM_NC(grids,max(t_index-1,0), vCoeff_in_NC)
    
    dP_C_growth_prime_lom=lom.LoM_C(grids,min(t_index+1,grids.vTime.size-1),vCoeff_in_C)-lom.LoM_C(grids,t_index,vCoeff_in_C)
    dP_NC_growth_prime_lom=lom.LoM_NC(grids,min(t_index+1,grids.vTime.size-1),vCoeff_in_NC)-lom.LoM_NC(grids,t_index,vCoeff_in_NC)
    
    dP_C_lom=dP_C_lag+dP_C_growth_lom
    dP_NC_lom=dP_NC_lag+dP_NC_growth_lom
    
    dP_C_prime=dP_C+dP_C_growth_prime_lom
    dP_NC_prime=dP_NC+dP_NC_growth_prime_lom
    
    dP_C_prime_lom=dP_C_lom+dP_C_growth_prime_lom
    dP_NC_prime_lom=dP_NC_lom+dP_NC_growth_prime_lom
    
    
    #THIS IS CUMULATIVE OVER TIME INTERVAL WHEREAS INT RATE IS YEARLY
    coastal_damage_frac=grids.vPi_S_median[t_index]*np.dot(grids.vPDF_z[1:],(1-grids.vZ[1:]))
    
    rental_price_C=ut.rental_price_calc(par, dP_C, dP_C_prime, coastal_damage_frac)
    rental_price_NC=ut.rental_price_calc(par, dP_NC, dP_NC_prime, 0.0)

    rental_price_lom_C=ut.rental_price_calc(par, dP_C_lom, dP_C_prime_lom, coastal_damage_frac)
    rental_price_lom_NC=ut.rental_price_calc(par, dP_NC_lom, dP_NC_prime_lom, 0.0)
    
    minpay_matrix_C, ltv_minpay_index_left_C, minpay_matrix_NC, ltv_minpay_index_left_NC, max_ltv_C,max_ltv_NC, max_ltv_index_C, max_ltv_index_NC=mortgage_matrix_solve(par, grids, dP_C_lag, dP_NC_lag, dP_C, dP_NC)
    
    housing_bequest=coastal_beq*(1-coastal_damage_frac-par.dDelta)*dP_C + noncoastal_beq*(1-par.dDelta)*dP_NC
    total_bequest = (housing_bequest+savings_beq*(1+par.r))*par.iNj
    mPi_joint=initial_joint_sim.initial_joint(par, grids, total_bequest)
    for k_index in range(k_dim):
        if sceptics==True:
            k_weight=grids.vTypes[k_index]
        else:
            k_weight=1.
        for g_index in range(grids.vG.size):
            mDist0_renter[0,k_index,g_index,:,:]= (1/iNj)*k_weight*(1/grids.vG.size)*mPi_joint
    
    mDist0_c[0,:,:,:,:,:,:]=0
    mDist0_nc[0,:,:,:,:,:,:]=0

    mass=np.empty((grids.vM_sim.size))
    for j in range(iNj):     
        for e_index in range(grids.vE.size):
            # coastal homeowners        
            for k_index in range(k_dim):     
                for g_index in range(grids.vG.size):
                    h_share_lom, w_lom, h_share, w, rental_price_lom, rental_price, coastal_rent_share, g_renter_lom, g_renter=renter_solve(par, grids, g_index, rental_price_lom_C, rental_price_lom_NC, rental_price_C, rental_price_NC)
                    vt_stay_c_input, vt_stay_nc_input = vt_stay_c[j, k_index, g_index, :, :, :, e_index],vt_stay_nc[j, k_index, g_index,  :, :, :, e_index]
                    vt_renter_input, b_renter_input = vt_renter[j, k_index, g_index, :, e_index], b_renter[j, k_index, g_index,  :, e_index]
                    for e_trans_index in range(grids.vE_trans.size):
                        if j<par.j_ret:                        
                           income_mass=grids.mMarkov_trans[e_trans_index]
                        else:
                            income_mass=1
                            if e_trans_index>0:
                                continue                     
                        for h_index in range(grids.vH.size):
                            h = grids.vH[h_index]                        
                            for l_index_sim in range(grids.vL_sim.size):
                                ltv=grids.vL_sim[l_index_sim]
                                #Prices to calculate beginning of period mortgage
                                
                                mortgage_start=ltv*h*dP_C_lag                                 
                                e, mortgage_rebate=misc.net_income(par, grids, j, e_index, e_trans_index, mortgage_start) 

                                for damage_index in range(grids.vZ.size):
                                    dZ = grids.vZ[damage_index]                                    
                                    if damage_index==0:
                                        prob_dZ=(1-dPi_S)
                                    else:                                    
                                        prob_dZ = dPi_S*grids.vPDF_z[damage_index] 
                                    for m_index_sim in range(grids.vM_sim.size):                                              
                                        mass[m_index_sim] = income_mass*prob_dZ*mDist0_c[j,k_index,g_index,m_index_sim,h_index,l_index_sim,e_index]
                                    mass_pos_idx=np.where(mass>0)[0]                               
                                    #For stayers, interpolate value function at cash in hand 
                                    vt_stay_c_sim= mortgage_sim_exc.solve(par,grids,vt_stay_c_input[:,h_index,:],j,h, e, dZ, dP_C,mortgage_start,max_ltv_index_C[j,h_index,e_index], minpay_matrix_C[j, h_index, l_index_sim], ltv_minpay_index_left_C[j, h_index, l_index_sim],mass_pos_idx) 
                                    x_sell_vec=grids.vM_sim+e+(1-(1-dZ)-par.dDelta-par.dKappa_sell)*h*dP_C-(1+par.r_m)*mortgage_start
                                    vt_buy_c, h_pol_C, _, _,_ =buy_sim.solve(par, grids,-1,x_sell_vec, j, dP_C, vt_stay_c_input,dP_C_lom,max_ltv_C[j,:,e_index],max_ltv_index_C[j,:,e_index],mass_pos_idx)
                                    vt_buy_nc, h_pol_NC, _, _,_ =buy_sim.solve(par, grids,-1,x_sell_vec, j, dP_NC, vt_stay_nc_input,dP_NC_lom,max_ltv_NC[j,:,e_index],max_ltv_index_NC[j,:,e_index],mass_pos_idx)
                                    
                                    vt_renter_sim, h_renter = renter_sim_demand(False, initialise, par,grids,j,vt_renter_input, b_renter_input, h_share_lom,w_lom,h_share,w,rental_price,rental_price_lom, g_renter_lom, g_renter, x_sell_vec, mass_pos_idx)
                                    if (1+par.r_m)*mortgage_start>(1-par.dDelta-par.dKappa_sell-(1-dZ))*h*dP_C:
                                        vt_default, h_default = renter_sim_demand(True, initialise, par,grids,j,vt_renter_input, b_renter_input, h_share_lom,w_lom,h_share,w,rental_price,rental_price_lom, g_renter_lom, g_renter, grids.vM_sim+e-mortgage_rebate, mass_pos_idx)      
                                    else:
                                        vt_default=np.ones(grids.vM_sim.size)*NEG_INF
                                        h_default=np.zeros(grids.vM_sim.size)                                     
               
                                         
                                    mass_stay,mass_rent,mass_buyc,mass_buync,mass_default = continuous_decide(grids,vt_stay_c_sim, vt_buy_c, vt_buy_nc, vt_renter_sim,vt_default,mass)
                                                      
                                    for m_index_sim in mass_pos_idx:
                                        depreciation_C += par.dDelta*mass[m_index_sim]*h
                                        stayer_demand_C += mass_stay[m_index_sim]*h
                                        if mass_rent[m_index_sim]*h_renter[m_index_sim]*coastal_rent_share<0 or mass_default[m_index_sim]*h_default[m_index_sim]*coastal_rent_share<0:
                                            assert mass_rent[m_index_sim]*h_renter[m_index_sim]*coastal_rent_share>0 and mass_default[m_index_sim]*h_default[m_index_sim]*coastal_rent_share>0
                                        
                                        stock_demand_rental_C += mass_rent[m_index_sim]*h_renter[m_index_sim]*coastal_rent_share + mass_default[m_index_sim]*h_default[m_index_sim]*coastal_rent_share
                                        stock_demand_rental_NC += mass_rent[m_index_sim]*h_renter[m_index_sim]*(1-coastal_rent_share) + mass_default[m_index_sim]*h_default[m_index_sim]*(1-coastal_rent_share)
                                        default_stock_C+=mass_default[m_index_sim]*h               
                                        demand_C += mass_buyc[m_index_sim]*h_pol_C[m_index_sim]                                  
                                        demand_NC += mass_buync[m_index_sim]*h_pol_NC[m_index_sim]
                                        supply_C += (mass[m_index_sim]-mass_stay[m_index_sim])*h-par.dDelta_default*mass_default[m_index_sim]*h  
                                 
                        # non coastal homeowners                             
                        for h_index in range(grids.vH.size):
                            h = grids.vH[h_index]                        
                            for l_index_sim in range(grids.vL_sim.size):
                                # ltv=grids.vL_sim[l_index_sim]
                                ltv=grids.vL_sim[l_index_sim]
                             
                                mortgage_start=ltv*h*dP_NC_lag                                    
                                e, mortgage_rebate=misc.net_income(par, grids, j, e_index, e_trans_index, mortgage_start) 

                                for m_index_sim in range(grids.vM_sim.size):                                            
                                    mass[m_index_sim] = income_mass*mDist0_nc[j,k_index,g_index,m_index_sim,h_index,l_index_sim,e_index]
                                mass_pos_idx=np.where(mass>0)[0]
                                vt_stay_nc_sim= mortgage_sim_exc.solve(par,grids,vt_stay_nc_input[:,h_index,:],j,h, e,1, dP_NC,mortgage_start,max_ltv_index_NC[j,h_index,e_index], minpay_matrix_NC[j, h_index, l_index_sim], ltv_minpay_index_left_NC[j, h_index, l_index_sim],mass_pos_idx)
                                x_sell_vec = grids.vM_sim+e+(1-par.dDelta-par.dKappa_sell)*h*dP_NC-(1+par.r_m)*mortgage_start
                                vt_buy_c, h_pol_C, _, _,_ =buy_sim.solve(par, grids,-1,x_sell_vec, j, dP_C, vt_stay_c_input,dP_C_lom,max_ltv_C[j,:,e_index],max_ltv_index_C[j,:,e_index],mass_pos_idx)
                                vt_buy_nc, h_pol_NC, _, _,_ =buy_sim.solve(par, grids,-1,x_sell_vec, j, dP_NC, vt_stay_nc_input,dP_NC_lom,max_ltv_NC[j,:,e_index],max_ltv_index_NC[j,:,e_index],mass_pos_idx)
                                             
                                vt_renter_sim, h_renter = renter_sim_demand(False, initialise, par,grids,j,vt_renter_input, b_renter_input, h_share_lom,w_lom,h_share,w,rental_price,rental_price_lom, g_renter_lom, g_renter,x_sell_vec, mass_pos_idx)
                                if (1+par.r_m)*mortgage_start>(1-par.dDelta-par.dKappa_sell)*h*dP_NC:
                                    vt_default, h_default = renter_sim_demand(True, initialise, par,grids,j,vt_renter_input, b_renter_input, h_share_lom,w_lom,h_share,w,rental_price,rental_price_lom, g_renter_lom, g_renter, grids.vM_sim+e-mortgage_rebate, mass_pos_idx)      
                                else:
                                    vt_default=np.ones(grids.vM_sim.size)*NEG_INF
                                    h_default=np.zeros(grids.vM_sim.size)
                                mass_stay,mass_rent,mass_buyc,mass_buync,mass_default = continuous_decide(grids,vt_stay_nc_sim, vt_buy_c, vt_buy_nc, vt_renter_sim,vt_default,mass)
                                                          
                                for m_index_sim in mass_pos_idx:
                                    depreciation_NC += par.dDelta*mass[m_index_sim]*h
                                    stayer_demand_NC += mass_stay[m_index_sim]*h
                                    if mass_rent[m_index_sim]*h_renter[m_index_sim]*coastal_rent_share<0 or mass_default[m_index_sim]*h_default[m_index_sim]*coastal_rent_share<0:
                                        assert mass_rent[m_index_sim]*h_renter[m_index_sim]*coastal_rent_share>0 and mass_default[m_index_sim]*h_default[m_index_sim]*coastal_rent_share>0

                                    
                                    stock_demand_rental_C += mass_rent[m_index_sim]*h_renter[m_index_sim]*coastal_rent_share + mass_default[m_index_sim]*h_default[m_index_sim]*coastal_rent_share
                                    stock_demand_rental_NC += mass_rent[m_index_sim]*h_renter[m_index_sim]*(1-coastal_rent_share) + mass_default[m_index_sim]*h_default[m_index_sim]*(1-coastal_rent_share)
                                    default_stock_NC+=mass_default[m_index_sim]*h                               
                                    demand_C += mass_buyc[m_index_sim]*h_pol_C[m_index_sim]   
                                    demand_NC += mass_buync[m_index_sim]*h_pol_NC[m_index_sim]
                                    supply_NC += (mass[m_index_sim]-mass_stay[m_index_sim])*h-par.dDelta_default*mass_default[m_index_sim]*h

    
                # renters      
                        for x_index_sim in range(grids.vX_sim.size):                             
                            mass[x_index_sim] = income_mass*mDist0_renter[j,k_index,g_index,x_index_sim,e_index]
                        mass_pos_idx=np.where(mass>0)[0]                    
                        e, mortgage_rebate=misc.net_income(par, grids, j, e_index, e_trans_index, 0) 

                        vt_buy_c, h_pol_C, _, _,_ =buy_sim.solve(par, grids,-1,grids.vX_sim+e, j, dP_C, vt_stay_c_input,dP_C_lom,max_ltv_C[j,:,e_index],max_ltv_index_C[j,:,e_index],mass_pos_idx)
                        vt_buy_nc, h_pol_NC, _, _,_ =buy_sim.solve(par, grids,-1,grids.vX_sim+e, j, dP_NC, vt_stay_nc_input,dP_NC_lom,max_ltv_NC[j,:,e_index],max_ltv_index_NC[j,:,e_index],mass_pos_idx)
                        vt_renter_sim, h_renter = renter_sim_demand(False, initialise, par,grids,j,vt_renter_input, b_renter_input, h_share_lom,w_lom,h_share,w,rental_price,rental_price_lom, g_renter_lom, g_renter, grids.vX_sim+e, mass_pos_idx)
                                
                        mass_rent,mass_buyc,mass_buync = continuous_decide_renter(grids,vt_buy_c, vt_buy_nc, vt_renter_sim, mass)
              
                                            
                        for x_index_sim in mass_pos_idx:
                            if mass_rent[x_index_sim]*h_renter[x_index_sim]*coastal_rent_share<0:
                                assert mass_rent[x_index_sim]*h_renter[x_index_sim]*coastal_rent_share>0
                            stock_demand_rental_C += mass_rent[x_index_sim]*h_renter[x_index_sim]*coastal_rent_share 
                            stock_demand_rental_NC += mass_rent[x_index_sim]*h_renter[x_index_sim]*(1-coastal_rent_share) 
                            demand_C += mass_buyc[x_index_sim]*h_pol_C[x_index_sim]
                            demand_NC += mass_buync[x_index_sim]*h_pol_NC[x_index_sim]

                            
    net_demand_C=demand_C-supply_C
    net_demand_NC=demand_NC-supply_NC
    
    investment_C = (par.dTheta*dP_C)**(par.dTheta/(1-par.dTheta))*par.dC_frac*par.dL
    investment_NC = (par.dTheta*dP_NC)**(par.dTheta/(1-par.dTheta))*par.dNC_frac*par.dL
 
      
    stock_demand_NC=(stayer_demand_NC+demand_NC)*par.dDelta+stock_demand_rental_NC*par.dDelta_deprec_rental+default_stock_NC*par.dDelta_default 
    stock_demand_C=(stayer_demand_C+demand_C)*par.dDelta+stock_demand_rental_C*par.dDelta_deprec_rental+default_stock_C*par.dDelta_default 
    excess_demand_C_stock = stock_demand_C - investment_C
    excess_demand_NC_stock = stock_demand_NC - investment_NC     
    
    excess_demand_C_flow=net_demand_C+depreciation_C-investment_C-coastal_beq*(1-par.dDelta)+stock_demand_rental_C-rental_stock_C*(1-par.dDelta_deprec_rental)
    excess_demand_NC_flow=net_demand_NC+depreciation_NC-investment_NC-noncoastal_beq*(1-par.dDelta)+stock_demand_rental_NC-rental_stock_NC*(1-par.dDelta_deprec_rental)
    
    

   
    # return excess_demand_C, excess_demand_NC
    if initialise:
        print("prices:",dP_C, dP_NC) 
        print("Rental prices:", rental_price_C, rental_price_NC)
        print("Excess demands:",excess_demand_C_stock, excess_demand_NC_stock)
        #print("Flow excess demands:",excess_demand_C_flow, excess_demand_NC_flow)
        #print("Net demands:", demand_C-supply_C, demand_NC-supply_NC)
        #print("Owner demands:", stayer_demand_C+demand_C, stayer_demand_NC+demand_NC)
        #print("Depreciation difference:", (stayer_demand_C+demand_C)*par.dDelta-depreciation_C, (stayer_demand_NC+demand_NC)*par.dDelta-depreciation_NC)
        print("Rental demands:",stock_demand_rental_C, stock_demand_rental_NC)
        return excess_demand_C_stock, excess_demand_NC_stock, net_demand_C, net_demand_NC, investment_C, investment_NC, stock_demand_rental_C, rental_stock_C, stock_demand_rental_NC, rental_stock_NC
    else:
        print("prices:",dP_C, dP_NC)   
        print("Rental prices:", rental_price_C, rental_price_NC)
        print("Excess demands:",excess_demand_C_flow, excess_demand_NC_flow)
        #print("Net demands:", demand_C-supply_C, demand_NC-supply_NC)
        #print("Owner demands:", stayer_demand_C+demand_C, stayer_demand_NC+demand_NC)
        #print("Depreciation difference:", (stayer_demand_C+demand_C)*par.dDelta-depreciation_C, (stayer_demand_NC+demand_NC)*par.dDelta-depreciation_NC)
        print("Rental demands:",stock_demand_rental_C, stock_demand_rental_NC)
        return excess_demand_C_flow, excess_demand_NC_flow, net_demand_C, net_demand_NC, investment_C, investment_NC, stock_demand_rental_C, rental_stock_C, stock_demand_rental_NC, rental_stock_NC


