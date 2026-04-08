import numpy as np
import LoM_epsilons as lom
import tauchen as tauch
import par_epsilons as parfile
import misc_functions as misc
import matplotlib.pyplot as plt
import grid_creation as grid_creation
import experiments as experiments
import equilibrium as equil
import household_problem_epsilons_nolearning as household_problem  
import simulation as sim

def plot_pricepaths(par, grids, mMarkov, vCoeff_C_initial, vCoeff_NC_initial, vCoeff_C, vCoeff_NC, vCoeff_C_RE, vCoeff_NC_RE, vCoeff_C_terminal_RE, vCoeff_NC_terminal_RE, vCoeff_C_terminal_HE, vCoeff_NC_terminal_HE):
    # import parameters
    normalisation=vCoeff_NC_initial[0]

    plt.style.use('seaborn-v0_8-whitegrid')
    
    plt.figure(figsize=(9, 5))
    
    # Compute trajectories
    yC_RE = lom.LoM_C(grids, grids.vTime, vCoeff_C_RE)/normalisation
    yNC_RE = lom.LoM_NC(grids, grids.vTime, vCoeff_NC_RE)/normalisation
    
    yC_HE = lom.LoM_C(grids, grids.vTime, vCoeff_C)/normalisation
    yNC_HE = lom.LoM_NC(grids, grids.vTime, vCoeff_NC)/normalisation
    
    # Rescale time: t=0 -> 1998, step=2 years
    years = 1998 + 2 * grids.vTime
    
    # Plot
    lineC, = plt.plot(years, yC_RE, linestyle=':', linewidth=2)
    lineNC, = plt.plot(years, yNC_RE, linestyle=':', linewidth=2)
    
    
    plt.plot(years, yC_HE,
      
         linewidth=2,
         color=lineC.get_color(),
         label='Flood-exposed price trajectory')

    plt.plot(years, yNC_HE,
   
         linewidth=2,
         color=lineNC.get_color(),
         label='Inland price trajectory')
    
    # Initial points
    x0_year = 1998
    y_coastal = vCoeff_C_initial[0]/normalisation
    y_inland = vCoeff_NC_initial[0]/normalisation
    
    plt.scatter([x0_year], [y_coastal], zorder=5)
    plt.scatter([x0_year], [y_inland], zorder=5)

    
    # Annotations above dots
    plt.annotate("Initial flood-exposed price",
                 (x0_year, y_coastal),
                 xytext=(10, 10),
                 textcoords="offset points",
                 ha='center', fontsize=9)
    
    plt.annotate("Initial inland price",
                 (x0_year, y_inland),
                 xytext=(10, 10),
                 textcoords="offset points",
                 ha='center', fontsize=9)
    
    xT_year = years[-1]

    # Terminal prices (given)
    yC_terminal_HE = vCoeff_C_terminal_HE[0]/normalisation
    yNC_terminal_HE = vCoeff_NC_terminal_HE[0]/normalisation
    
    # Scatter terminal points
    plt.scatter([xT_year], [yC_terminal_HE], 
                color=lineC.get_color(), zorder=5)
    
    plt.scatter([xT_year], [yNC_terminal_HE], 
                color=lineNC.get_color(), zorder=5)
    
        
    # Annotations
    plt.annotate("Terminal flood-exposed price",
                 (xT_year, yC_terminal_HE),
                 xytext=(-10, 10),
                 textcoords="offset points",
                 ha='right', fontsize=9)
    
    plt.annotate("Terminal inland price",
                 (xT_year, yNC_terminal_HE),
                 xytext=(-10, 10),
                 textcoords="offset points",
                 ha='right', fontsize=9)
    
    # Dotted guide lines
    plt.vlines(x0_year, y_coastal, yC_HE[0], linestyles='dotted', linewidth=1)
    plt.vlines(x0_year, y_inland, yNC_HE[0], linestyles='dotted', linewidth=1)
    
    # Axis ticks: start at 2000, every 4 years for readability
    start_year = 2000
    end_year = int(years[-1])
    xticks = np.arange(start_year, end_year + 1, 20)
    plt.xticks(xticks)
    
    # Labels & title
    plt.xlabel("Year")
    plt.ylabel("Price")
    plt.title("House price trajectories")
    
    plt.legend(frameon=False)
    plt.grid(True, linestyle='--', alpha=0.4)
    
    plt.tight_layout()
    plt.show()  
    
def plot_distribution_2026(grids, par, func, method, mMarkov, vCoeff_C, vCoeff_NC, vCoeff_C_initial, vCoeff_NC_initial):
    price_history, mDist0_c, mDist0_nc, mDist0_renter, mDist1_c, mDist1_nc, mDist1_renter, stock_demand_rental_C, stock_demand_rental_NC, vcoastal_beq, vnoncoastal_beq, vsavings_beq=experiments.gen_distribution_now(grids, par, func, method, mMarkov, vCoeff_C, vCoeff_NC, vCoeff_C_initial, vCoeff_NC_initial)
        
    plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.size": 12,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "font.family": "serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "lines.linewidth": 2.2,
    "lines.markersize": 5,
    })
    
    # ------------------------------------------------------------
    # Dimensions of mDist0_c:
    # (J, K, G, M, H, L, E)
    # ------------------------------------------------------------
   
    J_dim, K_dim, G_dim, M_dim, H_dim, L_dim, E_dim = mDist0_c.shape
    
       
    # ------------------------------------------------------------
    # 1) Distribution over L conditional on k=0 and k=1
    #    Sum out all dimensions except L, conditional on each K
    # ------------------------------------------------------------
    # For a fixed k, keep L and sum over J,G,M,H,E
    dist_L_k0 = mDist0_c[:, 0, :, :, :, :, :].sum(axis=(0, 1, 2, 3, 5))
    dist_L_k1 = mDist0_c[:, 1, :, :, :, :, :].sum(axis=(0, 1, 2, 3, 5))
    
    # Normalise so mass points sum to 1
    dist_L_k0 = dist_L_k0 / dist_L_k0.sum()
    dist_L_k1 = dist_L_k1 / dist_L_k1.sum()
    
    # Plot
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    

    ax.plot(grids.vL_sim, dist_L_k0)

    
    ax.set_xlabel('Loan-to-value ratio')
    ax.set_ylabel('Conditional mass')
    ax.legend(frameon=False)
    ax.set_ylim(0, 0.5)
    ax.margins(x=0.01)
    ax.ticklabel_format(style='plain', axis='y', scilimits=(0, 0))
    ax.grid(alpha=0.15)
    
    fig.tight_layout()
    plt.savefig("dist_over_L_conditional_on_k.pdf", bbox_inches="tight")
    plt.show()
    
    # ------------------------------------------------------------
    # 2) Distribution over M
    #    Sum out all dimensions except M
    # ------------------------------------------------------------
    # Keep M and sum over J,K,G,H,L,E
    m1_index = misc.binary_search(0, grids.vM_sim.size, grids.vM_sim,10)
    dist_M = mDist0_c.sum(axis=(0, 1, 2, 4, 5, 6))
    dist_M = dist_M[:m1_index+1]
    # Normalise so mass points sum to 1
    dist_M = dist_M / dist_M.sum()
    
    # Plot
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    
    ax.plot(grids.vM_sim[:m1_index+1], dist_M)
    
    ax.set_xlabel('Beginning-of-period savings')
    ax.set_ylabel('Conditional mass')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 0.2)
    ax.margins(x=0.01)
    ax.ticklabel_format(style='plain', axis='y', scilimits=(0, 0))
    ax.grid(alpha=0.15)
    
    fig.tight_layout()
    plt.savefig("dist_over_M.pdf", bbox_inches="tight")
    plt.show()
    
    J_dim, K_dim, G_dim, M_dim, H_dim, L_dim, E_dim = mDist1_c.shape
    
       
    # ------------------------------------------------------------
    # 1) Distribution over L conditional on k=0 and k=1
    #    Sum out all dimensions except L, conditional on each K
    # ------------------------------------------------------------
    # For a fixed k, keep L and sum over J,G,M,H,E
    dist_L_k0 = mDist1_c[:, 0, :, :, :, :, :].sum(axis=(0, 1, 2, 3, 5))
    dist_L_k1 = mDist1_c[:, 1, :, :, :, :, :].sum(axis=(0, 1, 2, 3, 5))
    
    # Normalise so mass points sum to 1
    dist_L_k0 = dist_L_k0 / dist_L_k0.sum()
    dist_L_k1 = dist_L_k1 / dist_L_k1.sum()
    
    # Plot
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    

    ax.plot(grids.vL_sim, dist_L_k0, label='Climate realist')
    ax.plot(grids.vL_sim, dist_L_k1, label='Climate sceptic')
    
    ax.set_xlabel('Loan-to-value ratio')
    ax.set_ylabel('Conditional mass')
    ax.legend(frameon=False)
    ax.set_ylim(0, 0.5)
    ax.margins(x=0.01)
    ax.ticklabel_format(style='plain', axis='y', scilimits=(0, 0))
    ax.grid(alpha=0.15)
    
    fig.tight_layout()
    plt.savefig("dist_over_L_conditional_on_k.pdf", bbox_inches="tight")
    plt.show()
    
    # ------------------------------------------------------------
    # 2) Distribution over M
    #    Sum out all dimensions except M
    # ------------------------------------------------------------
    # Keep M and sum over J,K,G,H,L,E
    m1_index = misc.binary_search(0, grids.vM_sim.size, grids.vM_sim,10)
    dist_M_k0 = mDist1_c[:, 0, :, :, :, :, :].sum(axis=(0, 1, 3, 4, 5))
    dist_M_k1 = mDist1_c[:, 1, :, :, :, :, :].sum(axis=(0, 1, 3, 4, 5))

    dist_M_k0 = dist_M_k0[:m1_index+1]
    dist_M_k1 = dist_M_k1[:m1_index+1]
    # Normalise so mass points sum to 1
    dist_M_k0 = dist_M_k0 / dist_M_k0.sum()
    dist_M_k1 = dist_M_k1 / dist_M_k1.sum()
    
    # Plot
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    
    ax.plot(grids.vM_sim[:m1_index+1], dist_M_k0, label='Climate realist')
    ax.plot(grids.vM_sim[:m1_index+1], dist_M_k1, label='Climate sceptic')
    
    ax.set_xlabel('Beginning-of-period savings')
    ax.set_ylabel('Conditional mass')
    ax.legend(frameon=False)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 0.2)
    ax.margins(x=0.01)
    ax.ticklabel_format(style='plain', axis='y', scilimits=(0, 0))
    ax.grid(alpha=0.15)
    
    fig.tight_layout()
    plt.savefig("dist_over_M.pdf", bbox_inches="tight")
    plt.show()
    
    
def plot_rentalpricepaths(par, grids, mMarkov, vCoeff_C_initial, vCoeff_NC_initial, vCoeff_C, vCoeff_NC, vCoeff_C_RE, vCoeff_NC_RE, vCoeff_C_terminal_RE, vCoeff_NC_terminal_RE, vCoeff_C_terminal_HE, vCoeff_NC_terminal_HE):
    # import parameters


    plt.style.use('seaborn-v0_8-whitegrid')
    
    plt.figure(figsize=(9, 5))
    
    
    coastal_damage_frac=grids.vPi_S_median[0]*np.dot(grids.vPDF_z[1:],(1-grids.vZ[1:]))    
    
    dP_C_lom=vCoeff_C_initial[0]
    dP_NC_lom=vCoeff_NC_initial[0]
    
    dP_C_prime_lom=dP_C_lom
    dP_NC_prime_lom=dP_NC_lom
    rental_price_C_initial=par.dPsi+max(dP_C_lom -(1-par.dDelta-coastal_damage_frac)/(1+par.r)*dP_C_prime_lom,0)
    rental_price_NC_initial=par.dPsi+max(dP_NC_lom -(1-par.dDelta)/(1+par.r)*dP_NC_prime_lom,0)
    normalisation=rental_price_NC_initial
    y_coastal = rental_price_C_initial/normalisation
    y_inland = rental_price_NC_initial/normalisation
    
    yC_RE=np.zeros(grids.vTime.size)
    yNC_RE=np.zeros(grids.vTime.size)
    
    yC_HE=np.zeros(grids.vTime.size)
    yNC_HE=np.zeros(grids.vTime.size)
    #Solve RE rental prices
    for t_index in range(grids.vTime.size):
        dP_C_lom=lom.LoM_C(grids,t_index,vCoeff_C_RE)
        dP_NC_lom=lom.LoM_NC(grids,t_index,vCoeff_NC_RE)
        
        dP_C_prime_lom=lom.LoM_C(grids,min(t_index+1,grids.vTime.size-1),vCoeff_C_RE)
        dP_NC_prime_lom=lom.LoM_NC(grids,min(t_index+1,grids.vTime.size-1),vCoeff_NC_RE)
                
        #THIS IS CUMULATIVE OVER TIME INTERVAL WHEREAS INT RATE IS YEARLY
        coastal_damage_frac=grids.vPi_S_median[t_index]*np.dot(grids.vPDF_z[1:],(1-grids.vZ[1:]))      
        
        yC_RE[t_index]=(par.dPsi+max(dP_C_lom -(1-par.dDelta-coastal_damage_frac)/(1+par.r)*dP_C_prime_lom,0))/normalisation
        yNC_RE[t_index]=(par.dPsi+max(dP_NC_lom -(1-par.dDelta)/(1+par.r)*dP_NC_prime_lom,0))/normalisation
    
    #Solve HE rental prices
    for t_index in range(grids.vTime.size):
        dP_C_lom=lom.LoM_C(grids,t_index,vCoeff_C)
        dP_NC_lom=lom.LoM_NC(grids,t_index,vCoeff_NC)
        
        dP_C_prime_lom=lom.LoM_C(grids,min(t_index+1,grids.vTime.size-1),vCoeff_C)
        dP_NC_prime_lom=lom.LoM_NC(grids,min(t_index+1,grids.vTime.size-1),vCoeff_NC)
                
        #THIS IS CUMULATIVE OVER TIME INTERVAL WHEREAS INT RATE IS YEARLY
        coastal_damage_frac=grids.vPi_S_median[t_index]*np.dot(grids.vPDF_z[1:],(1-grids.vZ[1:]))      
        
        yC_HE[t_index]=(par.dPsi+max(dP_C_lom -(1-par.dDelta-coastal_damage_frac)/(1+par.r)*dP_C_prime_lom,0))/normalisation
        yNC_HE[t_index]=(par.dPsi+max(dP_NC_lom -(1-par.dDelta)/(1+par.r)*dP_NC_prime_lom,0))/normalisation
    

    
    # Rescale time: t=0 -> 1998, step=2 years
    years = 1998 + 2 * grids.vTime
    
    # Plot
    lineC, = plt.plot(years, yC_RE, linestyle=':', linewidth=2)
    lineNC, = plt.plot(years, yNC_RE, linestyle=':', linewidth=2)
    
    
    plt.plot(years, yC_HE,
      
         linewidth=2,
         color=lineC.get_color(),
         label='Flood-exposed price trajectory')

    plt.plot(years, yNC_HE,
   
         linewidth=2,
         color=lineNC.get_color(),
         label='Inland price trajectory')
    
    # Initial points
    x0_year = 1998

    
    plt.scatter([x0_year], [y_coastal], zorder=5)
    plt.scatter([x0_year], [y_inland], zorder=5)

    
    # Annotations above dots
    #plt.annotate("Initial flood-exposed price",
    #             (x0_year, y_coastal),
    #             xytext=(10, 10),
    #             textcoords="offset points",
    #             ha='center', fontsize=9)
    
    #plt.annotate("Initial inland price",
    #             (x0_year, y_inland),
    #             xytext=(10, 10),
    #             textcoords="offset points",
    #             ha='center', fontsize=9)
    
    xT_year = years[-1]

    # Terminal prices (given)
    coastal_damage_frac=grids.vPi_S_median[-1]*np.dot(grids.vPDF_z[1:],(1-grids.vZ[1:]))    
    
    dP_C_lom=vCoeff_C_terminal_HE[0]
    dP_NC_lom=vCoeff_NC_terminal_HE[0]
    
    dP_C_prime_lom=dP_C_lom
    dP_NC_prime_lom=dP_NC_lom
    rental_price_C_terminal=par.dPsi+max(dP_C_lom -(1-par.dDelta-coastal_damage_frac)/(1+par.r)*dP_C_prime_lom,0)
    rental_price_NC_terminal=par.dPsi+max(dP_NC_lom -(1-par.dDelta)/(1+par.r)*dP_NC_prime_lom,0)

    yC_terminal_HE  = rental_price_C_terminal/normalisation
    yNC_terminal_HE = rental_price_NC_terminal/normalisation
    

    
    # Scatter terminal points
    plt.scatter([xT_year], [yC_terminal_HE], 
                color=lineC.get_color(), zorder=5)
    
    plt.scatter([xT_year], [yNC_terminal_HE], 
                color=lineNC.get_color(), zorder=5)
    
        
    # Annotations
    plt.annotate("Terminal flood-exposed price",
                 (xT_year, yC_terminal_HE),
                 xytext=(-10, 10),
                 textcoords="offset points",
                 ha='right', fontsize=9)
    
    plt.annotate("Terminal inland price",
                 (xT_year, yNC_terminal_HE),
                 xytext=(-10, 10),
                 textcoords="offset points",
                 ha='right', fontsize=9)
    
    # Dotted guide lines
    plt.vlines(x0_year, y_coastal, yC_HE[0], linestyles='dotted', linewidth=1)
    plt.vlines(x0_year, y_inland, yNC_HE[0], linestyles='dotted', linewidth=1)
    
    # Axis ticks: start at 2000, every 4 years for readability
    start_year = 2000
    end_year = 2100
    xticks = np.arange(start_year, end_year + 1, 20)
    plt.xticks(xticks)
    plt.xlim(1998-5, end_year)
    # Labels & title
    plt.xlabel("Year")
    plt.ylabel("Price")
    plt.title("Rental price trajectories")
    
    plt.legend(frameon=False)
    plt.grid(True, linestyle='--', alpha=0.4)
    
    plt.tight_layout()
    plt.show()  
    
    
def plot_stock_trajectories(par, grids, mMarkov, vCoeff_C_initial, vCoeff_NC_initial, vCoeff_C, vCoeff_NC, vCoeff_C_RE, vCoeff_NC_RE):
    method='secant'
    func=False
    initial=True
    welfare=False

    experiment = False
    plot_stocks = True
    # run and save SS without welfare: get stationary dist
    sceptics = False
    vt_stay_c, vt_stay_nc, vt_renter, b_stay_c, b_stay_nc, b_renter=household_problem.solve_ss(grids, par, par.iNj, mMarkov,vCoeff_C_initial[0], vCoeff_NC_initial[0], initial, sceptics, welfare)
    bequest_guess=np.zeros((3))
    mDist1_c_SS, mDist1_nc_SS, mDist1_renter_SS, rental_stock_C_out, rental_stock_NC_out, coastal_beq, noncoastal_beq, savings_beq, _, _, _, _, coastal_mass_J, noncoastal_mass_J, renter_mass_J=sim.stat_dist_finder(sceptics, grids, par, mMarkov, par.iNj, vt_stay_c[0,], vt_stay_nc[0,], vt_renter[0,], b_stay_c[0,], b_stay_nc[0,], b_renter[0,], vCoeff_C_initial,vCoeff_NC_initial, bequest_guess, initial)
    
    # get value functions over transition with SLR
    _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, coastal_stock, noncoastal_stock, rental_stock=equil.generate_pricepath(grids, par, func, mMarkov, vCoeff_C_RE,vCoeff_NC_RE, vCoeff_C_initial[0], vCoeff_NC_initial[0], mDist1_c_SS, mDist1_nc_SS, mDist1_renter_SS, rental_stock_C_out, rental_stock_NC_out, coastal_beq, noncoastal_beq, savings_beq, coastal_mass_J, noncoastal_mass_J, renter_mass_J,  method, sceptics, experiment, welfare, plot_stocks)
    
    coastal_stock_RE=np.copy(coastal_stock[:,0])
    noncoastal_stock_RE=np.copy(noncoastal_stock[:,0])
    rental_stock_RE=np.copy(rental_stock[:,0])
    
    # run and save SS without welfare: get stationary dist
    sceptics = True
    vt_stay_c, vt_stay_nc, vt_renter, b_stay_c, b_stay_nc, b_renter=household_problem.solve_ss(grids, par, par.iNj, mMarkov,vCoeff_C_initial[0], vCoeff_NC_initial[0], initial, sceptics, welfare)
    bequest_guess=np.zeros((3))
    mDist1_c_SS, mDist1_nc_SS, mDist1_renter_SS, rental_stock_C_out, rental_stock_NC_out, coastal_beq, noncoastal_beq, savings_beq, _, _, _, _, coastal_mass_J, noncoastal_mass_J, renter_mass_J=sim.stat_dist_finder(sceptics, grids, par, mMarkov, par.iNj, vt_stay_c[0,], vt_stay_nc[0,], vt_renter[0,], b_stay_c[0,], b_stay_nc[0,], b_renter[0,], vCoeff_C_initial,vCoeff_NC_initial, bequest_guess, initial)
    
    # get value functions over transition with SLR
    _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, coastal_stock, noncoastal_stock, rental_stock=equil.generate_pricepath(grids, par, func, mMarkov, vCoeff_C,vCoeff_NC, vCoeff_C_initial[0], vCoeff_NC_initial[0], mDist1_c_SS, mDist1_nc_SS, mDist1_renter_SS, rental_stock_C_out, rental_stock_NC_out, coastal_beq, noncoastal_beq, savings_beq, coastal_mass_J, noncoastal_mass_J, renter_mass_J, method, sceptics, experiment, welfare, plot_stocks)
    
    coastal_stock_HE=np.copy(coastal_stock)/grids.vTypes
    noncoastal_stock_HE=np.copy(noncoastal_stock)/grids.vTypes
    rental_stock_HE=np.copy(rental_stock)/grids.vTypes   
    
    

    renter_coastal_share_RE=np.zeros(grids.vTime.size)  
    renter_coastal_share_HE=np.zeros(grids.vTime.size)  

    #Solve RE rental prices
    for t_index in range(grids.vTime.size):
        dP_C_lom=lom.LoM_C(grids,t_index,vCoeff_C_RE)
        dP_NC_lom=lom.LoM_NC(grids,t_index,vCoeff_NC_RE)
        
        dP_C_prime_lom=lom.LoM_C(grids,min(t_index+1,grids.vTime.size-1),vCoeff_C_RE)
        dP_NC_prime_lom=lom.LoM_NC(grids,min(t_index+1,grids.vTime.size-1),vCoeff_NC_RE)
                
        #THIS IS CUMULATIVE OVER TIME INTERVAL WHEREAS INT RATE IS YEARLY
        coastal_damage_frac=grids.vPi_S_median[t_index]*np.dot(grids.vPDF_z[1:],(1-grids.vZ[1:]))      
        
        yC_RE=(par.dPsi+max(dP_C_lom -(1-par.dDelta-coastal_damage_frac)/(1+par.r)*dP_C_prime_lom,0))
        yNC_RE=(par.dPsi+max(dP_NC_lom -(1-par.dDelta)/(1+par.r)*dP_NC_prime_lom,0))
        
        g_indiff=yC_RE/yNC_RE
        renter_coastal_share_RE[t_index]=min(max((grids.vG[-1]-g_indiff)/(grids.vG[-1]-grids.vG[0]),0),1)
        
    #Solve RE rental prices
    for t_index in range(grids.vTime.size):
        dP_C_lom=lom.LoM_C(grids,t_index,vCoeff_C)
        dP_NC_lom=lom.LoM_NC(grids,t_index,vCoeff_NC)
        
        dP_C_prime_lom=lom.LoM_C(grids,min(t_index+1,grids.vTime.size-1),vCoeff_C)
        dP_NC_prime_lom=lom.LoM_NC(grids,min(t_index+1,grids.vTime.size-1),vCoeff_NC)
                
        #THIS IS CUMULATIVE OVER TIME INTERVAL WHEREAS INT RATE IS YEARLY
        coastal_damage_frac=grids.vPi_S_median[t_index]*np.dot(grids.vPDF_z[1:],(1-grids.vZ[1:]))      
        
        yC_HE=(par.dPsi+max(dP_C_lom -(1-par.dDelta-coastal_damage_frac)/(1+par.r)*dP_C_prime_lom,0))
        yNC_HE=(par.dPsi+max(dP_NC_lom -(1-par.dDelta)/(1+par.r)*dP_NC_prime_lom,0))
        
        g_indiff=yC_HE/yNC_HE
        renter_coastal_share_HE[t_index]=min(max((grids.vG[-1]-g_indiff)/(grids.vG[-1]-grids.vG[0]),0),1)
    
    coastal_rental_stock_RE=rental_stock_RE*renter_coastal_share_RE
    noncoastal_rental_stock_RE=rental_stock_RE-coastal_rental_stock_RE
    
    coastal_rental_stock_HE=np.zeros((grids.vTime.size,grids.vK.size))
    
    for k_index in range(2):
        coastal_rental_stock_HE[:,k_index]=rental_stock_HE[:,k_index]*renter_coastal_share_HE
    noncoastal_rental_stock_HE=rental_stock_HE-coastal_rental_stock_HE
    
    years = 1998 + 2 * grids.vTime

    for it in range(3):
        # ------------------------------------------------------------------
        # Cumulative layers for stacking
        # ------------------------------------------------------------------
        if it==0:
            y1 = noncoastal_stock_RE
            print(it,"noncoastal_stock",y1)
            y2 = y1 + coastal_stock_RE
            print(it,"coastal_stock",y2)
            y3 = y2 + noncoastal_rental_stock_RE
            print(it,"noncoastal_rental_stock",y3)
            y4 = y3 + coastal_rental_stock_RE
            print(it,"coastal_rental_stock",y4)
        if it==1:
            y1 = noncoastal_stock_HE[:,0]
            print(it,"noncoastal_stock",y1)
            y2 = y1 + coastal_stock_HE[:,0]
            print(it,"coastal_stock",y2)
            y3 = y2 + noncoastal_rental_stock_HE[:,0]
            print(it,"noncoastal_rental_stock",y3)
            y4 = y3 + coastal_rental_stock_HE[:,0]
            print(it,"coastal_rental_stock",y4)
        if it==2:
            y1 = noncoastal_stock_HE[:,1]
            print(it,"noncoastal_stock",y1)
            y2 = y1 + coastal_stock_HE[:,1]
            print(it,"coastal_stock",y2)
            y3 = y2 + noncoastal_rental_stock_HE[:,1]
            print(it,"noncoastal_rental_stock",y3)
            y4 = y3 + coastal_rental_stock_HE[:,1]
            print(it,"coastal_rental_stock",y4)
        
        # ------------------------------------------------------------------
        # Plot style (journal-ready)
        # ------------------------------------------------------------------
        plt.rcParams.update({
            "font.family": "serif",
            "font.size": 12,
            "axes.labelsize": 13,
            "axes.titlesize": 14,
            "legend.fontsize": 11,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "axes.linewidth": 0.8,
        })
        
        fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=300)
        
        # Stacked areas
        ax.fill_between(years, 0, y1,
                        label="Inland owner-occupied",
                        edgecolor="black", linewidth=0.5)
        
        ax.fill_between(years, y1, y2,
                        label="Coastal owner-occupied",
                        edgecolor="black", linewidth=0.5)
        
        ax.fill_between(years, y2, y3,
                        label="Inland rental",
                        edgecolor="black", linewidth=0.5)
        
        ax.fill_between(years, y3, y4,
                        label="Coastal rental",
                        edgecolor="black", linewidth=0.5)
        
        # Axes
        ax.set_xlim(years[0], years[51])
        ax.set_ylim(0, 1)
        ax.set_xlabel("Year")
        ax.set_ylabel("Mass of agents")
        
        # Ticks: show every ~10 years for readability
        ax.set_xticks(np.arange(2000, 2101, 20))
        
        # Clean look
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_yticks(np.linspace(0, 1, 6))
        
        # Light grid
        ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5)
        ax.set_axisbelow(True)
        
        # Legend (top, compact)
        ax.legend(loc="upper center",
                  bbox_to_anchor=(0.5, 1.18),
                  ncol=2,
                  frameon=False)
        
        plt.tight_layout()
        
        # Save
        plt.savefig("housing_sorting_stacked_area.pdf", bbox_inches="tight")
        plt.savefig("housing_sorting_stacked_area.png", dpi=300, bbox_inches="tight")
        
        plt.show()
    