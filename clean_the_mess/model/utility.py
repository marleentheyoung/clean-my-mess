
"""
utility.py

Purpose:
    Calculate utility function and marginal utility
"""
###########################################################
### Imports
import numpy as np
from numba import njit
#import schumaker as schum

###########################################################
### Functions
###########################################################
@njit
def u(j,c,h,g, par):
    dGamma = par.dGamma
    dPhi = par.dPhi
    dSigma = par.dSigma
    vAgeEquiv = par.vAgeEquiv
    
    u = vAgeEquiv[j]/(1-dSigma)*(((1-dPhi)*c**(1-dGamma)+dPhi*(g*h)**(1-dGamma))**((1-dSigma)/(1-dGamma)))#-1/(1-dSigma)+10
    
    return u

@njit
def u_c(j,c,h,g, par):
    dGamma = par.dGamma
    dPhi = par.dPhi
    dSigma = par.dSigma
    vAgeEquiv = par.vAgeEquiv
    
    u_c = vAgeEquiv[j]*((1-dPhi)*c**(1-dGamma)+dPhi*(g*h)**(1-dGamma))**((dGamma - dSigma)/(1-dGamma))*(1-dPhi)*c**(-dGamma)
    
    return u_c

@njit
def W_bequest(par, b):        
    vBequest = par.dNu*((par.b_bar +b)**(1-par.dSigma))/(1-par.dSigma)
    return vBequest

@njit
def Q_bequest(par,b):
    Q_Bequest=par.dNu*(par.b_bar + b)**(-par.dSigma)
    return Q_Bequest

@njit
def rental_price_calc(par, dP, dP_prime, damage_frac):
    """Compute rental price given current/future house price and flood damage fraction.

    For coastal: pass damage_frac = coastal_damage_frac
    For non-coastal: pass damage_frac = 0.0
    """
    return par.dPsi + max(dP - (1 - par.dDelta - damage_frac) / (1 + par.r) * dP_prime, 0)

@njit
def renter_solve(par,rental_price, g_renter):
         
    h_share=((rental_price/g_renter)**((par.dGamma-1)/par.dGamma)*((1-par.dPhi)/par.dPhi)**(-1/par.dGamma)
    )/((rental_price/g_renter)**((par.dGamma-1)/par.dGamma)*((1-par.dPhi)/par.dPhi)**(-1/par.dGamma)+1)
    c_share=1/((rental_price/g_renter)**((par.dGamma-1)/par.dGamma)*((1-par.dPhi)/par.dPhi)**(-1/par.dGamma)+1)
    w=(((1-par.dPhi)*c_share**(1-par.dGamma)+(par.dPhi)*(h_share/(rental_price/g_renter))**(1-par.dGamma))**
    ((1-par.dSigma)/(1-par.dGamma)))
    return h_share, c_share, w
