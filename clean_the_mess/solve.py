"""
solve.py

Purpose:
    Solve the model
"""
###########################################################
### Imports
import numpy as np
import time
import misc_functions as misc
import tauchen as tauch
import par as parfile
import grid_creation as grid_creation
import welfare as welfare_stats

###########################################################
### main
def main():
    # import parameters
    vCoeff_C_initial=np.array([0.69906474, 0.,         0.,         0.,         0.        ])
    vCoeff_NC_initial=np.array([0.78259554, 0.,         0.,         0.,         0.        ])
    vCoeff_C_terminal_RE=np.array([0.58952906 , 0.,0.,0.,0. ])
    vCoeff_NC_terminal_RE=np.array([0.85484033,0.,0.,0.,0.])
    vCoeff_C_terminal_HE=np.array([0.64908636, 0.,0.,0.,0. ])
    vCoeff_NC_terminal_HE=np.array([0.82124315,0.,0.,0.,0.])

    method='secant'

    par = misc.construct_jitclass(parfile.par_dict)

    # create grids
    grids, mMarkov=grid_creation.create(par)

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

###########################################################

### start main
if __name__ == "__main__":
    main()
