"""
config.py

Single source of truth for all model parameters.
Replaces par.py (formerly par_epsilons.py).
"""

import numpy as np

# --- Domain constants (used across multiple files) ---
MODEL_START_YEAR = 1998
EXPERIMENT_YEAR = 2026
NEG_INF = -1e12


def create_par_dict():
    """Create the parameter dictionary for the model.

    Returns a plain dict that is converted to a numba jitclass via
    utils.construct_jitclass().

    Values below are calibrated outputs — do not round or modify
    without re-running calibration. Significant-digit precision
    (e.g., 0.940074219) indicates machine-precision calibrated values.
    """

    time_increment = 2

    # --- Flood probabilities (annual → biennial) ---
    vPi_S_median_annual = np.array([
        0.0194, 0.0198, 0.0202, 0.0206, 0.0210, 0.0214, 0.0218, 0.0222,
        0.0226, 0.0230, 0.0234, 0.0239, 0.0243, 0.0248, 0.0254, 0.0259,
        0.0265, 0.0273, 0.0280, 0.0289, 0.0300, 0.0310, 0.0321, 0.0333,
        0.0347, 0.0361, 0.0376, 0.0392, 0.0410, 0.0427, 0.0444, 0.0461,
        0.0478, 0.0495, 0.0513, 0.0530, 0.0547, 0.0565, 0.0583, 0.0601,
        0.0619, 0.0637, 0.0654, 0.0672, 0.0690, 0.0708, 0.0726, 0.0744,
        0.0762, 0.0780, 0.0798, 0.0816, 0.0816, 0.0816, 0.0816, 0.0816,
    ])
    # Convert from annual to biennial probability
    vPi_S_median = 1 - (1 - vPi_S_median_annual)**time_increment

    # --- Damage states ---
    # Element 0 = no damage (multiplier 1.0)
    # Elements 1-3 = damage conditional on flood occurring
    vZ = np.array([1.0, 0.9, 0.7, 0.3])

    # Conditional probabilities:
    # Element 0 = P(no damage | no flood) = 1
    # Elements 1-3 = P(damage_level | flood). These sum to 1.0.
    # The full array intentionally does NOT sum to 1 — it's conditional probabilities.
    vPDF_z = np.array([1.0, 0.4, 0.4, 0.2])

    par_dict = {
        "time_increment": time_increment,
        "model_start_year": MODEL_START_YEAR,
        "experiment_year": EXPERIMENT_YEAR,

        # --- Demographics ---
        "iNj": 30,
        "j_ret": 23,

        # --- Preferences ---
        "dBeta": 0.940074219**time_increment,
        "dSigma": 2,
        "dGamma": 1/1.25,
        "dPhi": 0.18,
        "dNu": 44.5312500,
        "b_bar": 3.18164063,
        "dLambda": 0.8,
        "alpha_0": 0.4,
        "dOmega": 0.010156250,

        # --- Housing ---
        "dDelta": 1-(1-0.015)**time_increment,
        "dDelta_deprec_rental": 1-(1-0.015)**time_increment,
        "dDelta_default": 0,
        "dPsi": 0.00481015625,
        "h_max": 5.15,
        "dKappa_sell": 0.07,
        "dKappa_buy": 0,
        "dXi_foreclosure": 0.8,
        "dNC_frac": 0.5,
        "dC_frac": 0.5,
        "dTheta": 1.5/2.5,

        # --- Mortgage ---
        "r": 1.03**time_increment-1,
        "r_m": 1.04**time_increment-1,
        "dZeta": 0.01,
        "dZeta_fixed": 1/26,
        "lambda_pti": 0.25,
        "max_ltv": 0.95,

        # --- Income process ---
        "dRho": 0.97,
        "dSigmaeps": 0.20,
        "dSigmaeps_trans": 0.05,
        "iNumStates": 5,  # MUST BE ODD
        "iNumTrans": 3,
        "iM": 1,
        "dL": 0.311,
        "retirement_income_fraction": 0.7,
        # Also hardcoded in full_calibration.py — update there when calibration path is fixed.

        # --- Tax ---
        "tau_0": 0.75,
        "tau_1": 0.151,

        # --- Flood risk ---
        "vPi_S_median": vPi_S_median,
        "damage_states": 3,
        "dZ": 0.8,
        "vZ": vZ,
        "vPDF_z": vPDF_z,

        # --- Grid settings ---
        "iNb": 60,
        "iBmin": 0,
        "iBmax": 27,
        "dXi_min": 1-0.0223437500,
        "dXi_max": 1+0.0223437500,
        "iXin": 7,
        "nonlingrid": 1,
        "nonlingrid_big": 1,

        # --- Belief types ---
        "vTypes": np.array([0.58, 0.42]),

        # --- Initial wealth distribution ---
        "sd_income_initial": 0.3228617,
        "beta0_nowealth": -0.9910767,
        "beta1_nowealth": -0.3417672,
        "var_poswealth": 1.557037,
        "beta0_age": 2.0017337,
        "beta1_age": 0.78795906,
        "beta2_age": -0.02717023,
        "beta3_age": 0.00042535,
        "beta4_age": -2.505727467e-06,
        "corr_poswealth_income": 0.1984,
        "initial_wealth_ratio_cutoff": 5.014401,

        # --- Welfare ---
        "wf_wedge": np.array([0.]),
        "vAgeEquiv": np.ones(50),
    }

    return par_dict


# Backward compatibility: par_dict at module level (used by solve.py and tests)
par_dict = create_par_dict()
