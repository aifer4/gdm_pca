import numpy as np
from scipy.special import spherical_jn
import numba
import pickle

import modes
from util import trapz, backtrapz, deriv
"""Calculate the approximate CMB power spectrum (TT only)"""


def get_bessels(L, X):
    """Precompute Bessel Functions."""
    # try to load bessel functions from file:
    try:
        L_stored = pickle.load(open("output/bessel_L.p", "rb"))
        X_stored = pickle.load(open("output/bessel_X.p", "rb"))
        JLX = pickle.load(open("output/bessel_JLX.p", "rb"))
        DJLX = pickle.load(open("output/bessel_DJLX.p", "rb"))
        assert(np.array_equal(L, L_stored))
        assert(np.array_equal(X, X_stored))
        print('Bessel functions loaded from file.')
    except:
        print("Bessel functions not found. Computing bessel functions...\n")
        JLX = np.array([
            spherical_jn(ell, X)
            for ell in L])
        DJLX = np.array([
            spherical_jn(ell, X, derivative=True)
            for ell in L])
        pickle.dump(L, open("output/bessel_L.p", "wb"))
        pickle.dump(X, open("output/bessel_X.p", "wb"))
        pickle.dump(JLX, open("output/bessel_JLX.p", "wb"))
        pickle.dump(DJLX, open("output/bessel_DJLX.p", "wb"))
        print('Bessel functions stored.')
    return JLX, DJLX


@numba.njit(parallel=True, cache=True)
def get_Cl_3fld(L, L_eval, X, JLX, DJLX, A, K, wC, cs2C,
                wN, cs2N, OmegaB0, OmegaC0, OmegaG0, OmegaN0,
                As, TCMB0, h):
    NL = len(L)

    # Solve the ODE and get the source terms.
    Y, TAU = modes.solve_3fld(A, K, wC, cs2C, wN, cs2N, OmegaB0, OmegaC0,
                              OmegaG0, OmegaN0)
    tau_rec = TAU[-1]

    OmegaM0 = OmegaB0 + OmegaC0
    a_rec = 1/1100
    tau_r = .5*(OmegaM0/a_rec)**.5
    a_eq = (OmegaG0 + OmegaN0)/(OmegaB0 + OmegaC0)
    y_rec = a_rec/a_eq
    x_rec = tau_rec/tau_r
    Delta_phi = (2-8/y_rec + 16*x_rec*y_rec**-3)/(10*y_rec)

    DOP = Y[-1, 2, :]
    SW = Y[-1, 0, :] + Y[-1, 1, :]/4 + 2*Delta_phi

    # Get the current conformal time elapsed since recombination.
    A_late = np.linspace(a_rec, 1, 1000)
    H_late = A_late * np.sqrt((OmegaB0+OmegaC0)*A_late**-3
                              + (OmegaG0+OmegaN0)*A_late**-4)
    delta_tau = trapz(A_late, 1/(A_late * H_late))[-1]

    # Interpolate the source terms.
    K_INT = X/delta_tau
    DOP_terp = np.interp(K_INT, K, DOP)
    SW_terp = np.interp(K_INT, K, SW)

    # Calculate the damping factor
    tau_s = 0.6 * OmegaM0**.25 * OmegaB0**-.5 * a_rec**.75 * h**-.5 / tau_r
    T = np.exp(-2*(K_INT*tau_s)**2 - (.03*K_INT*tau_rec)**2)

    # Integrate
    Dl = SW_terp*JLX + DOP_terp*DJLX
    Cl_itgd = Dl**2 * T / K_INT
    Cl = np.array([trapz(K_INT, Cl_itgd[i, :])[-1] for i in range(NL)])
    Cl_norm = L*(L+1)*2*As*TCMB0**2*Cl
    Cl_terp = np.interp(L_eval, L, Cl_norm)
    return Cl_terp


@numba.njit(parallel=True, cache=True)
def get_Cl_2fld(L, L_eval, X, JLX, DJLX, A, K, wD, cs2D,
                deltaD0, vD0, OmegaB0, OmegaC0, OmegaG0, OmegaN0,
                As, TCMB0, h):
    NL = len(L)
    OmegaD0 = OmegaC0 + OmegaG0
    # Solve the ODE and get the source terms.
    Y, TAU = modes.solve_2fld(A, K, wD, cs2D, deltaD0, vD0, OmegaB0,
                              OmegaC0, OmegaG0, OmegaN0, OmegaD0)
    tau_rec = TAU[-1]

    OmegaM0 = OmegaB0 + OmegaC0
    a_rec = 1/1100
    tau_r = .5*(OmegaM0/a_rec)**.5
    a_eq = (OmegaG0 + OmegaN0)/(OmegaB0 + OmegaC0)
    y_rec = a_rec/a_eq
    x_rec = tau_rec/tau_r
    Delta_phi = (2-8/y_rec + 16*x_rec*y_rec**-3)/(10*y_rec)

    DOP = Y[-1, 2, :]
    SW = Y[-1, 0, :] + Y[-1, 1, :]/4 + 2*Delta_phi

    # Get the current conformal time elapsed since recombination.
    A_late = np.linspace(a_rec, 1, 1000)
    H_late = A_late * np.sqrt((OmegaB0+OmegaC0)*A_late**-3
                              + (OmegaG0+OmegaN0)*A_late**-4)
    delta_tau = trapz(A_late, 1/(A_late * H_late))[-1]

    # Interpolate the source terms.
    K_INT = X/delta_tau
    DOP_terp = np.interp(K_INT, K, DOP)
    SW_terp = np.interp(K_INT, K, SW)

    # Calculate the damping factor
    tau_s = 0.6 * OmegaM0**.25 * OmegaB0**-.5 * a_rec**.75 * h**-.5 / tau_r
    T = np.exp(-2*(K_INT*tau_s)**2 - (.03*K_INT*tau_rec)**2)

    # Integrate
    Dl = SW_terp*JLX + DOP_terp*DJLX
    Cl_itgd = Dl**2 * T / K_INT
    Cl = np.array([trapz(K_INT, Cl_itgd[i, :])[-1] for i in range(NL)])
    Cl_norm = L*(L+1)*2*As*TCMB0**2*Cl
    Cl_terp = np.interp(L_eval, L, Cl_norm)
    return Cl_terp


def get_Cl_err(L, Cl):
    f_sky = 1.0  # fraction of sky
    l_s = 480.  # filtering scale
    theta_pix = 0.0012  # rad
    sigma_pix = 16.e-6
    wbar = 1/(0.33e-15)
    B_cl = np.exp(-L*(L + 1)/l_s**2)

    err = np.sqrt(
        (2/((2*L+1)*f_sky)) * (Cl + wbar**(-1) * B_cl**-2)**2
    )
    return err
