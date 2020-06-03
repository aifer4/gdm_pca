"""
Mode evolution using 3-fluid (Seljak's 2-fluid + neutrinos) and GDM (Generalized Dark Matter).

"""

import numpy as np
from util import trapz, backtrapz, deriv
import numba


@numba.njit()
def DY_3fld(i, Y, A, K, wC, DwC, cs2C, wN, DwN, cs2N,
            OmegaC, OmegaN, OmegaB0, OmegaC0, OmegaG0, OmegaN0, H):
    """
    3-fluid perturbation mode derivative function.
    This function is called by the Runge-Kutta integrator in the solve_3fld
    function to get the derivative at each timestep. All quantities are in
    units where H0 = c = 1.

    Input:
    -i (int): Timestep index, used to get current values of H, w, etc.

    -Y (array [7, NK]): Perturbation mode values from last timestep. each
    row corresponds to one of the seven variables (phi, delta_g, v_g, 
    delta_c, v_c, delta_n, v_n). Each column corresponds to a
    specific value of k.

    -A (array [NT]): List of values of the scale factor.

    -K (array [NK]): List of values of the wavenumber magnitude.

    -wC (array [NT]): Equation of state of cold dark matter (fiducially 0).

    -DwC (array [NT]): Conformal-time derivative of wC.

    -cs2C (array [NT, NK]): Time- and scale-dependent sound speed of cold
    dark matter.

    -wN, DwN, cs2N: Same as above, but for neutrinos.

    Output:
    dY (array [7, NK]): Conformal-time derivative of perturbation modes,
    ordered the same way as the Y array.

    """
    dY = np.zeros((7, len(K)))
    Phi, deltaG, vG, deltaC, vC, deltaN, vN =\
        Y[0, :], Y[1, :], Y[2, :], Y[3, :], Y[4, :], Y[5, :], Y[6, :]

    OmegaBi = OmegaB0 * A[i]**-3.
    OmegaCi = OmegaC[i]
    OmegaGi = OmegaG0 * A[i]**-4.
    OmegaNi = OmegaN[i]

    ybi = OmegaBi/OmegaGi

    # compute the derivatives of the perturbations.
    DPhi = -H[i]*Phi + (3/2.*A[i]**2/K) *\
        (4./3.*(OmegaGi*vG + OmegaN[i]*vN) + OmegaC[i]*vC + OmegaBi*vG)

    DdeltaG = -4./3.*K*vG + 4*DPhi
    DvG = (-H[i] * ybi*vG + K*deltaG/3)/(
        4./3. + ybi) + K*Phi

    DdeltaC = -(1+wC[i])*(K*vC-3*DPhi) - 3*H[i]*(cs2C[i, :]-wC[i])*deltaC
    DvC = -H[i]*(1-3*wC[i])*vC - vC*DwC[i]/(1+wC[i]) + \
        K*deltaC*cs2C[i, :]/(1+wC[i]) + K*Phi

    DdeltaN = -(1+wN[i])*(K*vN-3*DPhi) - 3*H[i]*(cs2N[i, :]-wN[i])*deltaN
    DvN = -H[i]*(1-3*wN[i])*vN - vN*DwN[i]/(1+wN[i]) + \
        K*deltaN*cs2N[i, :]/(1+wN[i]) + K*Phi

    dY = np.stack((DPhi, DdeltaG, DvG, DdeltaC, DvC, DdeltaN, DvN))
    return dY


@numba.njit()
def solve_3fld(A, K, wC, cs2C, wN, cs2N, OmegaB0, OmegaC0,
               OmegaG0, OmegaN0):

    NT = len(A)

    # Compute perturbed background.
    OmegaC = OmegaC0 * A**-3 * np.exp(-3*backtrapz(A, wC/A))
    OmegaN = OmegaN0 * A**-4 * np.exp(-3*backtrapz(A, (wN-1/3)/A))
    H = A * np.sqrt(OmegaB0*A**-3 + OmegaC + OmegaG0*A**-4 + OmegaN)
    TAU = trapz(A, 1/(A * H))

    # Differentiate equations of state.
    DwC = deriv(TAU, wC)
    DwN = deriv(TAU, wN)

    # Use Seljak initial conditions.
    y0 = A[0]*(OmegaB0 + OmegaC0)/(OmegaG0 + OmegaN0)
    Phi0 = np.ones(len(K))
    deltaG0 = -2*Phi0*(1 + 3*y0/16)
    vG0 = -K/(H[0]) * (deltaG0/4 + (2*K**2 * (1 + y0)*Phi0) /
                       (9*(H[0])**2 * (4./3. + y0)))
    deltaC0 = .75 * deltaG0
    vC0 = vG0
    deltaN0 = deltaG0
    vN0 = vG0

    Y = np.zeros((NT//2, 7, len(K)))
    Y[0, :, :] = np.stack((Phi0, deltaG0, vG0, deltaC0, vC0, deltaN0, vN0))

    # Integrate using Runge-Kutta 4.
    for i in range(NT//2-1):
        ss = TAU[2*i+2] - TAU[2*i]
        k1 = ss*DY_3fld(2*i, Y[i, :, :], A, K, wC, DwC, cs2C, wN, DwN,
                        cs2N, OmegaC, OmegaN, OmegaB0, OmegaC0, OmegaG0, OmegaN0, H)
        k2 = ss*DY_3fld(2*i+1, Y[i, :, :]+k1/2, A, K, wC, DwC, cs2C, wN, DwN,
                        cs2N, OmegaC, OmegaN, OmegaB0, OmegaC0, OmegaG0, OmegaN0, H)
        k3 = ss*DY_3fld(2*i+1, Y[i, :, :]+k2/2, A, K, wC, DwC, cs2C, wN, DwN,
                        cs2N, OmegaC, OmegaN, OmegaB0, OmegaC0, OmegaG0, OmegaN0, H)
        k4 = ss*DY_3fld(2*i+2, Y[i, :, :]+k3, A, K, wC, DwC, cs2C, wN, DwN,
                        cs2N, OmegaC, OmegaN, OmegaB0, OmegaC0, OmegaG0, OmegaN0, H)
        Y[i+1, :, :] = Y[i, :, :] + k1/6 + k2/3 + k3/3 + k4/6
    return Y, TAU


@numba.njit()
def DY_2fld(i, Y, A, K, wD, DwD, cs2D, OmegaD,
            OmegaB0, OmegaC0, OmegaG0, OmegaN0, H):
    """
    2-fluid perturbation mode derivative function.
    This function is called by the Runge-Kutta integrator in the solve_2fld
    function to get the derivative at each timestep. All quantities are in
    units where H0 = c = 1.

    Input:
    -i (int): Timestep index, used to get current values of H, w, etc.

    -Y (array [7, NK]): Perturbation mode values from last timestep. each
    row corresponds to one of the seven variables (phi, delta_g, v_g, 
    delta_c, v_c, delta_n, v_n). Each column corresponds to a
    specific value of k.

    -A (array [NT]): List of values of the scale factor.

    -K (array [NK]): List of values of the wavenumber magnitude.

    -wC (array [NT]): Equation of state of cold dark matter (fiducially 0).

    -DwC (array [NT]): Conformal-time derivative of wC.

    -cs2C (array [NT, NK]): Time- and scale-dependent sound speed of cold
    dark matter.

    -wN, DwN, cs2N: Same as above, but for neutrinos.

    Output:
    dY (array [7, NK]): Conformal-time derivative of perturbation modes,
    ordered the same way as the Y array.

    """
    dY = np.zeros((5, len(K)))
    Phi, deltaG, vG, deltaD, vD =\
        Y[0, :], Y[1, :], Y[2, :], Y[3, :], Y[4, :]

    OmegaBi = OmegaB0 * A[i]**-3.
    OmegaCi = OmegaC0 * A[i]**-3.
    OmegaGi = OmegaG0 * A[i]**-4.
    OmegaNi = OmegaN0 * A[i]**-4
    OmegaDi = OmegaD[i]

    ybi = OmegaBi/OmegaGi

    # compute the derivatives of the perturbations.
    DPhi = -H[i]*Phi + (3/2.*A[i]**2/K) *\
        (4./3.*(OmegaGi*vG) + OmegaBi*vG + (1+wD[i])*OmegaD[i]*vD)

    DdeltaG = -4./3.*K*vG + 4*DPhi
    DvG = (-H[i] * ybi*vG + K*deltaG/3)/(
        4./3. + ybi) + K*Phi

    DdeltaD = -(1+wD[i])*(K*vD-3*DPhi) - 3*H[i]*(cs2D[i, :]-wD[i])*deltaD
    DvD = -H[i]*(1-3*wD[i])*vD - vD*DwD[i]/(1+wD[i]) + \
        K*deltaD*cs2D[i, :]/(1+wD[i]) + K*Phi
    dY[0, :] = DPhi
    dY[1, :] = DdeltaG
    dY[2, :] = DvG
    dY[3, :] = DdeltaD
    dY[4, :] = DvD
    dY = np.stack((DPhi, DdeltaG, DvG, DdeltaD, DvD))
    return dY


@numba.njit()
def solve_2fld(A, K, wD, cs2D, deltaD0, vD0, OmegaB0,
               OmegaC0, OmegaG0, OmegaN0, OmegaD0):

    NT = len(A)
    OmegaN = OmegaN0 * A**-4
    OmegaD_F = OmegaC0 * A**-3 + OmegaN0 * A**-4
    wD_F = OmegaN/(3*OmegaD_F)
    #wD_F = OmegaN0/(3* OmegaC0 * A + OmegaN0)
    OmegaD = OmegaD_F * np.exp(-3*trapz(A, (wD - wD_F)/A))

    H = A * np.sqrt(OmegaB0*A**-3 + OmegaG0*A**-4 + OmegaD)
    TAU = trapz(A, 1/(A * H))
    DwD = deriv(TAU, wD)

    # Use Seljak initial conditions for potential and photons.
    y0 = A[0]*(OmegaB0 + OmegaC0)/(OmegaG0 + OmegaN0)
    Phi0 = np.ones(len(K))
    deltaG0 = -2*Phi0*(1 + 3*y0/16)
    vG0 = -K/(H[0]) * (deltaG0/4 + (2*K**2 * (1 + y0)*Phi0) /
                       (9*(H[0])**2 * (4./3. + y0)))

    Y = np.zeros((NT//2, 5, len(K)))
    Y[0, :, :] = np.stack((Phi0, deltaG0, vG0, deltaD0, vD0))

    # Integrate using Runge-Kutta 4.
    for i in range(NT//2-1):
        ss = TAU[2*i+2] - TAU[2*i]
        k1 = ss*DY_2fld(2*i, Y[i, :, :], A, K, wD, DwD, cs2D, OmegaD,
                        OmegaB0, OmegaC0, OmegaG0, OmegaN0, H)
        k2 = ss*DY_2fld(2*i+1, Y[i, :, :]+k1/2, A, K, wD, DwD, cs2D, OmegaD,
                        OmegaB0, OmegaC0, OmegaG0, OmegaN0, H)
        k3 = ss*DY_2fld(2*i+1, Y[i, :, :]+k2/2, A, K, wD, DwD, cs2D, OmegaD,
                        OmegaB0, OmegaC0, OmegaG0, OmegaN0, H)
        k4 = ss*DY_2fld(2*i+2, Y[i, :, :]+k3, A, K, wD, DwD, cs2D, OmegaD,
                        OmegaB0, OmegaC0, OmegaG0, OmegaN0, H)
        Y[i+1, :, :] = Y[i, :, :] + k1/6 + k2/3 + k3/3 + k4/6
    return Y, TAU
