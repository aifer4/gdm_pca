"""Some helper functions are implemented here so that they can be called within numba-compiled functions."""

import numba
import numpy as np

@numba.njit
def trapz(x,f):
    """Trapezoidal rule cumulative integral.
    Inputs: 
    x (array): values of the independent variable.
    f (array): values of the dependent variable. Must be the same length as x. 
    
    Returns: 
    F (array): The cumulative integral of f(x) with F[0] = 0.
    """
    N = len(f)
    F = np.zeros(N)
    F[0] = 0
    for i in range(1, N):
        F[i] = F[i-1] + (x[i]-x[i-1])*(f[i]+f[i-1])/2
    return F


@numba.njit
def backtrapz(x,f):
    """Backwards trapezoidal rule cumulative integral.
    Inputs: 
    x (array): values of the independent variable.
    f (array): values of the dependent variable. Must be the same length as x. 
    
    Returns: 
    F (array): The cumulative integral of f(x) with F[-1] = 0.
    """
    N = len(f)
    F = np.zeros(N)
    F[-1] = 0
    for i in range(2, N+1):
        F[-i] = F[1-i] - (x[1-i]-x[-i])*(f[1-i]+f[-i])/2
    return F

@numba.njit
def deriv(x,f):
    df = np.zeros(len(f))
    df[1:] = np.diff(f)/np.diff(x)
    df[0] = df[1]
    return df
