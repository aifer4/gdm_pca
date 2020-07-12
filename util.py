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



@numba.njit()
def Powit(J_static, Sigma, m, V0):
<<<<<<< HEAD
<<<<<<< HEAD
    """Power iteration eigensolver. Gets largest m eigenvalues and eigenvectors of J^T \Sigma J."""
=======
    """Power iteration eigensolver. Gets largest m eigenvalues of J^T \Sigma J."""
>>>>>>> dae5f2455fcc600f40d1263b3305483a562231f0
=======
    """Power iteration eigensolver. Gets largest m eigenvalues of J^T \Sigma J."""
>>>>>>> dae5f2455fcc600f40d1263b3305483a562231f0
    J = np.copy(J_static)
    n = 50
    # get N largest eigenvalues and eigenvectors of A using power iteration
    N = len(J.T)
    err = np.zeros((m,n))
    
    # arrays U and L eigenvectors and eigenvalues respectively
    U = np.zeros((N,m))
    L = np.zeros(m)
    for i in range(m):
        # initialize the eigenvector guess
        u = V0[:,i]
        # iterate to find the ith eigenvector
        for j in range(n):
            # compute the product (J^T \Sigma J)u step by step.
            u_n = J.T@(np.diag(Sigma)@(J@u))
            u_n/=np.linalg.norm(u_n)
            err[i,j]=np.linalg.norm(u_n-u)
            u = u_n
        U[:,i]=u
        # compute the rayleigh quotient without making a big matrix:
        RQ_num = (J@u)@np.diag(Sigma)@(J@u)
        L[i]=RQ_num/(u.T@u)
        J -= np.outer(J@u,u)
    return L,U, err