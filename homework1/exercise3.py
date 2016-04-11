# Hint: you should only need the following functions from numpy and scipy
from numpy import diag, tril, triu, dot, array
from numpy.linalg import norm
from scipy.linalg import solve_triangular

import numpy as np

def decompose(A):
    D = diag(diag(A))
    L = tril(A,-1)
    U = triu(A,1)

    return D,L,U

def is_sdd(A):
    diagvals = diag(abs(A))
    rowsums = np.sum(abs(A),1)
    isstrictdom = all(2*diagvals > rowsums)

    if isstrictdom == True:
        return True
    else:
        return False

    

def jacobi_step(D, L, U, b, xk):
    pass

def jacobi_iteration(A, b, x0, epsilon=1e-8):
    pass

def gauss_seidel_step(D, L, U, b, xk):
    pass

def gauss_seidel_iteration(A, b, x0, epsilon=1e-8):
    pass


A = array([[10,2,3,4],[3,15,5,6],[5,6,20,8],[2,3,4,10]])
