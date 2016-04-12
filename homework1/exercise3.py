# Hint: you should only need the following functions from numpy and scipy
from numpy import diag, tril, triu, dot, array
from numpy.linalg import norm
from scipy.linalg import solve_triangular
from scipy import linalg

import numpy as np

def decompose(A):
    """Decomposes the matrix A into diagonal, upp-diagonal, and low-diagonal matrices

    Given a matrix A, this function returns three matrices: D (the matrix containing the
    diagonal components of A and zeros everywhere else), L (the lower triangular matrix
    excluding the diagonal of A), and U (the upper triangular matrix excluding the diagonal
    of A).

    Parameters:
    ----------
    A : 2d numpy array

    Returns:
    -------
    D : 2d numpy array
        A diagonal numpy array with the diagonal componenets of A and zeros everywhere else.

    L : 2d numpy array
        A lower triangular numpy array including the lower triangular components of A (excluding
        the diagonal of A) and zeros everywhere else.

    U : 2d numpy array
        An upper triangular numpy array including the upper triangular components of A (exluding
        the diagonal of A) and zeros everywhere else.
    """
    
    
    D = diag(diag(A))
    L = tril(A,-1)
    U = triu(A,1)

    return D,L,U

def is_sdd(A):
    """Determines whether the matrix A is strictly diagonally dominant.

    A matrix A is strictly diagonally dominant (SDD) if for each row of A, the absolute value
    of the sum of all non-diagonal components is strictly smaller than the absolute value
    of the diagonal component of the respective row. A simple way to test if a matrix is
    SDD is to check that the inequality 2*diagval > rowsum, where diagval is the absolute
    value of the diagonal component and rowsum is the absolute value of all non-diagonal
    components in the resepctive row

    Parameters:
    ----------
    A : 2d numpy array

    Returns:
    True : returns the value 'True' when A is SDD

    False : returns the value 'False' when A is not SDD
    """

    
    diagvals = diag(abs(A))
    rowsums = np.sum(abs(A),1)
    isstrictdom = all(2*diagvals > rowsums)

    if isstrictdom == True:
        return True
    else:
        return False

    

def jacobi_step(D, L, U, b, xk):
    """Given an initial guess x_k, returns the next iterate x_{k+1} by the Jacobi method

    The Jacobi iteration technique takes a diagonal matrix D, a lower triangular matrix L
    , and upper triangular matrix U, a vector b, and an initial guess xk. The aglorithm
    is as follows: D*x_{k+1} = b - (L + U)*x_k <==> x_{k+1} = D^(-1)*b - D^(-1)*(L + U)*x_k.

    Parameters:
    ----------
    D : 2d numpy array
        diagonal matrix

    L : 2d numpy array
        lower triangular matrix

    U : 2d numpy array
        upper traingular matrix

    b : 1d numpy array

    xk: 1d numpy array
        Initial guess 

    """
    diagD = diag(D)
    diagones = np.ones(len(diagD))
    Dinv = diag(diagones/diagD)

    xnext = Dinv.dot(b) -Dinv.dot(L + U).dot(xk)

    return xnext

def jacobi_iteration(A, b, x0, epsilon=1e-8):
    """Performs jacobi iteration to solve matrix equation Ax=b with initial guess x0

    Using the previously defined function 'jacobi_step' this function performs jacobi iteration
    until converging to a solution x_k, such that that x_k is a solution to the matrix equation
    Ax=b.

    Parameters:
    ----------
    A : 2d numpy array
        The matrix representation of the coefficients in the system of equations you want to solve.

    b : 1d numpy array
        The vector representation of the known values each equation in the respective system is equal
        to.

    x0 : 1d numpy array
        Initial guess for the unknown variables to solve the matrix equation Ax=b.

    epsilon : keyword argument
        Optional parameter to adjust convergence tolerance.

    Returns:
    -------
    xsol : 1d numpy array
        Returns the vector 'xsol' such that, 'xsol' is a solution to the matrix equation Ax=b.
    
    """

    #Initialize parameters
    D, L, U = decompose(A)
    xnext = jacobi_step(D, L, U, b, x0)
    iterations = []

    #Perform Jacobi-iteration until converging to a solution

    while norm(xnext - x0) > epsilon:
        x0 = xnext
        xnext = jacobi_step(D, L, U, b, x0)
        iterations.append(1)


    xsol = xnext
    numiter = len(iterations)

    return xsol, numiter

def gauss_seidel_step(D, L, U, b, xk):
    pass

def gauss_seidel_iteration(A, b, x0, epsilon=1e-8):
    pass


#A = array([[10,2,3,4],[3,15,5,6],[5,6,20,8],[2,3,4,10]])

A = array([[2,1],[3,4]])
b = array([7,18])
x0 = np.random.rand(2)
print jacobi_iteration(A, b, x0)
