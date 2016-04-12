import numpy as np
import pdb
#import matplotlib.pyplot as plt


def gradient_step(xk, df, sigma):
    """Returns the next iterate x_{k+1} given x_k, the derivative "f'", and parameter sigma

    The next iterate x_{k+1} is found by evaluating the difference between x_k and the
    product of sigma with the derivative function f' evaluated at x_k.

    Parameters
    ----------
    xk : float
       The value at iteration k

    df : function
       The derivative of the function 'f'

    sigma : float
       parameter between 0 and 1


    Returns
    -------
    x_next : float
       gives the next iterate x_{k+1}
     """

    
    return   xnext = xk - sigma*df(xk) 



    def gradient_descent(f, df, x, sigma=0.5, epsilon=1e-8):
    """Returns a minima of `f` using the Gradient Descent method.

    A local minima, x*, is such that `f(x*) <= f(x)` for all `x` near `x*`.
    This function returns a local minima which is accurate to within `epsilon`.

    `gradient_descent` raises a ValueError if `sigma` is not strictly between
    zero and one.


    Parameters
    ----------
    f : function
      This is the function you want to minimize.

    df : function
      This is the derivative of the function you want to minimize.

    x : float
      Initial guess to start gradient descent. This value is important! If you
      guess closer to the actual value, convergence will be faster.

    sigma : keyword argument
      Optional parameter between 0 and 1 for scaling descent.
      
    epsilon: keyword argument
      Optional parameter between 0 and 1 to adjust convergence tolerance


    Returns
    -------
    xmin : float
      This is the local minima x* such that f(x*)<=f(x) in the neighborhood of x*.


     """

    if not(0<sigma<1):
        raise ValueError('Sigma must be a value strictly between 0 and 1')
    
    #epsilon = 1e-8
    #sigma = 0.25

    xnext = gradient_step(x,df,sigma)
   # x = x + 1


    while abs(xnext-x) > epsilon:
        #pdb.set_trace()
        x = xnext
        xnext = gradient_step(x,df,sigma)
        #print (x,xnext)

        
    return xnext
        
   
#sigma = 0.4
def f(x):
    return x**3 - x**2

def df(x):
    return 3*x**2 - 2*x

print gradient_descent(f,df,0,sigma=0.5)


#for i in range(20):
#        x = xnext
#        xnext = gradient_step(x,df,sigma)
#        print (x,xnext)
        


        
