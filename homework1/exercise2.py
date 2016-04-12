import numpy as np
import pdb
from math import pi
import matplotlib.pyplot as plt


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

    xnext = xk - sigma*df(xk)
    
    return   xnext  



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

    x = x + 1
    xnext = gradient_step(x,df,sigma)
    


    while abs(xnext-x) > epsilon:
        #pdb.set_trace()
        x = xnext
        xnext = gradient_step(x,df,sigma)
        #print (x,xnext)

        
    return xnext

#####Stuff for Report########
def f(x):
    return np.cos(x)

def df(x):
    df = -np.sin(x)
    return df


"""Actual plot of cos"""
x = np.linspace(0,2*pi,1000)
f = np.cos(x)

"""Plot iterations for gradient descent"""
sigma = 0.5
epsilon = 1e-8
x0 = 0.1
xnext = gradient_step(x0,df,sigma)

xvals = []

while abs(xnext-x0) > epsilon:
    x0 = xnext
    xnext = gradient_step(x0,df,sigma)

    xvals.append(xnext)


fvals = np.cos(xvals)
    


plt.hold(True)
plt.plot(x,f,c='k',linewidth=2,label='Cos(x)')
plt.plot(xvals,fvals,'ob',linewidth=20,label='Iteration Points')
plt.xlabel('x')
plt.ylabel('f(x)=cos(x)')
plt.title('Finding a Local Minima for Cos(x)')
plt.legend()
plt.plot()
plt.show()




        
   
#sigma = 0.4
##def f(x):
##    return x**3 - x**2
##
##def df(x):
##    return 3*x**2 - 2*x
##
##print gradient_descent(f,df,0,sigma=0.5)
