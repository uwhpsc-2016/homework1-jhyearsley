
def gradient_step(xk, df, sigma):
    """Returns the next iterate x_{k+1} given x_k, the derivative 'f', and parameter sigma

     """
   return  x_next = xk - simga*df(xk) 


    
def gradient_descent(f, df, x, sigma=0.5, epsilon=1e-8):
    """Returns a minima of `f` using the Gradient Descent method.

    A local minima, x*, is such that `f(x*) <= f(x)` for all `x` near `x*`.
    This function returns a local minima which is accurate to within `epsilon`.

    `gradient_descent` raises a ValueError if `sigma` is not strictly between
    zero and one.

    [STUDENTS: FILL IN THE REST OF THE DOCUMENTATION!]
    """
    pass
