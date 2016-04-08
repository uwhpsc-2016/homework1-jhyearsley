# the documenation has been written for you in this exercise

def collatz_step(n):
    """Returns the result of the Collatz function.

    The Collatz function C : N -> N is used in `collatz` to generate collatz
    sequences. Raises an error if n < 1.

    Parameters
    ----------
    n : int

    Returns
    -------
    int
        The result of C(n).

    """

    if n<1:
        raise ValueError('n cannot be less than 1')
    elif n==1:
        return 1
    elif n%2==0:
        return (n/2)
    elif n%2==1:
        return (3*n+1)
        
        

def collatz(n):
    """Returns the Collatz sequence beginning with `n`.

    It is conjectured that Collatz sequences all end with `1`. Calls
    `collatz_step` at each iteration.

    Parameters
    ----------
    n : int

    Returns
    -------
    sequence : list
        A Collatz sequence.

    """
    seq = [n]

    while n>1:
        n = collatz_step(n)
        seq.append(n)

        if seq[-1]==1:
            return seq
