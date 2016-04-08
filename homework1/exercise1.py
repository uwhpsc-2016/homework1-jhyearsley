import numpy as np
import matplotlib.pyplot as plt



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
        raise ValueError('wrong')
    elif n==1:
        return 1
    elif n%2==0:
        return (n/2)
    elif n%2==1:
        return(3*n+1)


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
    if n==1:
        return [n]
    
    if n>1:
        seq = [n]
        while n>1:
         n = collatz_step(n)
         seq.append(n)

         if seq[-1]==1:
             return seq
N = 5000
m = np.zeros(N-1)

x = np.arange(1,N)

for n in np.arange(1,N):
    m[n-1] = len(collatz(n))

    
     
     
     
#plt.scatter(x,m,c='b',marker='o')
#plt.title('Collatz Function Stopping Times')
#plt.xlabel('Input to Collatz')
#plt.ylabel('Stopping Time')
#plt.show()
    

    
    



