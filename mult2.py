import numdifftools as nd
import numpy as np

def multi_newtons(x0, f, tol=1e-5):
    x = x0
    H0 = np.linalg.inv(nd.Hessian(f)(x0))
    grad0 = nd.Gradient(f)(x0)
    x1 = x - H0@grad0

    #while abs(x1 - x) > tol:
     #   x = x1
      #  H = np.linalg.inv(nd.core.Hessian(f(x)))
       # grad = nd.core.Gradient(f(x))
        #x1 = x - Hgrad
        
    return x1