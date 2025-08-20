import numpy as np

def first_derivative(x, f, eps = 0.001):
    return (f(x + eps) - f(x)) / eps

def second_derivative(x, f, eps = 0.001):
    return (first_derivative(x + eps, f) - first_derivative(x, f)) / eps


def newtons_method(x0, f_0, eps = 0.001):
    '''Run Newton's method to minimize a function.
    '''
    f_1 = first_derivative(x0, f_0, eps)
    f_2 = second_derivative(x0, f_0, eps)

    xt_1 = x0
    xt = x0 - f_1 / f_2
    
    while abs(xt - xt_1) > eps:
        f_1 = first_derivative(xt, f_0)
        f_2 = second_derivative(xt, f_0)
        xt_1 = xt
        xt = xt_1 - f_1 / f_2

    return xt

def f_0(x):
    return np.cos(x)


print(newtons_method(1, f_0))