from scipy.optimize import fsolve, newton
import math
import random
#import numpy as np
from decimal import *

def equation(p, *data):
    n, z = data

    with localcontext() as ctx:
        ctx.prec = 64
        y= Decimal(p[0].astype(float))
        sumation = 0
        for i in range(n):
            sumation += ctx.power(y, i)/math.factorial(i)

##
        remainder = y.exp()*(1-Decimal(z)) - sumation
##        remainder_float = math.exp(epsilon*p[0])*(1-z)-float(sumation)
        
    return float(remainder)

def fprime(p, *data):
    n, z = data
    with localcontext() as ctx:
        ctx.prec = 128
        y= Decimal(p)
        sumation = 0
        for i in range(n-1):
            sumation += ctx.power(y, i)/math.factorial(i)

        remainder = y.exp()*(1-Decimal(z)) - sumation

    return float(remainder)


## differentially private sampling: n - size of the noise vector,
## epsilon - privacy budget, estimate - initialize solution
def sampling(n, epsilon, estimate = 200):
    ## sample the radius
    z = random.random()
    data = (n, z)

    while True:
#        y =  newton(equation, estimate, fprime, args=data, maxiter=100000)
        y = fsolve(equation, estimate, args=data, maxfev=5000)
        if y>0:
            break
        else:
            estimate +=100

    x = y/epsilon


    ## sample unit vector
    p_sum = 0
    u = []

    for i in range(n):
        num = random.gauss(0,1)
        u.append(num)
        p_sum += num**2
        

    ## normalize u
    for i in range(len(u)):
        u[i] = u[i] / math.sqrt(p_sum)


    return u*x


##noise_vec = sampling(100, 0.1, 150)


## E.g., add noise_vec to singular values in ICME 2019  
