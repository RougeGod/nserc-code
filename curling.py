import numpy as n
from scipy.integrate import quad
from scipy.integrate import solve_ivp

#constants. These are global to the entire program
ROCK_RAD = 0.14 #rock radius r (metres)
RUNNING_RAD = 0.063 #running band radius R (metres)
ETA = ROCK_RAD**2 / (2*(RUNNING_RAD**2)) #dimensionless
ROCK_MASS = 19 #kilograms
MU1 = 0.004 #ice-ice dimensionless friction coefficient
MU2 = 0.016 #ice-stone dimensionless friction coefficient

PHI0 = n.pi #critical angle
TOL = 1e-8 #for tolerance testing

def fphi(angle, phi0):
    GRAV = 9.81
    return (GRAV * ROCK_MASS * MU1) if (angle < phi0) else (GRAV * ROCK_MASS * MU2)

#global precomputed integrals (except for fInt, which is just a weighted average)
fInt =    ((MU1 * PHI0) + (MU2 * (2 * n.pi) - PHI0)) / (2 * n.pi)
sinInt  = ((quad(lambda intVar: fphi(intVar, PHI0)*n.sin(intVar), 0, PHI0)[0] 
          + quad(lambda intVar: fphi(intVar, PHI0)*n.sin(intVar),PHI0, 2*n.pi)[0]))
cosInt  = ((quad(lambda intVar: fphi(intVar, PHI0)*n.cos(intVar), 0, PHI0)[0] 
          + quad(lambda intVar: fphi(intVar, PHI0)*n.cos(intVar),PHI0, 2*n.pi)[0]))
sin2Int = ((quad(lambda intVar: fphi(intVar, PHI0)*(n.sin(intVar) ** 2), 0, PHI0)[0] 
          + quad(lambda intVar: fphi(intVar, PHI0)*(n.sin(intVar) ** 2),PHI0, 2*n.pi)[0]))

INITIAL_X = INITIAL_Y = INITIAL_N = 0 #yes, this is legal and there's no reference shenanigans
INITIAL_W = -0.05 #m/s
INITIAL_P = 2.196
initialParams = [INITIAL_X, INITIAL_Y, INITIAL_N, INITIAL_P, INITIAL_W]
def dX_dt(fx, params):
    return params[2]
def dY_dt(fy, params):
    return params[3]
def dN_dt(fn, params):
    constPack = 2 * n.pi * params[3] * ROCK_MASS
    return ((params[4] * (sinInt)) - (params[2] * fInt)) / constPack
def dP_dt(fP, params): 
    return fInt / (2 * n.pi * ROCK_MASS)
def dW_dt(fW, params):
    constPack = 4 * n.pi * ETA * ROCK_MASS
    t1 = params[2] * sinInt / params[3]
    t2 = cosInt
    t3 = params[4] * sin2Int / params[3]
    return ((t1 - t2) - t3) / constPack
def wholeSystem(fparams, params):
    bob =  [dX_dt(fparams, params), dY_dt(fparams, params), 
    dN_dt(fparams, params), dP_dt(fparams, params), dW_dt(fparams, params)]
    return bob
