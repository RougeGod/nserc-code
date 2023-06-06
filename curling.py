'''TODO on Monday: Make sure that the changing value of phi0 (which depends on w by some yet-unknown function 
that probably also needs to be figured out) is incorporated in the equations, along with the recomputation 
of the trigonometric integrals. https://docs.scipy.org/doc/scipy/tutorial/interpolate/1D.html#monotone-interpolants
will hopefully find some kind of fuction from w to phi0. Could play around with that

Figure out what's causing the division by near-zero is curling1, and what's NOT causing it in curling2, since
both do appear to have a 1/P term as P goes to zero, but only curling1 produces the wacky results. 

Update the endpoint (either with events or with hacky pre-checking) so taht teh integration stops when 
the angular speed is <= the CoM speed and static friction kicks in. This might fix the wonky division stuff. 
Would still like to know where taht comes from though. '''





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
GRAV = 9.81

PHI0 = n.pi #critical angle
TOL = 1e-9 #for tolerance testing

def fphi(angle, phi0):
    GRAV = -9.81
    return (GRAV * ROCK_MASS * MU1) if (angle < phi0) else (GRAV * ROCK_MASS * MU2)

#global integrals (except for fInt, which is just a weighted average)
fInt =  ROCK_MASS * ((MU1 * PHI0) + (MU2 * (2 * n.pi) - PHI0)) / (2 * n.pi)

def computeIntegrals(xVel, yVel): 
  #calculating psi (as opposed to leaving it constant)
  #cuts x-vel and w by about 9 orders of magnitude,and x-pos by about 7 orders of magnitude
  #does recalculating PHI0 do the same???????
  PSI = n.arctan(-xVel/yVel)
  sinInt  = ((quad(lambda intVar: fphi(intVar, PHI0)*n.sin(intVar), PSI, PHI0 + PSI)[0] 
          + quad(lambda intVar: fphi(intVar, PHI0)*n.sin(intVar),PHI0 + PSI, 2*n.pi + PSI)[0])) * ROCK_MASS
  cosInt  = ((quad(lambda intVar: fphi(intVar, PHI0)*n.cos(intVar), PSI, PHI0 + PSI)[0] 
          + quad(lambda intVar: fphi(intVar, PHI0)*n.cos(intVar),PHI0 + PSI, 2*n.pi + PSI)[0])) * ROCK_MASS
  sin2Int = ((quad(lambda intVar: fphi(intVar, PHI0)*(n.sin(intVar) ** 2), PSI,  + PSI)[0] 
          + quad(lambda intVar: fphi(intVar, PHI0)*(n.sin(intVar) ** 2),PHI0 + PSI, PSI + 2*n.pi)[0])) * ROCK_MASS
  return (sinInt, cosInt, sin2Int)

INITIAL_X = INITIAL_Y = INITIAL_N = 0 #yes, this is legal and there's no reference shenanigans
INITIAL_W = -0.05 #m/s
INITIAL_P = 2
initialParams = [INITIAL_X, INITIAL_Y, INITIAL_N, INITIAL_P, INITIAL_W]

def dX_dt(fx, params):
    return params[2]
def dY_dt(fy, params):
    return params[3]
#the simplified version from the paper cuts 13 orders of magnitude from final w and 
#final x-vel, and 12 orders of magnitude from x-pos
#def dN_dt(time, params): #paper's simplification
#    MUX = (MU2 - MU1) * (1 - n.cos(PHI0)) / (2 * n.pi)
#    MUY = MU2 - ((MU2 - MU1) * PHI0 / (2*n.pi))
#    return (-GRAV/params[3]) * ((MUX * params[4]) + (MUY * params[2]))
def dN_dt(fn, params): #direct equation plugging
    trigs = computeIntegrals(params[2], params[3])
    constPack = 2 * n.pi * params[3] * ROCK_MASS
    return ((params[4] * (trigs[0])) - (params[2] * fInt)) / (constPack * ROCK_MASS)
def dP_dt(fP, params): 
    return fInt / (2 * n.pi * ROCK_MASS)
def dW_dt(fW, params):
    constPack = 4 * n.pi * ETA * ROCK_MASS
    trigs = computeIntegrals(params[2], params[3])
    t1 = params[2] * trigs[0] / params[3]
    t2 = trigs[1]
    t3 = params[4] * trigs[2] / params[3]
    return ((t1 - t2) - t3) / constPack
def wholeSystem(fparams, params):
    bob =  [dX_dt(fparams, params), dY_dt(fparams, params), 
    dN_dt(fparams, params), dP_dt(fparams, params), dW_dt(fparams, params)]
    return bob
#def velStop(t,yVel): #finds when y-velocity ceases
#    if (yVel[3] < TOL):
#        print(t, yVel[3])
#    return yVel[3]
END = 26.071308328544745 #from velStop, I can't figure out terminal events so here's a workaround
print(solve_ivp(wholeSystem, (0.0,END), initialParams, t_eval=[0,10,20,END], atol=TOL, rtol=TOL))#, events=velStop))

#something is terribly wrong. debug tomorrow