'''TODO on Monday: Make sure that the changing value of phi0 (which depends on w by some yet-unknown function 
that probably also needs to be figured out) is incorporated in the equations, along with the recomputation 
of the trigonometric integrals. https://docs.scipy.org/doc/scipy/tutorial/interpolate/1D.html#monotone-interpolants
will hopefully find some kind of fuction from w to phi0. Could play around with that

Figure out what's causing the division by near-zero is curling1, and what's NOT causing it in curling2, since
both do appear to have a 1/P term as P goes to zero, but only curling1 produces the wacky results. 

Update the endpoint (either with events or with hacky pre-checking) so that the integration stops when 
the angular speed is <= the CoM speed and static friction kicks in. This might fix the wonky division stuff. 
Would still like to know where taht comes from though. '''








import numpy as n
from scipy.integrate import quad
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

#constants. These are global to the entire program
ROCK_RAD = 0.14 #rock radius r (metres)
RUNNING_RAD = 0.063 #running band radius R (metres)
ETA = ROCK_RAD**2 / (2*(RUNNING_RAD**2)) #dimensionless
ROCK_MASS = 19 #kilograms
MU1 = 0.004 #ice-ice dimensionless friction coefficient
MU2 = 0.016 #ice-stone dimensionless friction coefficient
GRAV = 9.81 #if negative, breaks the equations, so it must be positive

PHI0 = n.pi #critical angle
TOL = 1e-4 #for tolerance testing

MUX = (MU2 - MU1) * (1 - n.cos(PHI0)) / (2 * n.pi)
MUY = MU2 - ((MU2 - MU1) * PHI0 / (2*n.pi))

INITIAL_X = INITIAL_Y = INITIAL_N = 0 #yes, this is legal and there's no reference shenanigans
INITIAL_W = 0.05 #m/s
INITIAL_P = 2.196
initialParams = [INITIAL_X, INITIAL_Y, INITIAL_N, INITIAL_P, INITIAL_W]
def dX_dt(fx, params):
    return params[2] #dx/dt = N
def dY_dt(fy, params):
    return params[3] #dy/dt = P
def dN_dt(fn, params): #dN/dt == dx2/d2t
    return (-1/params[3]) * ((MUX * GRAV * params[4]) + (MUY * GRAV * params[2]))
def dP_dt(fP, params): 
    return (-1) * MUY * GRAV
def dW_dt(fW, params):
    #constPack = 4 * n.pi * ETA * ROCK_MASS
    t1 = (-MUY * GRAV * params[4] / params[3])
    t2 = (-1) * (MU2 - MU1) * GRAV * n.sin(2 * PHI0) / (4*n.pi)
    t3 = (MU2 - MU1) * GRAV * n.sin(PHI0) / n.pi
    return (t1 + t2 + t3) / (4 * ETA)

X_ARRAY = []
X_VEL_ARRAY = []
T_ARRAY = []
def wholeSystem(fparams, params):
    bob =  [dX_dt(fparams, params), dY_dt(fparams, params), 
    dN_dt(fparams, params), dP_dt(fparams, params), dW_dt(fparams, params)]
    T_ARRAY.append(fparams)
    X_VEL_ARRAY.append(params[0])
    return bob
#def velStop(t,yVel): #finds when y-velocity ceases, sets the END parameter
#    if (abs(yVel[3]) < TOL):
#        print(t, yVel[3])
#    return yVel[3]
#values: phi0 = 3pi/2, END = 31.97903014416767, pi: 22.385321100917416
END = 22.385321100917316 #the point at which y-velocity stops
solution = (solve_ivp(wholeSystem, (0.0,END), initialParams, t_eval=[5,10,20,END], atol=TOL, rtol=TOL))#,events=velStop))

import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.plot(T_ARRAY, X_VEL_ARRAY, "-b", linewidth=1)
#ax.set(xlim=(min(T_ARRAY),max(T_ARRAY)), ylim=(min(X_VEL_ARRAY),max(X_VEL_ARRAY)), title="Forward Position Evolution",
#xlabel="Time (s)", ylabel="Y-Position (m)")
plt.show()


