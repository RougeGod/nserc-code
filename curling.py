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

TOL = 1e-10 #for tolerance testing

def fphi(angle, phi0):
    GRAV = -9.81
    return (GRAV * ROCK_MASS * MU1) if (angle < phi0) else (GRAV * ROCK_MASS * MU2)




def phi0(w):
  return n.pi
  A = 13.1286615
  B = 0.122522839
  C = 75.00122
  D = 249.260950
  w = abs(w)
  if (w <= 0.03): #experimental data showed a phi0 of 180 degrees at this value
                  #so at this value and below, phi0 is taken as 0. The atan-based
                  #function isn't super accurate in this area
    return n.pi
  elif (w <= 0.92786298936722): #interval ending at the point that the atan-based function 
                                #reaches 360 degrees. converted to radians for the numpy trig functions
    return n.deg2rad((C*n.arctan(A*(w - B)) + D))
  else: #if the atan function results in a value higher than 360 degrees, return 2pi
    return 2*n.pi

PHI0 = n.deg2rad(185)
def computeIntegrals(xVel, yVel, w): 
  PSI = n.arctan(-xVel/yVel)
  #PHI0 = phi0(w)
  #print(PHI0)
  #NONE OF THESE FUNCTIONS DIVIDE BY 2PI, as the equations in the DE system do that themselves
  #because it makes the equations simpler to write. 
  fInt =  ROCK_MASS * ((MU1 * PHI0) + (MU2 * (2 * n.pi) - PHI0)) / (2 * n.pi)
  fsinInt  = ROCK_MASS * GRAV * (MU2 - MU1) * (n.cos(PHI0 + PSI) - n.cos(PSI))
  fcosInt  = ROCK_MASS * GRAV * (MU2 - MU1) * (n.sin(PSI) - n.sin(PHI0 + PSI))
  fsin2Int = (ROCK_MASS * GRAV * (MU2 - MU1) / 2) * ((n.sin(2*PHI0 + 2*PSI)/2) - (n.sin(2*PSI)/2) - PHI0) 
  return (fsinInt, fcosInt, fsin2Int,fInt)

INITIAL_X = INITIAL_Y = INITIAL_N = 0 #yes, this is legal and there's no reference shenanigans
                                      #INITIAL_N is the x-velocity
INITIAL_W = +0.05 #m/s, positive value indocates anti-clockwise spin
INITIAL_P = 2.196 #m/s, initial y-velocity of the throw
initialParams = [INITIAL_X, INITIAL_Y, INITIAL_N, INITIAL_P, INITIAL_W]


#something's still not quite right, I don't think. Spin goes significantly faster and the rock travels further
#than the paper says in y and not as far in x, and also travels for longer
def dX_dt(fx, params):
    return params[2]
def dY_dt(fy, params):
    return params[3]
def dN_dt(fn, params): #direct equation plugging
    trigs = computeIntegrals(params[2], params[3], params[4])
    constPack = 2 * n.pi * params[3] * ROCK_MASS
    return ((params[4] * (trigs[0])) - (params[2] * trigs[3])) / (constPack * ROCK_MASS)
def dP_dt(fP, params): 
    trigs = computeIntegrals(params[2], params[3], params[4])
    return trigs[3] / (2 * n.pi * ROCK_MASS)
def dW_dt(fW, params):
    constPack = 4 * n.pi * ETA * ROCK_MASS
    trigs = computeIntegrals(params[2], params[3], params[4])
    t1 = params[2] * trigs[0] / params[3]
    t2 = trigs[1]
    t3 = params[4] * trigs[2] / params[3]
    return ((t1 - t2) - t3) / constPack 
    
T_ARRAY = []
X_POS_ARRAY = []
Y_POS_ARRAY = []
X_VEL_ARRAY = []
Y_VEL_ARRAY = []
W_ARRAY = []

def wholeSystem(fparams, params):
    bob =  [dX_dt(fparams, params), dY_dt(fparams, params), 
    dN_dt(fparams, params), dP_dt(fparams, params), dW_dt(fparams, params)]
#    velStop(fparams, params[3])
    return bob
#def velStop(t,yVel): #finds when y-velocity ceases
#    if (yVel < TOL / 1000):
#        print(t)

END = 28.62629654474325 #25.06186400932 for constant phi0, 28.626296544743 for non-constant

solution = solve_ivp(wholeSystem, (0.0,END), initialParams, atol=TOL, rtol=TOL)
#print(solution.y)
