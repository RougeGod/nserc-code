import numpy as n
from scipy.integrate import quad

ROCK_RAD = 0.14 #rock radius r (metres)
RUNNING_RAD = 0.063 #running band radius R (metres)
ETA = ROCK_RAD**2 / (2*(RUNNING_RAD**2)) #dimensionless
ROCK_MASS = 19 #kilograms
MU1 = 0.004 #ice-ice dimensionless friction coefficient
MU2 = 0.016 #ice-stone dimensionless friction coefficient
GRAV = 9.81
PHI0 = n.pi

def fphi(angle, phi0):
    GRAV = -9.81
    return (GRAV * ROCK_MASS * MU1) if (angle < phi0) else (GRAV * ROCK_MASS * MU2)

def integrals(PSI):
    sinInt  = ((quad(lambda intVar: fphi(intVar, PHI0)*n.sin(intVar), PSI, PHI0 + PSI)[0] 
          + quad(lambda intVar: fphi(intVar, PHI0)*n.sin(intVar),PHI0 + PSI, 2*n.pi + PSI)[0])) * ROCK_MASS
    cosInt  = ((quad(lambda intVar: fphi(intVar, PHI0)*n.cos(intVar), PSI, PHI0 + PSI)[0]
          + quad(lambda intVar: fphi(intVar, PHI0)*n.cos(intVar),PHI0 + PSI, 2*n.pi + PSI)[0])) * ROCK_MASS
    sin2Int = ((quad(lambda intVar: fphi(intVar, PHI0)*(n.sin(intVar) ** 2), PSI,  + PSI)[0] 
          + quad(lambda intVar: fphi(intVar, PHI0)*(n.sin(intVar) ** 2),PHI0 + PSI, PSI + 2*n.pi)[0])) * ROCK_MASS
    return (sinInt, cosInt, sin2Int)
def sums(PSI):
    sinSum = ROCK_MASS * GRAV * (MU2 - MU1) * (n.cos(PHI0 + PSI) - n.cos(PSI))
    cosSum = ROCK_MASS * GRAV * (MU2 - MU1) * (n.sin(PSI) - n.sin(PHI0 + PSI))
    sin2Sum = (ROCK_MASS * GRAV * (MU2 - MU1) / 2) * ((n.sin(2*PHI0 + 2*PSI)/2) - (n.sin(2*PSI)/2) - PHI0)
    return (sinSum, cosSum, sin2Sum)

testVals = n.linspace(0,n.pi,num=500)
sinResults = n.empty(500)
cosResults = n.empty(500)
sin2Results = n.empty(500)

for count in range(500):
    intRes = integrals(testVals[count])
    sumRes = sums(testVals[count])
    sinResults[count] = abs(intRes[0] - sumRes[0])
    cosResults[count] = abs(intRes[1] - sumRes[1])
    sin2Results[count] = abs(intRes[2] - sumRes[2])
print(min(sinResults))
print(max(cosResults))
print(max(sin2Results))

import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.plot(testVals, sin2Results, "-b", linewidth=1)
ax.set(xlim=(min(testVals),max(testVals)), ylim=(min(sin2Results),max(sin2Results)), title="Spin Rate Evolution",
xlabel="Time (s)", ylabel="Linear Angular Velocity w (m/s)")
plt.show()