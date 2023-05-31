import numpy
import matplotlib.pyplot as plt
from scipy.integrate import quad

def fphi(angle, phi0):
 MU1 = 0.004
 MU2 = 0.016
 GRAV = 9.81
 ROCK_MASS = 19
 return (GRAV * ROCK_MASS * MU1) if (angle < phi0) else (GRAV * ROCK_MASS * MU2)

cosAvg = numpy.empty(180)
sinAvg = numpy.empty(180)
sin2Avg = numpy.empty(180)

for CRIT_ANGLE in range (180,360):
    cosAvg[CRIT_ANGLE - 180]  = quad(lambda t: fphi(t, numpy.deg2rad(CRIT_ANGLE))*numpy.cos(t), 0, numpy.deg2rad(CRIT_ANGLE))[0]/(2 * numpy.pi) 
    + quad(lambda t: fphi(t, numpy.deg2rad(CRIT_ANGLE))*numpy.cos(t), numpy.deg2rad(CRIT_ANGLE), numpy.pi * 2)[0] / (2 * numpy.pi)
    sinAvg[CRIT_ANGLE - 180]  = quad(lambda t: fphi(t, numpy.deg2rad(CRIT_ANGLE))*numpy.sin(t), 0, numpy.deg2rad(CRIT_ANGLE))[0]/(2 * numpy.pi) 
    + quad(lambda t: fphi(t, numpy.deg2rad(CRIT_ANGLE))*numpy.sin(t), numpy.deg2rad(CRIT_ANGLE), numpy.pi * 2)[0] / (2 * numpy.pi)
    sin2Avg[CRIT_ANGLE - 180] = quad(lambda t: fphi(t, numpy.deg2rad(CRIT_ANGLE))*numpy.sin(t) ** 2, 0, numpy.deg2rad(CRIT_ANGLE))[0]/(2 * numpy.pi) 
    + quad(lambda t: fphi(t, numpy.deg2rad(CRIT_ANGLE))*numpy.sin(t) ** 2, numpy.deg2rad(CRIT_ANGLE), numpy.pi * 2)[0] / (2 * numpy.pi)
print(errAvg)

figure, axes = plt.subplots()
axes.plot(range(180,360), cosAvg, "-r", linewidth=2)
axes.set(title="Average of f(ɸ)cos(ɸ) vs Critical Angle", ylabel="<f(ɸ)cos(ɸ)> (Newtons)", xlabel="ɸₒ (degrees)")
plt.show()
figure, axes = plt.subplots()
axes.set(title="Average of f(ɸ)sin(ɸ) vs Critical Angle", ylabel="<f(ɸ)sin(ɸ)> (Newtons)", xlabel="ɸₒ (degrees)")
axes.plot(range(180,360), sinAvg, "-m", linewidth=2)
plt.show()
figure, axes = plt.subplots()
axes.set(title="Average of f(ɸ)sin²(ɸ) vs Critical Angle", ylabel="<f(ɸ)sin²(ɸ)> (Newtons)", xlabel="ɸₒ (degrees)")
axes.plot(range(180,360),sin2Avg, "-g", linewidth=2)
plt.show()
