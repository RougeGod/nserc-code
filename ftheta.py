#trying to see what the force-average terms come out to. 
#does a pass every degree

import numpy
import matplotlib.pyplot as plt


MU1 = 0.004
MU2 = 0.016
GRAV = 9.81

cosAvg = numpy.empty(360)
sinAvg = numpy.empty(360)
sin2Avg = numpy.empty(360)

for CRIT_ANGLE in range (0,360):
  cos = numpy.empty(360)
  sin = numpy.empty(360)
  sin2 = numpy.empty(360)
  
  for count in range(0,360):
    ftheta = (GRAV * MU1) if (count >= CRIT_ANGLE) else (GRAV * MU2)
    cos[count] = ftheta * numpy.cos(count*numpy.pi/180)
    sin[count] = ftheta * numpy.sin(count*numpy.pi/180)
    sin2[count] = sin[count] * numpy.sin(count*numpy.pi/180)
  cosAvg[CRIT_ANGLE - 180]  = numpy.mean(cos)
  sinAvg[CRIT_ANGLE - 180]  = numpy.mean(sin)
  sin2Avg[CRIT_ANGLE - 180] = numpy.mean(sin2)
print(numpy.amax(cosAvg) , numpy.amin(cosAvg)) #this indicates that the output is not exactly a sine wave



figure, axes = plt.subplots()
axes.axvline(180, color="black")
axes.plot(range(0,360), cosAvg, "-r", linewidth=2)
plt.show()
figure, axes = plt.subplots()
axes.plot(range(0,360), sinAvg, "-m", linewidth=2)
plt.show()
figure, axes = plt.subplots()
axes.plot(range(0,360),sin2Avg, "-g", linewidth=2)
plt.show()