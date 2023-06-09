import matplotlib.pyplot as plt
import json
timeFile = open("timesteps.txt", "r", encoding="utf-8")
allArrays = json.load(timeFile)

'''structure of allArrays:
0 - 5 (inclusive): phi0 is treated as a constant pi
6 - 11 (inclusive): phi0 is recalculated and changes the trajectories
0,6: time
1,7: x-position
2,8: y-position
3,9: x-velocity
4,10: y-velocity
5,11: w (related to spin rate)

for both scenarios, atol = rtol = 1e-10 '''


#print(max(allArrays[5]))
#print(max(allArrays[11]))
#exit()

maxTime = max(max(allArrays[0]),max(allArrays[6]))
def setupGraphs(paramNum, yLabel, graphTitle):
  figure, axes = plt.subplots()
  axes.plot(allArrays[0],allArrays[paramNum],"-c", label="Constant phi0")
  axes.plot(allArrays[6],allArrays[paramNum + 6],"-m", label="Variable phi0")
  minX = min(min(allArrays[paramNum]),min(allArrays[paramNum + 6]))
  maxX = max(max(allArrays[paramNum]),max(allArrays[paramNum + 6]))
  axes.set(xlabel="time (s)", ylabel=yLabel)
  axes.set(xlim=(0,maxTime), ylim=(minX,maxX), title=graphTitle)
  axes.legend()
  plt.show()

setupGraphs(1,"x-position(m)", "X-position Evolution")
setupGraphs(2,"y-position(m)", "Y-position Evolution")
setupGraphs(5,"omega * R (m/s)","Spin Rate Evolution")

figure, axes = plt.subplots()
axes.plot(allArrays[1],allArrays[2],"-c", label="Constant phi0")
axes.plot(allArrays[7],allArrays[8],"-m", label="Variable phi0")
minX = min(min(allArrays[1]),min(allArrays[7]))
maxX = max(max(allArrays[1]),max(allArrays[7]))
minY = min(min(allArrays[2]),min(allArrays[8]))
maxY = max(max(allArrays[2]),max(allArrays[8]))
axes.set(xlabel="X-position (m)", ylabel="Y-position (m)")
axes.set(xlim=(minX,maxX), ylim=(minY,maxY), title="Rock Trajectory")
axes.legend()
plt.show()
