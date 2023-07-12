'''Make a 3-d graph relating expected points to distance and first down'''

import numpy as n
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle

mainModel = pickle.load(open("mainmodel.dat","rb"))
fourthModel = pickle.load(open("4thmodel.dat","rb"))
KOModel = pickle.load(open("KOModel.dat","rb"))
decisionModel = pickle.load(open("4thChoiceModel.dat","rb"))

#this function exists so that the main function can be on top where it belongs
def runThings():
    plotDecisions()
    
    
#wasn't expecting this to end up being recursive but it makes sense, 
#i guess, with teams kicking back and forth until time runs out
def EP(probArray, time, sd):
    MTTS = 261 #mean time to score
    kickoffAdjustment = 0 if time <= MTTS else EP(KOModel.predict_proba([[35, time - MTTS, sd]])[0], time < MTTS, sd)
    return ((probArray[6] - probArray[0]) * (6.97 - kickoffAdjustment) + #our TD prob minus their TD prob, multiplied by TD value
            (probArray[5] - probArray[1]) * (3 - kickoffAdjustment) + #same but for FGs
            (probArray[4] - probArray[2]) * (2 + kickoffAdjustment)) #you receive the KO after a safety so add the KO adjustment

def EPFunc(down, toGoal, toFirst, time, sd, gtg):
    if (down == 4):
       return EP(fourthModel.predict_proba([[toGoal, toFirst, time, int(toGoal <= 40), sd, int(gtg)]])[0],time,sd)  
    else: 
       return EP(mainModel.predict_proba([[down,min(toFirst,toGoal),toGoal,time,sd,int(gtg)]])[0],time,sd)
def plotEP(down, sd=0, time=1200):
    z = n.zeros((100, 100))  # Initialize z with zeros. the first array axis is yardsToFirst, and the second is yardsToGoal
    x = n.linspace(1,100,num=100)
    y = n.linspace(1,25,num=100)
    x,y = n.meshgrid(x,y)
    for yardsToGoal in range(100):
        for yardsToFirst in range(100):
            index = yardsToFirst
            yardsToFirst = (yardsToFirst * (25/100) + 1)
            z[index][yardsToGoal] = EPFunc(down, yardsToGoal+1, min(yardsToFirst, yardsToGoal+1), time, sd, (yardsToFirst >= yardsToGoal))
    figure = plt.figure()
    axes = figure.add_subplot(projection="3d")
    axes.plot_surface(x,y,z,cmap="cool",rcount=100,ccount=100)
    axes.set(xlabel="Yards To Opponent's End Zone",ylabel="Yards to First Down",zlabel="Expected Points",
             title=("EP by Field Position and Distance to First Down, Possession team " + ("winning" if sd >= 0 else "losing")
             + " by " + str(abs(sd)) + " with " + str(time) + " seconds remaining in the half on down #" + str(down)),zlim=(-3,7),ylim=(0,25),xlim=(0,100))
    plt.subplots_adjust(top=0.97,right=1,left=0.0,bottom=0.0)
    plt.show()
def plotDecisions(sd=0,time=1200):
    goForIt = n.zeros((100, 100))  #the first array axis is yardsToFirst, and the second is yardsToGoal
    kickFG = n.zeros((100,100))
    punt = n.zeros((100,100))
    x = n.linspace(1,100,num=100)
    y = n.linspace(1,25,num=100)
    x,y = n.meshgrid(x,y)
    for yardsToGoal in range(100):
        for yardsToFirst in range(100):
            index = yardsToFirst
            yardsToFirst = (yardsToFirst / 4)
            punt[index][yardsToGoal], kickFG[index,yardsToGoal],goForIt[index][yardsToGoal] = decisionModel.predict_proba([[yardsToGoal, min(yardsToFirst, yardsToGoal),time, int(yardsToGoal <= 40), sd, int(yardsToFirst >= yardsToGoal)]])[0]
            #sets the value of all three arrays in the same command
            
    #make goForIt graph        
    figure = plt.figure()
    axes = figure.add_subplot(projection="3d")
    axes.plot_surface(x,y,goForIt,cmap="cool",rcount=100,ccount=100,label="Go for it Probability")
    axes.set(xlabel="Yards To Opponent's End Zone",ylabel="Yards to First Down",zlabel="Probabilities",
             title=("Probability of a team Going for it on 4th down, Possession team " + ("winning" if sd >= 0 else "losing")
             + " by " + str(abs(sd)) + " with " + str(time) + " seconds remaining in the half"),zlim=(0,1),ylim=(0,25),xlim=(0,100))
    plt.subplots_adjust(top=0.97,right=1,left=0,bottom=0)
    plt.show()
    
    #make field goal graph
    figure = plt.figure()
    axes = figure.add_subplot(projection="3d")
    axes.plot_surface(x,y,kickFG,cmap="winter",rcount=100,ccount=100,label="Field Goal Probability")
    axes.set(xlabel="Yards To Opponent's End Zone",ylabel="Yards to First Down",zlabel="Probabilities",
             title=("Probability of a team attempting a FG on 4th down, Possession team " + ("winning" if sd >= 0 else "losing")
             + " by " + str(abs(sd)) + " with " + str(time) + " seconds remaining in the half"),zlim=(0,1),ylim=(0,25),xlim=(0,100))
    plt.subplots_adjust(top=0.97,right=1,left=0,bottom=0)
    plt.show()

    figure = plt.figure()
    axes = figure.add_subplot(projection="3d")
    axes.plot_surface(x,y,punt,cmap="Wistia",rcount=100,ccount=100,label="Punt Probability")
    axes.set(xlabel="Yards To Opponent's End Zone",ylabel="Yards to First Down",zlabel="Probabilities",
             title=("Probability of a team Going for it on 4th down, Possession team " + ("winning" if sd >= 0 else "losing")
             + " by " + str(abs(sd)) + " with " + str(time) + " seconds remaining in the half"),zlim=(0,1),ylim=(0,25),xlim=(0,100))
    plt.subplots_adjust(top=0.97,right=1,left=0,bottom=0)
    plt.show()
runThings()