#script for the expected points modelling, using scikit-learn. makes and tests the model
import numpy as n
import pickle
from sklearn.linear_model import LogisticRegression

#pxp = open("2009New.csv","r",encoding="utf-8")
pxp = open("biggerdata.csv","r",encoding="utf-8")

#NUM_OF_PLAYS = 39142 #where down is non-NaN (2009)
#NUM_OF_PLAYS = 237594 #(2010-2015)
NUM_OF_PLAYS = 236743 #2009 - 2015, excl 2012
MAX_SD = 27 #plays with higher score diff are weighted zero. affects scaling for weights of other scores as well. (when weighting is enabled)
WEIGHTING_ENABLED = False
NEED_NEW_MODEL = True
plays = n.empty((NUM_OF_PLAYS,5),dtype=n.int16) #time can exceed 255
scores = n.empty((NUM_OF_PLAYS), dtype=n.int8)
weights = n.ones((NUM_OF_PLAYS),dtype=n.single)
count = 0
for line in pxp:
    line = line.split(sep=",")
    try:
       plays[count][0] = int(line[3]) #down
       plays[count][1] = (int(line[5])) #distance to first (natural log)
       plays[count][2] = int(line[0]) #distance to end zone
       plays[count][3] = int(line[1]) #time remaining in half
       plays[count][4] = int(line[4]) #is the situation goal-to-go
       sd = abs(int(line[8]) - int(line[9])) #score differeential (absolute)
       #weight is geo-mean of (0.55 ** number of drives to next score, 1 - differential/27)
       #but weight is set to zero if the differential is >= 27 since those plays don't matter
       if (WEIGHTING_ENABLED): 
           weights[count] = n.sqrt((0.55 ** (int(line[14]) // 2)) * (1 - sd/MAX_SD)) if (sd < MAX_SD) else 0
       scores[count] = int(line[13])#next score type
       count += 1 #if it gets partway through and errors out, this won't trigger so partially written data can be properly overwritten
    except ValueError:
        pass 
        #value errors will happen if data is missing (as it usually is for penalty or eoq)
        #we just ignore these and don't include them in the dataset
print(count)
pxp.close()

if (NEED_NEW_MODEL):
    lr = LogisticRegression(max_iter=10000, multi_class="multinomial", n_jobs=1).fit(plays,scores, sample_weight=weights)
    pickle.dump(lr, open("mlmodel.dat","wb"), protocol=4) #opens file and stores the object there
    print("done fitting model!")
else: 
    lr = pickle.load(open("mlmodel.dat","rb"))
    print("opened the pickle jar")

'''
Code to plot the various probabilities for first and 10 or first and goal 
all the way down the field based on the fitted model. 
oppTDProb = []
oppFGProb = []
oppSafetyProb = []
scorelessProb = []
safetyProb = []
FGProb = []
TDProb = []
for yardsToEZ in range(1,100):
    predictions = lr.predict_proba([[1,min(10,yardsToEZ),yardsToEZ, 900]])
    oppTDProb.append(predictions[0][0])
    oppFGProb.append(predictions[0][1])
    oppSafetyProb.append(predictions[0][2])
    scorelessProb.append(predictions[0][3])
    safetyProb.append(predictions[0][4])
    FGProb.append(predictions[0][5])
    TDProb.append(predictions[0][6])

import matplotlib.pyplot as plt
figure, axes = plt.subplots()
axes.plot(range(1,100),oppTDProb,("#dd1317")) #red
axes.plot(range(1,100),oppFGProb,("#dd7513")) #orange
axes.plot(range(1,100),oppSafetyProb,("#ddda13")) #yellow
axes.plot(range(1,100),scorelessProb,("#000000")) #black
axes.plot(range(1,100),safetyProb,("#2fdae0")) #teal
axes.plot(range(1,100),FGProb,("#0a3a0c")) #dark green
axes.plot(range(1,100),TDProb,("#05fd09")) #bright green
axes.set(xlim=(0,100), ylim=(0,1),ylabel="Probability of Score", xlabel="Yards from Endzone")
plt.show()'''

'''Testing on the excluded 2012 season'''
season12 = open("2012Results.csv","r",encoding="utf-8")
plays12 = season12.read().splitlines()
season12.close()

ourTD = [[] for count in range(20)]
ourFG = [[] for count in range(20)]
noScore = [[] for count in range(20)]
theirFG = [[] for count in range(20)]
theirTD = [[] for count in range(20)]

playCount = 0
maxTDProb = 0
while (playCount < len(plays12)):
    try:
        play = plays12[playCount].split(sep=",")
        play[2] = int(play[0]) #distance to goal
        play[0] = int(play[3]) #down
        play[3] = int(play[1]) #time
        play[1] = int(play[5]) #distance
        play[4] = int(play[4]) #is it goal-to-go? (1/0)
        #must be in this exact order or else it may be overwritten
        play[5] = int(play[13]) #next score. not used for prediction (duh), only for testing
        plays12[playCount] = play[0:6] 
        #will not be written if parsing errored out. total array should be usable though, and elements not relevant to the model are thrown out
        #make it so future uses of the 2012 plays (ie when testing) don't have to re-parse the play text
        predictions = lr.predict_proba([[play[0],play[1],play[2],play[3],play[4]]])[0] 
      
        #give probabilities for all plays in 2012
        '''each of these arrays representing scores has 20 lists as their elements. 
        each one of those lists represents a 5% bucket, with ourTD[0] holding the numbers of all plays 
        with less than a 5% chance of resulting in the possession team scoring a TD, etc. Safeties
        are almost always a <5% chance so they are excluded from probability analysis'''
        ourTD[int(predictions[6] * 20)].append(playCount)
        ourFG[int(predictions[5] * 20)].append(playCount)
        noScore[int(predictions[3] * 20)].append(playCount)
        theirFG[int(predictions[1] * 20)].append(playCount)
        theirTD[int(predictions[0] * 20)].append(playCount)     
    except ValueError:
        pass #ignore plays that don't have all required info (usually penalties/game breaks)
    playCount += 1

TDF = n.zeros((20),dtype=n.int32) 
#count of times the next score was a Touchdown for, when predicted at certain probabilities
FGF = n.zeros((20),dtype=n.int32)
NS  = n.zeros((20),dtype=n.int32)
FGA = n.zeros((20),dtype=n.int32)
TDA = n.zeros((20),dtype=n.int32)

for count in range(len(plays12)): #will probably error.
    for d in range(20):
        if ((count in ourTD[d]) and (plays12[count][5] == +7)):
            TDF[d] += 1
        if ((count in ourFG[d]) and (plays12[count][5] == +3)):
            FGF[d] += 1
        if ((count in noScore[d]) and (plays12[count][5] ==  0)):
            NS[d] += 1
        if ((count in theirFG[d]) and (plays12[count][5] == -3)):
            FGA[d] += 1
        if ((count in theirTD[d]) and (plays12[count][5] == -7)):
            TDA[d] += 1

#print the results 
for count in range(20):
    print("Number of Touchdowns For in this Bucket:", TDF[count], 
          "Number of Plays in this Bucket:",len(ourTD[count]), 
          "Actual Probability:",TDF[count]/len(ourTD[count]))
    
    
    
    
    
    
    




