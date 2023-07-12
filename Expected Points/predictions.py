import numpy as n 
import pickle
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

mainModel = pickle.load(open("mainmodel.dat","rb"))
fourthModel = pickle.load(open("4thModel.dat","rb"))
clutchModel = pickle.load(open("final5.dat","rb"))
KOModel = pickle.load(open("KOModel.dat","rb"))
print("all models loaded!")

def testOn2018(wantPlots): 
  season18 = open("2018.csv","r",encoding="utf-8")
  plays18 = season18.read().splitlines()
  season18.close()
  
  NUMBER_OF_BUCKETS = 25 #increasing this number gives more granular probabilities but smaller samples
  
  ourTD = [{} for count in range(NUMBER_OF_BUCKETS)]
  ourFG = [{} for count in range(NUMBER_OF_BUCKETS)]
  noScore = [{} for count in range(NUMBER_OF_BUCKETS)]
  theirFG = [{} for count in range(NUMBER_OF_BUCKETS)]
  theirTD = [{} for count in range(NUMBER_OF_BUCKETS)]
  
  playCount = 0
  maxTDProb = 0
  while (playCount < len(plays18)):
        predictions = []
        try: 
          play = plays18[playCount].split(sep=",")
          nextScore = int(play[13])
          if (play[10] == "kickoff"): #play[10] is the play-type. if that's kickoff, use the kickoff model
            play[0] = int(play[0])
            play[1] = int(play[1])
            play[2] = int(play[9]) - int(play[8])  #score differential wrt the kicking team
            play = play[0:3]
            predictions = KOModel.predict_proba([play])[0]
            
          elif ((int(play[1]) <= 300) and (int(play[2]) == 2)): #time under 300 seconds and half == 2, use the late-game model
            play[2] = int(play[0]) #distance to goal
            play[0] = int(play[3]) #down
            play[3] = int(play[1]) #time
            play[1] = int(play[5]) #distance to first
            play[5] = int(play[4]) #is it goal-to-go? (1/0)
            play[4] = int(play[8]) - int(play[9]) #score differential (wrt the team with possession)
            play = play[0:6] 
            predictions = clutchModel.predict_proba([play])[0]
         
          elif (int(play[3]) == 4): #if down is 4th, use the 4th down model
            play[0] = int(play[0])
            play[2] = int(play[1])
            play[1] = int(play[5])
            play[3] = (play[0] <= 40)
            play[5] = int(play[4])
            play[4] = int(play[8]) - int(play[9])
            play = play[0:6]
            predictions = fourthModel.predict_proba([play])[0]
            
          else: #no special scenario, use the main model
            play[2] = int(play[0]) 
            play[0] = int(play[3]) 
            play[3] = int(play[1]) 
            play[1] = int(play[5]) 
            play[5] = int(play[4]) 
            play[4] = int(play[8]) - int(play[9]) #score differential (wrt the team with possession)    
            play = play[0:6]
            predictions = mainModel.predict_proba([play])[0]
        except ValueError:
          playCount += 1
          continue
        plays18[playCount] = play
        plays18[playCount].append(nextScore)        
        ourTD[int(predictions[6] * NUMBER_OF_BUCKETS)][playCount] = predictions[6]
        ourFG[int(predictions[5] * NUMBER_OF_BUCKETS)][playCount] = predictions[5]
        noScore[int(predictions[3] * NUMBER_OF_BUCKETS)][playCount] = predictions[3]
        theirFG[int(predictions[1] * NUMBER_OF_BUCKETS)][playCount] = predictions[1]
        theirTD[int(predictions[0] * NUMBER_OF_BUCKETS)][playCount] = predictions[0]     

        playCount += 1
  
  TDF = n.zeros((NUMBER_OF_BUCKETS),dtype=n.int32) 
  #count of times the next score was a Touchdown for, when predicted at certain probabilities
  FGF = n.zeros((NUMBER_OF_BUCKETS),dtype=n.int32)
  NS  = n.zeros((NUMBER_OF_BUCKETS),dtype=n.int32)
  FGA = n.zeros((NUMBER_OF_BUCKETS),dtype=n.int32)
  TDA = n.zeros((NUMBER_OF_BUCKETS),dtype=n.int32)
  
  for count in range(len(plays18)): #will probably error.
      for d in range(NUMBER_OF_BUCKETS):
          if ((count in ourTD[d]) and (plays18[count][-1] == +7)):
              TDF[d] += 1
          if ((count in ourFG[d]) and (plays18[count][-1] == +3)):
              FGF[d] += 1
          if ((count in noScore[d]) and (plays18[count][-1] ==  0)):
              NS[d] += 1
          if ((count in theirFG[d]) and (plays18[count][-1] == -3)):
              FGA[d] += 1
          if ((count in theirTD[d]) and (plays18[count][-1] == -7)):
              TDA[d] += 1
  
  #print the results 
  predErr = [0,0,0,0,0]
  
  TDoProbs = []
  FGoProbs = []
  NSoProbs = []
  FGAoProbs = []
  TDAoProbs = []
  
  for count in range(NUMBER_OF_BUCKETS):
      if (len(ourTD[count]) > 0):
        xProb = n.mean(list(ourTD[count].values()))
        oProb = TDF[count]/len(ourTD[count])
        TDoProbs.append(oProb)
        predErr[0] += abs(xProb - oProb) * len(ourTD[count]) * (16010/45693)
      if (len(ourFG[count]) > 0):
        xProb = n.mean(list(ourFG[count].values()))
        oProb = FGF[count]/len(ourFG[count])
        FGoProbs.append(oProb)
        predErr[1] += abs(xProb - oProb) * len(ourFG[count]) * (10984/45693)
      if (len(noScore[count]) > 0):
        xProb = n.mean(list(noScore[count].values()))
        oProb = NS[count]/len(noScore[count])
        NSoProbs.append(oProb)
        predErr[2] += abs(xProb - oProb) * len(noScore[count]) * (7323/45693)
      if (len(theirFG[count]) > 0):
        xProb = n.mean(list(theirFG[count].values()))
        oProb = FGA[count]/len(theirFG[count])
        FGAoProbs.append(oProb)
        predErr[3] += abs(xProb - oProb) * len(theirFG[count]) * (4231/45693)
      if (len(theirTD[count]) > 0):
        xProb = n.mean(list(theirTD[count].values()))
        oProb = TDA[count]/len(theirTD[count])
        TDAoProbs.append(oProb)
        predErr[4] += abs(xProb - oProb) * len(theirTD[count]) * (6985/45693)
  print("Prediction Errors:", predErr)
  print("Total Error:",sum(predErr))
  if (not wantPlots):
    return sum(predErr)
  scoreProbs = [TDoProbs,FGoProbs,NSoProbs,FGAoProbs,TDoProbs]
  scoreTypes = [ourTD,ourFG,noScore,theirFG,theirTD]
  scoreNames = ["Possession Team Touchdown","Possession Team Field Goal","No Score This Half","Defending Team Field Goal","Defending Team Touchdown"]
  for count in range(5): 
    figure, axes = plt.subplots()
    axes.scatter(n.linspace(0.5/NUMBER_OF_BUCKETS,(0.5/NUMBER_OF_BUCKETS + (1/NUMBER_OF_BUCKETS)*len(scoreProbs[count])),num=len(scoreProbs[count])),scoreProbs[count],color="black",
    s=[len(scoreTypes[count][i])/50 for i in range(len(scoreProbs[count]))])
    axes.plot(n.linspace(0.02,0.98),n.linspace(0.02,0.98),color="blue")
    axes.set(xlim=(0,1),ylim=(0,1),xlabel=("Expected Probability of " + scoreNames[count]),ylabel="Observed Probability")
    plt.show()
testOn2018(False)