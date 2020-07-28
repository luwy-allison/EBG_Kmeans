## Data object

'''
attributes:
    .wide(List)(情境下行為總數完整資料): 依照情境整理成 1 subj 1 vector
    .ForK_pd(Dataframe): 行為比率，含na
    .ForK(List)(行為比率完整資料 含na): 去掉column name
    .ForK_dropna(List)(丟入kmeans用): 行為比率，去掉na，去掉subj column，
    .wide_dropna_pd(Dataframe)(貼labels用):
    .ForK_na_pd(Dataframe)
    .wide_na_pd(Dataframe)
    .label_pd(Dataframe)
    .label(List)(bar plot用)

'''
import math
import pandas as pd
import os
import csv
import numpy as np
import json

class Data:
    
    def __init__(self, condition="", startTrial=1, endTrial=100,subjList=[n for n in range(1,161)]):
        with open('allSubjData.txt') as json_file:
            allData = json.load(json_file)
        allData = json.loads(allData)
        # allData=allSubjData()  # /*this was comment out due to decode problem in reading csv file on server*/ 
        self.wide = [["subj","more.buy","more.no trade","more.sell","equal.buy","equal.no trade","equal.sell","less.buy","less.no trade","less.sell"]] 
        # create self.wide
        if condition == "delta price":
            conditionId = 2
            for subjNum in subjList:
                lastPrice = 0
                nowPrice = 0
                gain = [0,0,0]
                same = [0,0,0]
                loss = [0,0,0]
                for trial in range(startTrial, endTrial+1):
                    lastPrice = nowPrice
                    nowPrice = int(allData[subjNum-1][trial][2])
                    deltaPrice = nowPrice-lastPrice
                    if deltaPrice > 0:
                        if allData[subjNum-1][trial][6] == "buy":
                            gain[0]+=1
                        elif allData[subjNum-1][trial][6] == "no trade":
                            gain[1]+=1
                        elif allData[subjNum-1][trial][6] == "sell":
                            gain[2]+=1
                    if deltaPrice == 0:
                        if allData[subjNum-1][trial][6] == "buy":
                            same[0]+=1
                        elif allData[subjNum-1][trial][6] == "no trade":
                            same[1]+=1
                        elif allData[subjNum-1][trial][6] == "sell":
                            same[2]+=1
                    if deltaPrice<0:
                        if allData[subjNum-1][trial][6] == "buy":
                            loss[0]+=1
                        elif allData[subjNum-1][trial][6] == "no trade":
                            loss[1]+=1
                        elif allData[subjNum-1][trial][6] == "sell":
                            loss[2]+=1
                self.wide.append([subjNum,gain[0],gain[1],gain[2],same[0],same[1],same[2],loss[0],loss[1],loss[2]])
        elif condition == "delta asset":
            conditionId = 5
            for pairNum in subjList:
                p1Num = (pairNum-1)*2
                p2Num = (pairNum-1)*2+1
                p1lastAsset = 10000
                p1nowAsset = 10000
                p1gain = [0,0,0]
                p1same = [0,0,0]
                p1loss = [0,0,0]
                p1nowDeltaAsset = 0

                p2lastAsset = 10000
                p2nowAsset = 10000
                p2gain = [0,0,0]
                p2same = [0,0,0]
                p2loss = [0,0,0]
                p2nowDeltaAsset = 0

                for trial in range(startTrial,endTrial+1):
                    p1lastAsset = p1nowAsset
                    p1nowAsset = int(allData[p1Num][trial][5])
                    p2lastAsset = p2nowAsset
                    p2nowAsset = int(allData[p2Num][trial][5])

                    p1lastDeltaAsset = p1nowDeltaAsset
                    p1nowDeltaAsset = p1nowAsset-p2nowAsset
                    p2lastDeltaAsset = p2nowDeltaAsset
                    p2nowDeltaAsset = p2nowAsset-p1nowAsset

                    p1DeltaAsset = p1nowDeltaAsset-p1lastDeltaAsset
                    p2DeltaAsset = p2nowDeltaAsset-p2lastDeltaAsset

                    ###check if delta(i-j) == -delta(j-i)
                    if p1DeltaAsset != -p2DeltaAsset:
                        print("delta asset not match!")
                        break

                    if p1DeltaAsset > 0:
                        if allData[p1Num][trial][6] == "buy":
                            p1gain[0]+=1
                        elif allData[p1Num][trial][6] == "no trade":
                            p1gain[1]+=1
                        elif allData[p1Num][trial][6] == "sell":
                            p1gain[2]+=1

                    elif p1DeltaAsset == 0:
                        if allData[p1Num][trial][6] == "buy":
                            p1same[0]+=1
                        elif allData[p1Num][trial][6] == "no trade":
                            p1same[1]+=1
                        elif allData[p1Num][trial][6] == "sell":
                            p1same[2]+=1

                    elif p1DeltaAsset < 0:
                        if allData[p1Num][trial][6] == "buy":
                            p1loss[0]+=1
                        elif allData[p1Num][trial][6] == "no trade":
                            p1loss[1]+=1
                        elif allData[p1Num][trial][6] == "sell":
                            p1loss[2]+=1

                    if p2DeltaAsset > 0:
                        if allData[p2Num][trial][6] == "buy":
                            p2gain[0]+=1
                        elif allData[p2Num][trial][6] == "no trade":
                            p2gain[1]+=1
                        elif allData[p2Num][trial][6] == "sell":
                            p2gain[2]+=1                
                    elif p2DeltaAsset == 0:
                        if allData[p2Num][trial][6] == "buy":
                            p2same[0]+=1
                        elif allData[p2Num][trial][6] == "no trade":
                            p2same[1]+=1
                        elif allData[p2Num][trial][6] == "sell":
                            p2same[2]+=1           
                    elif p2DeltaAsset < 0:
                        if allData[p2Num][trial][6] == "buy":
                            p2loss[0]+=1
                        elif allData[p2Num][trial][6] == "no trade":
                            p2loss[1]+=1
                        elif allData[p2Num][trial][6] == "sell":
                            p2loss[2]+=1

                self.wide.append([p1Num+1,p1gain[0],p1gain[1],p1gain[2],p1same[0],p1same[1],p1same[2],p1loss[0],p1loss[1],p1loss[2]])
                self.wide.append([p2Num+1,p2gain[0],p2gain[1],p2gain[2],p2same[0],p2same[1],p2same[2],p2loss[0],p2loss[1],p2loss[2]])
     
        elif condition == "delta cash":
            conditionId = 3
            for subjNum in subjList:
                lastCash = 0
                nowCash = 0
                gain = [0,0,0]
                same = [0,0,0]
                loss = [0,0,0]
                for trial in range(startTrial,endTrial+1):
                    lastCash = nowCash
                    nowCash = int(allData[subjNum-1][trial][3])
                    deltaCash = nowCash-lastCash

                    if deltaCash > 0:
                        if allData[subjNum-1][trial][6]=="buy":
                            gain[0]+=1
                        elif allData[subjNum-1][trial][6]=="no trade":
                            gain[1]+=1
                        elif allData[subjNum-1][trial][6]=="sell":
                            gain[2]+=1
                    if deltaCash == 0:
                        if allData[subjNum-1][trial][6]=="buy":
                            same[0]+=1
                        elif allData[subjNum-1][trial][6]=="no trade":
                            same[1]+=1
                        elif allData[subjNum-1][trial][6]=="sell":
                            same[2]+=1
                    if deltaCash < 0:
                        if allData[subjNum]-1[trial][6]=="buy":
                            loss[0]+=1
                        elif allData[subjNum-1][trial][6]=="no trade":
                            loss[1]+=1
                        elif allData[subjNum-1][trial][6]=="sell":
                            loss[2]+=1
                self.wide.append([subjNum,gain[0],gain[1],gain[2],same[0],same[1],same[2],loss[0],loss[1],loss[2]])
        else:
            print("Condition not support!")
        #
        self.ForK_pd = pd.DataFrame(self.wide[1:],columns=self.wide[0])
        for i in self.ForK_pd.index:
            decisionCount=[sum(self.ForK_pd.loc[i,"more.buy":"more.sell"]),sum(self.ForK_pd.loc[i,"equal.buy":"equal.sell"]),sum(self.ForK_pd.loc[i,"less.buy":"less.sell"])]
            for j in self.ForK_pd.columns[1:4]:
                if decisionCount[0]==0:
                    self.ForK_pd.loc[i,j]=pd.NA
                else:
                    self.ForK_pd.loc[i,j]/=decisionCount[0]
            for j in self.ForK_pd.columns[4:7]:
                if decisionCount[1]==0:
                    self.ForK_pd.loc[i,j]=pd.NA
                else:
                    self.ForK_pd.loc[i,j]/=decisionCount[1]
            for j in self.ForK_pd.columns[7:10]:
                if decisionCount[2]==0:
                    self.ForK_pd.loc[i,j]=pd.NA
                else:
                    self.ForK_pd.loc[i,j]/=decisionCount[2]
                    
        self.ForK_pd.columns=["subj","more.buy.ratio","more.no trade.ratio","more.sell.ratio","equal.buy.ratio","equal.no trade.ratio","equal.sell.ratio","less.buy.ratio","less.no trade.ratio","less.sell.ratio"]
        self.wide_pd=pd.DataFrame(self.wide[1:],columns=self.wide[0]).join(self.ForK_pd.set_index('subj'),on='subj')
        self.ForK_dropna=self.ForK_pd.dropna().drop(labels=['subj'],axis=1).values.tolist()
        self.ForK_na_pd=self.ForK_pd[self.ForK_pd.isnull().any(axis=1)]
        self.wide_dropna_pd=self.wide_pd.dropna()
        self.wide_na_pd=self.wide_pd[self.wide_pd.isnull().any(axis=1)]
        self.ForK=self.ForK_pd.values.tolist()

                
# def allSubjData():
    # subjDataList = []
    # path = "./CSV format data/"
    # fileList = os.listdir(path)
    # readP = []
    # for i in range(0,len(fileList)):
        # readP.append(path+fileList[i])
    # for f in readP[0:160]:
        # content = []
        # with open(f,"r") as csvfile:
            # rows = csv.reader(csvfile)
            # for row in rows:
                # content.append(row)
        # subjDataList.append(content)
    # return subjDataList