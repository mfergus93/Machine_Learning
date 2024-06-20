# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 13:02:11 2022

@author: Matt
"""

import pandas as pd
import numpy as np
import sklearn as sk
import os

path = os.getcwd()

filename = (path+'\\BattingSalaries_Preprocessed.xlsx')
df= pd.read_excel(filename)

df=df[df['yearID']==2016]
train_target=df['teamID'].tolist()

df=df.drop(['playerID','teamID','lgID','yearPlayer'], axis=1)
df=(df-df.min())/(df.max()-df.min()) #normalize

feature_names= ['G','AB','R','H','SH','2B','3B','HR','RBI','IBBrat','Hrat']
train_features=df[feature_names].values.tolist()


X = np.array(train_features)
Y = np.array(train_target)

(trainX, testX, trainY, testY) = sk.model_selection.train_test_split(X, Y, test_size=0.3, random_state=11111)

for k in range(1,16):
    for weighting in ['distance']:

        salary_model = sk.neighbors.KNeighborsClassifier(n_neighbors=k, weights=weighting)
        salary_model = salary_model.fit(trainX, trainY)
        accuracy = salary_model.score(testX, testY)
        
        print((k, weighting, accuracy))


