# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 14:43:57 2022

@author: Matt
"""

from sklearn import linear_model as linmod
import pandas as pd
import numpy as np
import sklearn as sk

import os

path=os.getcwd()

filename = (path+'\\BattingSalaries_Preprocessed.xlsx')
odf= pd.read_excel(filename)

One_Hot=pd.get_dummies(odf.teamID,prefix='Team')

seasons= odf['yearID'].unique().tolist()
mses=[]
R_Squared=[]
Size=[]

odf=odf.drop(['playerID','teamID','lgID','yearPlayer'], axis=1)
odf=pd.concat([One_Hot,odf],axis=1)

odf=(odf-odf.min())/(odf.max()-odf.min()) #normalize
seasons= odf['yearID'].unique().tolist()
# odf=odf.replace(np.nan,0) #Catch nans for range=0
   
for season in seasons:
    
    results=[]
    df=odf[odf['yearID']==season]
    
    train_target=df['Salary'].tolist()
    df=df.drop(['Salary'], axis=1)
    feature_names= df.columns
    train_features=df[feature_names].values.tolist()

    
    X = np.array(train_features)
    Y = np.array(train_target)
    
    (trainX, testX, trainY, testY) = sk.model_selection.train_test_split(X, Y, test_size=0.3, random_state=22222)
    
    salary_model = linmod.LinearRegression()
    salary_model.fit(trainX, trainY)
    
    R_Squared.append(salary_model.score(testX, testY))
    Size.append(len(df))
    print("W = ", salary_model.intercept_, salary_model.coef_, R_Squared)
    
    accuracy = salary_model.score(testX, testY)
    
    for row in testX: #predict the rain amount for each row in the test data

        prediction=salary_model.predict(row.reshape(1, -1))
        results.append(prediction[0])
    
    mses.append((1/len(results))*np.sum(((testY-np.array(results))**2)))
    
    # mses.append(sklmetrics.mean_squared_error(testY))