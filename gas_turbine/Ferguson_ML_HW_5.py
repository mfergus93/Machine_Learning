# -*- coding: utf-8 -*-
"""
Created on Fri Apr 15 12:30:44 2022

@author: Matt
"""
import pandas as pd
import numpy as np
import sklearn as sk
from sklearn import neural_network, metrics


filename = (r'C:\Users\Matt\Desktop\ccpp.xlsx')
odf= pd.read_excel(filename)

auroc=[]
accuracy=[]
layers=[]
nodes=[]
functions=[]
sf_check=[]

activation_functions=['identity','logistic', 'tanh', 'relu']

ar=np.array((odf-odf.min())/(odf.max()-odf.min()))

x=ar[:,1:5]
y=ar[:,-1]

(x_train, x_test, y_train, y_test) = sk.model_selection.train_test_split(x, y, test_size=0.3, random_state=22222)

for node1 in range(1,11): #Iterate nodes 1 to 10
    for node2 in range(11):
        for node3 in range(11):
            for function in activation_functions: 
                
                size=[node1,node2,node3]
                size=[node for node in size if node!=0]
                size_and_function=str(size)+function
                
                if size_and_function not in sf_check:
                    
                    clf = neural_network.MLPClassifier(hidden_layer_sizes=(size), activation=function, learning_rate=
                    'adaptive', tol=0.0001, solver='adam', alpha=0.0001, early_stopping=True, validation_fraction=0.1)
                    clf.fit(x_train, y_train)
                    y_pred=clf.predict(x_test)
                    
                    auroc.append(metrics.roc_auc_score(y_test, y_pred, multi_class='ovr'))
                    accuracy.append(metrics.accuracy_score(y_test, y_pred))
                    layers.append(len(size))
                    nodes.append(size)
                    functions.append(function)
                    
                    sf_check.append(str(size)+function)

                    print(size)








