# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 19:53:02 2022

@author: Matt
"""

from sklearn import tree
import pandas as pd
import pydot
import os

path=os.getcwd()

df = pd.read_excel(path+'\\FlareData.xlsx')
cm=df.corr()
df[['Zurich Class', 'Spot Size', 'Spot Dist']] = df[['Zurich Class', 'Spot Size', 'Spot Dist']].apply(lambda x: pd.factorize(x)[0])
Feature=df[['Zurich Class','Spot Size','Spot Dist', 'Activity', 'Evolution', 'Prev Activity', 'Historical', 'New Historical', 'Area', 'Spot Area']].values.tolist()
Target=df['C class'].tolist()

clf=tree.DecisionTreeClassifier(criterion="gini")
clf=clf.fit(Feature, Target)

Fnames=['Zurich Class','Spot Size','Spot Dist', 'Activity', 'Evolution', 'Prev Activity', 'Historical', 'New Historical', 'Area', 'Spot Area']
Tnames='C Class =1', 'C Class=0'

dot_data=tree.export_graphviz(clf, out_file=None, feature_names=Fnames, class_names=Tnames, filled=True, rounded=True, special_characters=True, max_depth=4)

(graph,) =pydot.graph_from_dot_data(dot_data)
graph.write_png(path+'\\output\\FlareData_Gini.png')