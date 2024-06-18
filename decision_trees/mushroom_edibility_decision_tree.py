# -*- coding: utf-8 -*-
"""
Created on Sun Feb 20 15:51:22 2022

@author: Matt
"""

from sklearn import tree
import pandas as pd
import pydot
import os

path=os.getcwd()

df = pd.read_excel(path+'\\AlienMushrooms.xlsx')
cm=df.corr()
Feature=df[['White','Tall','Frilly']].values.tolist()
df['Edible'] = df['Edible'].map({'T': 1, 'F': 0})
Target=df['Edible'].tolist()

clf=tree.DecisionTreeClassifier(criterion="entropy")
clf=clf.fit(Feature, Target)

Fnames=['White','Tall','Frilly']
Tnames='Inedible', 'Edible'

dot_data=tree.export_graphviz(clf, out_file=None, feature_names=Fnames, class_names=Tnames, filled=True, rounded=True, special_characters=True)

(graph,) =pydot.graph_from_dot_data(dot_data)
graph.write_png(path+'\\output\\AlienMushrooms.png')