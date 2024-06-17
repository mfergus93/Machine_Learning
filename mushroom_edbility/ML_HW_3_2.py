# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 19:53:02 2022

@author: Matt
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Feb 20 15:51:22 2022

@author: Matt
"""

from sklearn import tree
import pandas as pd
import pydot

df = pd.read_excel(r'C:\Users\Matt\Desktop\FlareData.xlsx')
cm=df.corr()
Feature=df[['Zurich ClassN','Spot SizeN','Spot DistN', 'Activity', 'Evolution', 'Prev Activity', 'Historical', 'New Historical', 'Area', 'Spot Area']].values.tolist()
Target=df['C class'].tolist()

clf=tree.DecisionTreeClassifier(criterion="gini")
clf=clf.fit(Feature, Target)

Fnames=['Zurich ClassN','Spot SizeN','Spot DistN', 'Activity', 'Evolution', 'Prev Activity', 'Historical', 'New Historical', 'Area', 'Spot Area']
Tnames='C Class =1', 'C Class=0'

dot_data=tree.export_graphviz(clf, out_file=None, feature_names=Fnames, class_names=Tnames, filled=True, rounded=True, special_characters=True, max_depth=4)

(graph,) =pydot.graph_from_dot_data(dot_data)
graph.write_png(r'C:\Users\Matt\Desktop\FlareData_Gini.png')