#AML Homework 1
import pandas as pd
from sklearn import tree, ensemble, model_selection, metrics
import pydot
import os

path=os.getcwd()

df=pd.read_excel(path+'\\Census_Supplement.xlsx')
df=(df-df.min())/(df.max()-df.min())

features_cont=df[['AGI','A_AGE','WKSWORK', 'A_SEX']].values.tolist()
features_bin=df[['AGI_BIN','A_AGE_BIN','WKSWORK_BIN', 'A_SEX']].values.tolist()
features='features_bin','features_cont'
target=df['HAS_DIV'].values.tolist()

fnames=['Income','Age','Weeks Worked', 'Sex']
tnames='Dividend'

depths=[10,5,4,3]
criteria = 'gini', 'entropy'

for crit in criteria:
      for depth in depths:

          (x_train_bin, x_test_bin, y_train_bin, y_test_bin) = model_selection.train_test_split(features_bin, target, test_size=0.3, random_state=1)
          clf_bin=tree.DecisionTreeClassifier(max_depth=depth, criterion=crit)
          bin_tree=clf_bin.fit(x_train_bin,y_train_bin)
          acc_1=model_selection.cross_val_score(bin_tree, x_test_bin, y_test_bin, cv=5)
          y_pred_bin=bin_tree.predict(x_test_bin)
          conf_1=metrics.confusion_matrix(y_test_bin, y_pred_bin)
          print(crit, depth, conf_1)
          # print(crit, depth, 'binary', acc_1)

          if depth==3 and crit=='entropy':
                dot_data=tree.export_graphviz(bin_tree, out_file=None, feature_names=fnames, class_names=tnames, filled=True, rounded=True, special_characters=True)
                (graph,) =pydot.graph_from_dot_data(dot_data)
                graph.write_png(path+'\\output\\Dividend_Tree_Binary.png')

          (x_train_cont, x_test_cont, y_train_cont, y_test_cont) = model_selection.train_test_split(features_cont, target, test_size=0.3, random_state=22222)
          clf_cont=tree.DecisionTreeClassifier(max_depth=depth, criterion=crit)
          cont_tree=clf_cont.fit(x_train_cont,y_train_cont)
          acc_2=model_selection.cross_val_score(cont_tree,x_test_cont, y_test_cont, cv=5)
          y_pred_cont=cont_tree.predict(x_test_cont)
          conf_2=metrics.confusion_matrix(y_test_cont,y_pred_cont)
          print(crit, depth, conf_2)

          if depth==3 and crit=='entropy':
                dot_data=tree.export_graphviz(cont_tree, out_file=None, feature_names=fnames, class_names=tnames, filled=True, rounded=True, special_characters=True)
                (graph,) =pydot.graph_from_dot_data(dot_data)
                graph.write_png(path+'\\output\\Dividend_Tree.png')


estimators=[10,100,1000]
for crit in criteria:
      for estimator in estimators:
          for depth in depths:
                    (x_train, x_test, y_train, y_test) = model_selection.train_test_split(features_cont, target, test_size=0.3, random_state=22222)
                    rf=ensemble.RandomForestClassifier(max_depth=depth, criterion=crit, n_estimators=estimator, bootstrap=False)
                    forest=rf.fit(x_train,y_train)
                    acc_3=model_selection.cross_val_score(forest,x_test,y_test,cv=5)
                    y_pred=forest.predict(x_test)
                    conf_3=metrics.confusion_matrix(y_test,y_pred)
                    print(crit, depth, estimator, acc_3)
