# AML_HW_2
# Matthew Ferguson

import pandas as pd
import numpy as np
from sklearn import model_selection, neural_network, metrics, tree
import tensorflow as tf
import seaborn as sb
import matplotlib.pyplot as plt

# %%

df_cs=pd.read_excel(r'C:\Data\Census_Supplement.xlsx')
df_cs=(df_cs-df_cs.min())/(df_cs.max()-df_cs.min())
df_rd=pd.read_csv(r'C:\Data\RawData.csv')
# df_rd['F_BIN'] = df_rd['F_BIN'].astype(int)
# Additional Preprocessing to be performed as dfs read in if needed
df_rd=(df_rd-df_rd.min())/(df_rd.max()-df_rd.min())

x_cs=np.array(df_cs[['AGI','A_AGE','A_SEX','WKSWORK']])
y_cs=np.array(df_cs[['HDIVVAL']])
x_rd=np.array(df_rd[['S','T','U','V','X','Y','Z']])
y_rd=np.array(df_rd[['F_BIN']]).astype(np.int32)

(x_train, x_test, y_train, y_test) = model_selection.train_test_split(x_cs, y_cs,
test_size=0.3, random_state=22222)

(x_train_2, x_test_2, y_train_2, y_test_2) = model_selection.train_test_split(x_rd, y_rd,
test_size=0.3, random_state=22222)

# %%

epochs=100
alphas=[0,0.0001, 0.001, 0.01]

for node in range(3,7):
     for alph in alphas:

          loss_array=np.zeros((100,2))
          weight_array=np.zeros((epochs,5*node))
          annr=neural_network.MLPRegressor(random_state=22222,hidden_layer_sizes=(node), alpha=alph)

          for epoch in range(epochs):
               annr.partial_fit(x_train,y_train.ravel())
               loss_array[epoch,0]=(1-annr.score(x_train,y_train))
               loss_array[epoch,1]=(1-annr.score(x_test,y_test))

               trainmse = metrics.mean_squared_error(y_train, annr.predict(x_train))
               testmse = metrics.mean_squared_error(y_test, annr.predict(x_test))
               epoch_weights=np.concatenate((annr.coefs_[0].reshape(4*node),annr.coefs_[1].reshape(node)))
               weight_array[epoch]=epoch_weights

          ptitle='Nodes='+str(node)+', '+'Alpha='+str(alph)+' Performance'
          sb.lineplot(data=weight_array)
          plt.title(ptitle)
          plt.ylabel('Coefficient')
          plt.xlabel('Epoch')
          plt.savefig('C:\Data\Output\Weight_Plot'+str(node)+'_'+str(alph)+'.png')
          plt.clf()

          # plt.ylim(10**-12,1**2)
          # sb.lineplot(data=loss_array)
          fig, ax=plt.subplots()
          ax.plot(loss_array)
          plt.ylabel('Loss')
          plt.xlabel('Epoch')
          plt.legend(['train_loss','val_loss'])
          # plt.yscale('log')
          plt.title(ptitle)
          plt.yscale('log')

          plt.savefig('C:\Data\Output\Loss_Plot'+str(node)+'_'+str(alph)+'.png')
          plt.clf()

# %%

cm=df_rd.corr()
annc=neural_network.MLPClassifier(hidden_layer_sizes=(10,10,10),max_iter=100, activation='relu')
annc_model=annc.fit(x_train_2,y_train_2.ravel())
annc_score=annc.score(x_test_2,y_test_2.ravel())
annc_acc=model_selection.cross_val_score(annc_model, x_test_2, y_test_2.ravel(), cv=5)

clf=tree.DecisionTreeClassifier(max_depth=5, criterion='entropy')
tree_model=clf.fit(x_train_2,y_train_2)
tree_acc=model_selection.cross_val_score(tree_model, x_test_2, y_test_2, cv=5)
y_pred_tree=tree_model.predict(x_test_2)


#%%

inputwidth=x_train.shape[1]
accuracy_list=[]
for node1 in range(1,11):
     for node2 in range(1,5):
          model=tf.keras.models.Sequential([
               tf.keras.layers.InputLayer(input_shape=(inputwidth,)),
               tf.keras.layers.Dense(node1, activation='relu'),
               tf.keras.layers.Dense(node2, activation='relu'),
               tf.keras.layers.Dense(1, activation='sigmoid')])
          model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

          history=model.fit(x_train, y_train, batch_size=1000, epochs=100, validation_split=0.2, verbose=0)
          y_pred = model.predict(x_test)
          r2=metrics.r2_score(y_test,y_pred)
          accuracy_list.append([node1, node2, (history.history["val_accuracy"][-1])])
          # plt.plot(history.history['loss'])
          # plt.plot(history.history['val_loss'])
          # plt.title('Model Performance')
          # plt.ylabel('Error')
          # plt.xlabel('Epoch')
          # plt.legend(['loss','val_loss'], loc='upper left')
          # plt.savefig('C:\Data\Output\Keras_Plot'+str(node1)+'_'+'str(node2)'+'.png')
          # plt.clf()


#%%
inputwidth=x_train.shape[1]
accuracy_list=[]
model=tf.keras.models.Sequential([
tf.keras.layers.InputLayer(input_shape=(inputwidth,)),
tf.keras.layers.Dense(1, activation='relu'),
# tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history=model.fit(x_train, y_train, batch_size=1000, epochs=100, validation_split=0.2, verbose=0)

y_pred = model.predict(x_test)
r2=metrics.r2_score(y_test,y_pred)
accuracy_list.append([(history.history["val_accuracy"][-1])])
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Performance')
plt.ylabel('Error')
plt.xlabel('Epoch')
plt.legend(['loss','val_loss'], loc='upper left')
plt.savefig('C:\Data\Output\Keras_Plot'+str(node1)+'_'+'str(node2)'+'.png')
plt.clf()