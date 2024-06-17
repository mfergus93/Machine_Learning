#AML HW 3
#Matthew Ferguson

import pandas as pd
import sklearn as sk
from sklearn import feature_selection, model_selection
import numpy as np

df=pd.read_csv(r'C:\Data\pbp-2020.csv')
df=df.drop(['DefenseTeam','IsChallenge','IsChallengeReversed','Challenger','IsMeasurement',
           'IsInterception','IsFumble','IsPenalty','IsTwoPointConversion','IsTwoPointConversionSuccessful',
           'RushDirection','YardLineFixed','YardLineDirection','IsPenaltyAccepted','PenaltyTeam','IsNoPlay',
           'PenaltyType','PenaltyYards', 'Description'], 1)
df=df.loc[df['PlayType']=='PASS']

d1=pd.get_dummies(df['OffenseTeam']).astype(int)
d2=pd.get_dummies(df['Formation']).astype(int)
d3=pd.get_dummies(df['PlayType']).astype(int)
d4=pd.get_dummies(df['PassType']).astype(int)

df=pd.concat([df,d1,d2,d3,d4],1)
df=df.drop(['OffenseTeam','Formation','PlayType','PassType'], 1)
df=df.drop(['NextScore','TeamWin','SeasonYear','IsRush','IsPass','IsSack','PASS','GameDate','GameId'], 1)

df=(df-df.min())/(df.max()-df.min())
cm=df.corr()

y=np.array(df['IsIncomplete'])
x=np.array(df.drop(['IsIncomplete'], 1))

(x_train, x_test, y_train, y_test) = model_selection.train_test_split(x, y, test_size=0.3, random_state=2)
model_1=sk.linear_model.LogisticRegression(max_iter=1000)
model_1.fit(x_train,y_train)
m1_acc=model_1.score(x_test,y_test)

model_2=sk.linear_model.LogisticRegression(max_iter=1000)
sfs=feature_selection.SequentialFeatureSelector(estimator=model_2,n_features_to_select='auto'
                                                ,tol=float(0.0001), direction='forward')
sfs.fit(x,y)
feature_bool=sfs.get_support()
x_new=sfs.transform(x)
(x_train_2, x_test_2, y_train_2, y_test_2) = model_selection.train_test_split(x_new, y, test_size=0.3, random_state=2)
model_2.fit(x_train_2,y_train_2)
m2_acc=model_2.score(x_test_2,y_test_2)