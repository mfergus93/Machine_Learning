"""
Created on Mon Feb 14 17:01:30 2022

@author: Matt
"""

import pandas as pd

filename = r'C:\Users\Matt\Desktop\Heart Disease.xlsx'
df = pd.read_excel(filename)
labels = df.columns
dqrlist=[]
dqr2=pd.DataFrame()

for label in labels:
                       
    col = df[label]
   
    cardinality=col.nunique()
    mode=(col.mode(dropna=True)).iloc[0]
    nmode=col[col==mode].count()
    missing = col.isnull().sum()
    zeroes = col[col==0].count()
   
    try:  
        mean = col.mean()
        median = col.median()
        nmedian=col[col==median].count()
        deviation = col.std()
        minimum = col.min()
        maximum = col.max()

    except:
        mean = 'nan'
        median = 'nan'
        deviation = 'nan'
        minimum = 'nan'
        maximum = 'nan'
       
    series1=pd.Series([cardinality, mean, median, nmedian, mode, nmode, deviation, minimum, maximum, zeroes, missing])
    dqrlist.append(series1)
    dqr2[label]=[series1]

dqr=pd.DataFrame(dqrlist)
covm=df.cov()
corm=df.corr()

dqr.columns=['cardinality', 'mean', 'median', 'nmedian', 'mode', 'nmode', 'deviation', 'minimum', 'maximum', 'zeroes', 'missing']
dqr.index=labels
dqr=dqr.T

dqr.to_excel(r'C:\Users\Matt\Desktop\DQR.xlsx')
covm.to_excel(r'C:\Users\Matt\Desktop\COV.xlsx')
corm.to_excel(r'C:\Users\Matt\Desktop\COR.xlsx')

