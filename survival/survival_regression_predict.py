# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 13:18:34 2018

@author: mrade_000
"""

import pandas as pd
from lifelines import KaplanMeierFitter, AalenAdditiveFitter, CoxPHFitter
from pylab import show
import pickle

#%%

df = pd.read_excel('BPE&INDdataForSurivivalModel.xlsx')

for m in [6, 12, 18, 24, 30, 36]:
    out = open('Regression_Test_LessThan_%s.pkl' % m, 'wb')

    df = df[df['Duration'] < m]
    
#    df.reset_index(inplace=True)
    
    df2 = df.loc[:, ['DISTRIBUTION CHANNEL', 'GENDER', 'SMOKER STATUS', 
                     'PremiumPattern', 'BENEFITS TYPE', 'BROKER COMM',
                     'DEBITORDERPERIOD', 'PREM % EARNINGS BAND']]
    
    for c in df2.columns.values:
            if c != 'AGE AT DOC':
                df2[c] = df2[c].astype('category')
    
    df2 = pd.get_dummies(df2)
    
    T = df['Duration']
    
    E = df['LapseIndicator'].apply(lambda x: True if x == 1 else False)
    
    df2['E'] = E
    df2['T'] = T
    
#    X_train = df2.sample(frac=0.9)
    
#    X_test = df2.drop(df2.index[list(X_train.index)])
    
#%%
    
    aaf = AalenAdditiveFitter()
    aaf.fit(df2, 'T', event_col='E', show_progress=True)
    pickle.dump(aaf, out)
    out.close()
#%%