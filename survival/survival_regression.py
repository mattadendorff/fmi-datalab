# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 13:18:34 2018

@author: mrade_000
"""

import pandas as pd
from lifelines import KaplanMeierFitter, AalenAdditiveFitter, CoxPHFitter
from pylab import show
import pickle

df = pd.read_excel('BPE&INDdataForSurivivalModel.xlsx')

df = df[df['Duration'] != 0]

df2 = df.loc[:, ['DISTRIBUTION CHANNEL', 'GENDER', 'SMOKER STATUS',
                'PremiumPattern', 'BENEFITS TYPE', 'BROKER COMM']]

#df2 = df.loc[:, ['GENDER', 'SMOKER STATUS', 'PremiumPattern']]

#df2 = df.loc[:, ['SMOKER STATUS', 'GENDER']]

df2 = pd.get_dummies(df2)

#T = df['Duration']

E = df['LapseIndicator'].apply(lambda x: True if x == 1 else False)

df2['E'] = E
df2['T'] = T

aaf = AalenAdditiveFitter()
aaf.fit(df2, 'T', event_col='E', show_progress=True)
pickle.dump(aaf, open('Smoker_Gender_All.pkl', 'wb'))
aaf.plot()

#cph = CoxPHFitter()
#cph.fit(df2, duration_col='T', event_col='E', show_progress=True, strata=['SMOKER STATUS_No','SMOKER STATUS_Yes',
#                                                                          'GENDER_F', 'GENDER_M'])
#pickle.dump(cph, open('Smoker_Gender_CPF.pkl', 'wb'))
#cph.plot()