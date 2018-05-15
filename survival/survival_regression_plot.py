# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 13:18:34 2018

@author: mrade_000
"""

import pandas as pd
from lifelines import KaplanMeierFitter, AalenAdditiveFitter, CoxPHFitter
import matplotlib.pyplot as plt
import pickle

df = pd.read_excel('BPE&INDdataForSurivivalModel.xlsx')

df = df[df['Duration'] != 0]

#df = df.loc[:, ['AGE AT DOC', 'DISTRIBUTION CHANNEL', 'GENDER', 'SMOKER STATUS',
#                'PremiumPattern', 'BENEFITS TYPE', 'BROKER COMM', 'LapseDuration']]

df2 = df.loc[:, ['GENDER', 'SMOKER STATUS', 'PremiumPattern']]

df2 = pd.get_dummies(df2)

T = df['Duration']

E = df['LapseIndicator'].apply(lambda x: True if x == 1 else False)

df2['E'] = E
df2['T'] = T

indices_smoke = df2['SMOKER STATUS_Yes'] == 1
indices_nonsmoke = df2['SMOKER STATUS_No'] == 1
indices_male = df2['GENDER_M'] == 1
indices_female = df2['GENDER_F'] == 1

aaf = pickle.load(open('Smoker_Gender_Premium.pkl', 'rb'))
aaf.plot()

survival = aaf.predict_survival_function(df2)
expect = aaf.predict_expectation(df2)

p_smokers = expect.loc[indices_smoke].median()
p_nonsmokers = expect.loc[indices_nonsmoke].median()
p_male = expect.loc[indices_male].median()
p_female = expect.loc[indices_female].median()
p_base = expect.median()

smokers = survival.loc[:, indices_smoke]
nonsmokers = survival.loc[:, indices_nonsmoke]
male = survival.loc[:, indices_male]
female = survival.loc[:, indices_female]

plt.figure()
plt.plot(smokers.mean(axis=1), c='#FF7F0E', linewidth=2, label='Smoker %.1f' % p_smokers)
plt.plot(nonsmokers.mean(axis=1), c='#1F77B4', linewidth=2, label='Non Smoker %.1f' % p_nonsmokers)
plt.plot(male.mean(axis=1), c='#D62728', linewidth=2, label='Male %.1f' % p_male)
plt.plot(female.mean(axis=1), c='#2CA02C', linewidth=2, label='Female %.1f' % p_female)
plt.plot(survival.mean(axis=1), '--', c='#9467BD', linewidth=3, label='Baseline %.1f' % p_base)
plt.legend()
plt.show()