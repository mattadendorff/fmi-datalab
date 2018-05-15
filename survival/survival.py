# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 10:44:56 2018

@author: mrade_000
"""

import pandas as pd
from lifelines import KaplanMeierFitter
from pylab import show

df = pd.read_excel('lapse-data-pure.xlsx')
df = df[df['Duration Days'] != 0]

T = df['Duration Months'].apply(lambda x: 0 if x < 0 else x)
E = T.apply(lambda x: True if x > 0 else False)
df['Random Class'] = df['Random Class'].apply(lambda x: 'A' if x >= 0.5 else 'B')

#df.to_excel('lapse-data-ready.xlsx')

groups = df['Random Class']
ix = (groups == 'A')

kmf = KaplanMeierFitter()
kmf.fit(T[ix], event_observed=E[ix], label='Class A')  # or, more succiently, kmf.fit(T, E)

ax = kmf.plot()

ix = (groups == 'B')

kmf.fit(T[ix], event_observed=E[ix], label='Class B')

ax = kmf.plot(ax=ax)

show()