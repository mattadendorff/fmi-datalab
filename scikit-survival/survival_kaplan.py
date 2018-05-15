# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 13:18:34 2018

@author: mrade_000
"""

import pandas as pd
from sksurv.nonparametric import kaplan_meier_estimator
import matplotlib.pyplot as plt
import numpy as np


#%%

def find_nearest(array, val):
    idx = (np.abs(array-val)).argmin()
    return array[idx], idx


def age_bands(x):
    if x < 20:
        return 'Under 20'
    if x < 30 and x >= 20:
        return '20 - 30'
    if x < 40 and x >= 30:
        return '30 - 40'
    if x < 50 and x >= 40:
        return '40 - 50'
    if x < 60 and x >= 50:
        return '50 - 60'
    if x < 70 and x >= 60:
        return '60 - 70'
    if x >= 70:
        return 'Over 70'


#%%

if __name__ == '__main__':

    df = pd.read_excel('BPE&INDdataForSurivivalModel.xlsx')

    df = df[df['Duration'] != 0]

    df2 = df.loc[:, ['DISTRIBUTION CHANNEL', 'GENDER', 'SMOKER STATUS', 
                     'AGE AT DOC', 'PremiumPattern', 'BENEFITS TYPE', 
                     'BROKER COMM', 'DEBITORDERPERIOD', 'PREM % EARNINGS BAND']]

    df2['AGE BAND'] = df2['AGE AT DOC'].apply(age_bands)

    # print(df2['AGE BAND'])

    T = df['Duration']

    E = df['LapseIndicator'].apply(lambda x: True if x == 1 else False)

    df2['E'] = E
    df2['T'] = T

#%%

    # time, survival_prob = kaplan_meier_estimator(df2["E"], df2["T"])
    # plt.step(time, survival_prob, where="post")
    # plt.ylabel("est. probability of survival $\hat{S}(t)$")
    # plt.xlabel("time $t$")

#%%

    for column_name in ['DISTRIBUTION CHANNEL', 'GENDER', 'SMOKER STATUS', 'PremiumPattern',
                        'BENEFITS TYPE', 'BROKER COMM', 'DEBITORDERPERIOD', 'PREM % EARNINGS BAND', 'AGE BAND']:

        for value in df2[column_name].unique():
            mask = df2[column_name] == value
            time, survival_prob = kaplan_meier_estimator(df2["E"][mask], df2["T"][mask])
            p = plt.step(time, survival_prob, where="post",
                     label="%s (n = %d)" % (value, mask.sum()))

            nearest, ind = find_nearest(survival_prob, 0.5)

            if nearest < 0.6:
                plt.vlines(time[ind], 0.2, 1, colors=p[0].get_color())
                plt.annotate(time[ind], xy=(time[ind], nearest), xytext=(30, 20), textcoords='offset points', ha='right',
                             va='bottom', bbox=dict(boxstyle='round,pad=0.5', fc=p[0].get_color(), alpha=0.5),
                             arrowprops=dict(arrowstyle='->',  connectionstyle='arc3,rad=0'))

            else:
                plt.annotate('NA', xy=(time[ind], nearest), xytext=(30, 20), textcoords='offset points', ha='right'
                             , va='bottom', bbox=dict(boxstyle='round,pad=0.5', fc=p[0].get_color(), alpha=0.5),
                             arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

        plt.hlines(0.5, 0, 120, colors='0.5', linestyles='dashed')

        plt.ylabel("est. probability of survival $\hat{S}(t)$")
        plt.xlabel("time $t$")
        plt.legend(loc="best")
        plt.title(column_name.capitalize())

        plt.show()
