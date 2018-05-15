# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 13:18:34 2018

@author: mrade_000
"""

import pandas as pd
from sksurv.datasets import get_x_y
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
from sksurv.preprocessing import OneHotEncoder
import numpy as np
import pickle

#%%

def fit_and_score_features(X, y):
    n_features = X.shape[1]
    scores = np.empty(n_features)
    m = GradientBoostingSurvivalAnalysis(verbose=True, n_estimators=500)
    for j in range(n_features):
        Xj = X[:, j:j+1]
        m.fit(Xj, y)
        scores[j] = m.score(Xj, y)
    return scores

#%%


if __name__ == '__main__':

    df = pd.read_excel('BPE&INDdataForSurivivalModel.xlsx')

    df = df[df['Duration'] != 12]

    df2 = df.loc[:, ['DISTRIBUTION CHANNEL', 'GENDER', 'SMOKER STATUS', 'AGE AT DOC',
                    'PremiumPattern', 'BENEFITS TYPE', 'BROKER COMM', 'DEBITORDERPERIOD', 'PREM % EARNINGS BAND']]

    T = df['Duration']

    E = df['LapseIndicator'].apply(lambda x: True if x == 1 else False)

    df2['E'] = E
    df2['T'] = T

    X, y = get_x_y(df2, ['E', 'T'], pos_label=True)

    for c in X.columns.values:
        if c != 'AGE AT DOC':
            X[c] = X[c].astype('category')

    data_x_numeric = OneHotEncoder().fit_transform(X)
    #%%

    estimator = GradientBoostingSurvivalAnalysis(verbose=True, n_estimators=500)
    estimator.fit(data_x_numeric, y)

    print(estimator.score(data_x_numeric, y))
    print()

    scores = fit_and_score_features(data_x_numeric.values, y)
    print(pd.Series(scores, index=data_x_numeric.columns).sort_values(ascending=False))

    pickle.dump(estimator, open('GradientRegressor.pkl', 'wb'))

    #%%

    from sklearn.feature_selection import SelectKBest
    from sklearn.pipeline import Pipeline

    pipe = Pipeline([('encode', OneHotEncoder()),
                     ('select', SelectKBest(fit_and_score_features, k=3)),
                     ('model', GradientBoostingSurvivalAnalysis(verbose=True, n_estimators=500))])

    from sklearn.model_selection import GridSearchCV

    param_grid = {'select__k': np.arange(1, data_x_numeric.shape[1] + 1)}
    gcv = GridSearchCV(pipe, param_grid, n_jobs=4, verbose=60)
    gcv.fit(X, y)

    results = pd.DataFrame(gcv.cv_results_).sort_values(by='mean_test_score', ascending=False)
    results.to_excel('CrossValidationGradient.xlsx', index=False)

