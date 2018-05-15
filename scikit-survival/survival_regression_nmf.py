# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 13:18:34 2018

@author: mrade_000
"""

import pandas as pd
from sksurv.datasets import get_x_y
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.preprocessing import OneHotEncoder
import numpy as np
import pickle
import prince

#%%

def fit_and_score_features(X, y):
    n_features = X.shape[1]
    scores = np.empty(n_features)
    m = CoxPHSurvivalAnalysis(verbose=True, n_iter=10000)
    for j in range(n_features):
        Xj = X[:, j:j+1]
        m.fit(Xj, y)
        scores[j] = m.score(Xj, y)
    return scores

#%%


if __name__ == '__main__':

    df = pd.read_excel('BPE&INDdataForSurivivalModel.xlsx')

    df = df[df['Duration'] >= 12]

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

    from sklearn.decomposition import NMF

    model = NMF(n_components=8)
    data_x_numeric = model.fit_transform(data_x_numeric)

    #%%

    estimator = CoxPHSurvivalAnalysis(verbose=True, n_iter=10000)
    estimator.fit(data_x_numeric, y)
    #%%

    print()
    print(pd.Series(estimator.coef_, index=data_x_numeric.columns))
    print()

    print(estimator.score(data_x_numeric, y))
    print()

    scores = fit_and_score_features(data_x_numeric.values, y)
    print(pd.Series(scores, index=data_x_numeric.columns).sort_values(ascending=False))
    #%%

    from sklearn.feature_selection import SelectKBest
    from sklearn.pipeline import Pipeline

    pipe = Pipeline([('select', SelectKBest(fit_and_score_features, k=3)),
                     ('model', CoxPHSurvivalAnalysis(verbose=True, n_iter=10000))])

    from sklearn.model_selection import GridSearchCV

    param_grid = {'select__k': np.arange(1, data_x_numeric.shape[1] + 1)}
    gcv = GridSearchCV(pipe, param_grid, n_jobs=4, verbose=60)
    gcv.fit(data_x_numeric, y)

    results = pd.DataFrame(gcv.cv_results_).sort_values(by='mean_test_score', ascending=False)
    results.to_excel('CrossValidationMCA_8.xlsx', index=False)

    pipe.set_params(**gcv.best_params_)
    pipe.fit(data_x_numeric, y)

    encoder, transformer, final_estimator = [s[1] for s in pipe.steps]
    print(pd.Series(final_estimator.coef_, index=encoder.encoded_columns_[transformer.get_support()]))
    pickle.dump(final_estimator, open('RegressorMCA_8.pkl', 'wb'))
