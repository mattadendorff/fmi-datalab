# -*- coding: utf-8 -*-
"""
Created on Mon Mar 05 14:17:24 2018

@author: mrade_000
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost import plot_importance
from matplotlib import pyplot
from xgboost.sklearn import XGBClassifier
from sklearn import cross_validation, metrics
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.feature_selection import SelectFromModel

import matplotlib.pylab as plt
from matplotlib.pylab import rcParams

import warnings

def fxn():
    warnings.warn("deprecated", DeprecationWarning)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    # df = pd.read_csv('data/combined_stab_v1.06_datalab.csv')
    df = pd.read_csv('data/datalab_persona_run1_with_scale_id.csv')
    # split data into X and y

    print(df.columns.values)

    cols = ['Gender', 'Marital Status', 'Alcohol', 'FKEducation', 'Occ Class']

    X = pd.get_dummies(df[df.columns.values[:-1]], columns=cols)
    Y = df[df.columns.values[-1]]

    train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.3, random_state=27)

    test = test_x
    test_results = pd.DataFrame(data={'ID': test_x['ID'], 'FKSmoker': test_y})

    train = train_x
    train.loc[:, 'FKSmoker'] = train_y

    target = 'FKSmoker'
    IDcol = 'ID'

    def modelfit(alg, dtrain, dtest, predictors, useTrainCV=1, cv_folds=5, early_stopping_rounds=200):
        if useTrainCV:
            xgb_param = alg.get_xgb_params()
            xgtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain[target].values)
            cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
                              metrics='auc', early_stopping_rounds=early_stopping_rounds)
            alg.set_params(n_estimators=cvresult.shape[0])

        # Fit the algorithm on the data
        alg.fit(dtrain[predictors], dtrain['FKSmoker'], eval_metric='auc')

        # Predict training set:
        dtrain_predictions = alg.predict(dtrain[predictors])
        dtrain_predprob = alg.predict_proba(dtrain[predictors])[:, 1]

        dtest_predictions = alg.predict(dtest[predictors])

        # Print model report:
        print("\nModel Report")
        print("Accuracy : %.4g" % metrics.accuracy_score(dtrain['FKSmoker'].values, dtrain_predictions))
        print("AUC Score (Train): %f" % metrics.roc_auc_score(dtrain['FKSmoker'], dtrain_predprob))

        #     Predict on testing data:
        dtest.loc[:,'predprob'] = alg.predict_proba(dtest[predictors])[:, 1]
        results = test_results.merge(dtest[['ID', 'predprob']], on='ID')
        print('AUC Score (Test): %f' % metrics.roc_auc_score(results['FKSmoker'], results['predprob']))

        print("Confusion matrix \n", confusion_matrix(test_y, dtest_predictions))

        plot_importance(alg)
        pyplot.show()
        return alg


    predictors = [x for x in train.columns if x not in [target, IDcol]]
    xgb1 = XGBClassifier(
        learning_rate=0.1,
        n_estimators=100,
        max_depth=6,
        min_child_weight=1,
        gamma=0,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='binary:logistic',
        nthread=4,
        scale_pos_weight=1,
        seed=27)

    xgb1 = modelfit(xgb1, train, test, predictors)
