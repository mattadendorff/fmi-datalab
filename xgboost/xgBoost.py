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
from sklearn.feature_selection import SelectFromModel

import matplotlib.pylab as plt
from matplotlib.pylab import rcParams

rcParams['figure.figsize'] = 12, 4

df = pd.read_csv('data/train_modified.csv')

train_x, test_x, train_y, test_y = train_test_split(df[df.columns.values[1:]], df['Disbursed'], train_size=0.7,
                                                    test_size=0.3)

print(test_x.columns)

test = test_x
test_results = pd.DataFrame(data={'ID': test_x['ID'], 'Disbursed': test_y})

train = train_x
train['Disbursed'] = train_y

target = 'Disbursed'
IDcol = 'ID'
print(train['Disbursed'].value_counts())
print(test_y.value_counts())


def modelfit(alg, dtrain, dtest, predictors, useTrainCV=1, cv_folds=5, early_stopping_rounds=50):
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain[target].values)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
                          metrics='auc', early_stopping_rounds=early_stopping_rounds)
        alg.set_params(n_estimators=cvresult.shape[0])

    # Fit the algorithm on the data
    alg.fit(dtrain[predictors], dtrain['Disbursed'], eval_metric='auc')

    # Predict training set:
    dtrain_predictions = alg.predict(dtrain[predictors])
    dtrain_predprob = alg.predict_proba(dtrain[predictors])[:, 1]

    # Print model report:
    print("\nModel Report")
    print("Accuracy : %.4g" % metrics.accuracy_score(dtrain['Disbursed'].values, dtrain_predictions))
    print("AUC Score (Train): %f" % metrics.roc_auc_score(dtrain['Disbursed'], dtrain_predprob))

    #     Predict on testing data:
    dtest['predprob'] = alg.predict_proba(dtest[predictors])[:, 1]
    results = test_results.merge(dtest[['ID', 'predprob']], on='ID')
    print('AUC Score (Test): %f' % metrics.roc_auc_score(results['Disbursed'], results['predprob']))

    plot_importance(alg)
    pyplot.show()
    return alg


predictors = [x for x in train.columns if x not in [target, IDcol]]
xgb1 = XGBClassifier(
    learning_rate=0.1,
    n_estimators=1000,
    max_depth=5,
    min_child_weight=1,
    gamma=0,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='binary:logistic',
    nthread=4,
    scale_pos_weight=1,
    seed=27)

xgb1 = modelfit(xgb1, train, test, predictors)

thresholds = np.sort(xgb1.feature_importances_)

print('Test 1 done')

# Grid seach on subsample and max_features
# Choose all predictors except target & IDcols
param_test1 = {
    'max_depth': range(3, 10, 2),
    'min_child_weight': range(1, 6, 2)
}
gsearch1 = GridSearchCV(estimator=XGBClassifier(learning_rate=0.1, n_estimators=140, max_depth=5,
                                                min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
                                                objective='binary:logistic', nthread=4, scale_pos_weight=1, seed=27),
                        param_grid=param_test1, scoring='roc_auc', n_jobs=4, iid=False, cv=5)
gsearch1.fit(train[predictors], train[target])

print('Grid search results:')
print(gsearch1.cv_results_, gsearch1.best_params_, gsearch1.best_score_)

print('Test 2 done')
