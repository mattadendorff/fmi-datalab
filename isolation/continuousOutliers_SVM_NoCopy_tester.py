import pandas as pd
import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")

if __name__ == '__main__':

    X_cont_train = pd.read_csv('data/datalab_persona_cont_reduced.csv')

    X_cat_train = pd.read_csv('data/datalab_persona_cat_reduced.csv')

    X_cont_test = pd.read_csv('data/joined_test_cont.csv')

    X_cat_test = pd.read_csv('data/joined_test_cat.csv')

    # fit the SVM model
    svm = OneClassSVM(gamma=0.3, kernel='rbf', cache_size=1000, nu=0.1)
    svm.fit(X_cont_train)

    # fit the Isolation model
    forest = IsolationForest(max_samples=1000)
    forest.fit(X_cat_train)

    svm_out = svm.predict(X_cont_test)
    forest_out = forest.predict(X_cat_test)

    print(svm_out)
    print(forest_out)



