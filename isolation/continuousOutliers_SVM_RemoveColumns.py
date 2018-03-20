import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")

if __name__ == '__main__':
    number_of_tests = 50
    smoker_accuracy = np.zeros((number_of_tests, 1))
    nonsmoker_accuracy = np.zeros((number_of_tests, 1))

    for i in range(number_of_tests):
        X_outliers = pd.read_csv('data/fmi_smokers.csv')

        X_train = pd.read_csv('data/fmi_non_smokers.csv')
        print(i)

        # fit the model
        clf = OneClassSVM(gamma=0.3, kernel='rbf', cache_size=1000, nu=0.1)
        clf.fit(X_train)
        y_pred_train = clf.predict(X_train)
        y_pred_outliers = clf.predict(X_outliers)

        inliers = pd.Series(y_pred_train, name='verdict')

        nonsmoker_accuracy[i] = len(inliers[inliers == 1]) / len(y_pred_train)

        outliers = pd.Series(y_pred_outliers, name='verdict')

        smoker_accuracy[i] = len(outliers[outliers == -1]) / len(y_pred_outliers)

        print(nonsmoker_accuracy[i], smoker_accuracy[i])

    print(nonsmoker_accuracy.mean(), smoker_accuracy.mean())
