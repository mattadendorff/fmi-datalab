import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.ensemble import IsolationForest
import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")

if __name__ == '__main__':
    number_of_tests = 100
    training_accuracy = np.zeros((number_of_tests, 1))
    smoker_accuracy = np.zeros((number_of_tests, 1))
    nonsmoker_accuracy = np.zeros((number_of_tests, 1))
    for i in range(number_of_tests):
        df = pd.read_csv('data/datalab_persona_cat.csv')

        df = pd.get_dummies(df, columns=[c for c in df.columns.values if c not in ['FKSmoker']])

        X_outliers = df[df['FKSmoker'] == 1]

        X_outliers.drop(['FKSmoker'], inplace=True, axis=1)

        X = df[df['FKSmoker'] == 0]

        X.drop(['FKSmoker'], inplace=True, axis=1)

        X_train = X.sample(frac=0.9)

        X_test = X.drop(df.index[list(X_train.index)])

        # fit the model
        clf = IsolationForest(max_samples=1000)
        clf.fit(X_train)
        y_pred_train = clf.predict(X_train)
        y_pred_test = clf.predict(X_test)
        y_pred_outliers = clf.predict(X_outliers)

        trainers = pd.Series(y_pred_train, name='verdict')

        training_accuracy[i] = len(trainers[trainers == 1]) / len(y_pred_train)

        inliers = pd.Series(y_pred_test, name='verdict')

        nonsmoker_accuracy[i] = len(inliers[inliers == 1]) / len(y_pred_test)

        outliers = pd.Series(y_pred_outliers, name='verdict')

        smoker_accuracy[i] = len(outliers[outliers == -1]) / len(y_pred_outliers)
        print(training_accuracy[i], nonsmoker_accuracy[i], smoker_accuracy[i])

    print(training_accuracy.mean(), nonsmoker_accuracy.mean(), smoker_accuracy.mean())

