import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.ensemble import IsolationForest
import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")

if __name__ == '__main__':
    rng = np.random.RandomState(42)

    df = pd.read_csv('data/datalab_persona_cont.csv')

    cols = [x for x in df.columns.values if
            x not in ['Age Next at DOC', 'Height', 'Weight', 'Annual Salary', 'Travel %', 'FKSmoker']]

    df = pd.get_dummies(df, columns=cols)

    X_outliers = df[df['FKSmoker'] == 0]

    X_outliers.drop(['FKSmoker'], inplace=True, axis=1)

    X = df[df['FKSmoker'] == 1]

    X.drop(['FKSmoker'], inplace=True, axis=1)

    X_train = X.sample(frac=0.95)

    X_test = X.drop(df.index[list(X_train.index)])

    # fit the model
    clf = IsolationForest(max_samples=10, random_state=rng)
    clf.fit(X_train)
    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test)
    y_pred_outliers = clf.predict(X_outliers)

    outliers = pd.Series(y_pred_test, name='verdict')

    print(len(outliers[outliers == 1]))
    print(len(y_pred_test))

    outliers = pd.Series(y_pred_outliers, name='verdict')

    print(len(outliers[outliers == -1]))
    print(len(y_pred_outliers))

