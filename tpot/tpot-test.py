from tpot import TPOTClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")

if __name__ == '__main__':

    df = pd.read_csv('data/datalab_persona_run1_with_scale.csv')

    target = df['FKSmoker'].values

    df.drop(['FKSmoker'], inplace=True, axis=1)

    cols = [x for x in df.columns.values if
            x not in ['Age Next at DOC', 'Height', 'Weight', 'Annual Salary', 'Travel %']]

    df = pd.get_dummies(df, columns=cols)

    data = df.values

    X_train, X_test, y_train, y_test = train_test_split(data,
        target, train_size=0.75, test_size=0.25)

    tpot = TPOTClassifier(generations=100, population_size=100, verbosity=2, n_jobs=2, config_dict='TPOT sparse', scoring='balanced_accuracy')
    tpot.fit(X_train, y_train)
    print(tpot.score(X_test, y_test))
    tpot.export('tpot_fmi_pipeline_sparse_100_100_full.py')
