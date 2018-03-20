from tpot import TPOTClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np
import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")

if __name__ == '__main__':

    iris = load_iris()

    X_train, X_test, y_train, y_test = train_test_split(iris.data.astype(np.float64),
        iris.target.astype(np.float64), train_size=0.75, test_size=0.25)

    tpot = TPOTClassifier(generations=10, population_size=50, verbosity=2, n_jobs=8)
    tpot.fit(X_train, y_train)
    print(tpot.score(X_test, y_test))
    tpot.export('tpot_iris_pipeline.py')