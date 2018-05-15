#import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Generate tain and test data
df = pd.read_csv('data/datalab_persona_run1_with_scale.csv')

target = df['FKSmoker'].values

df.drop(['FKSmoker'], inplace=True, axis=1)

cols = [x for x in df.columns.values if
        x not in ['Age Next at DOC', 'Height', 'Weight', 'Annual Salary', 'Travel %']]

df = pd.get_dummies(df, columns=cols)

data = df.values

# sm = SMOTE(kind='svm')
# data, target = sm.fit_sample(data, target)

X_embedded = TSNE(n_components=3, init='pca').fit_transform(data)

vals = X_embedded

cols = ['g' if x == 0 else 'b' for x in target]

plt.scatter(vals[:, 0], vals[:, 1], c=cols)
plt.show()

np.savetxt('TSNE_Embedding_Heuristic_3.txt', vals, delimiter=',')