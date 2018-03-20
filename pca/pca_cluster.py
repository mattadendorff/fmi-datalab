# -*- coding: utf-8 -*-
"""
Created on Fri Mar 16 13:38:53 2018

@author: mrade_000
"""

from imblearn.over_sampling import SMOTE
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import AffinityPropagation, Birch, KMeans
import matplotlib.pyplot as plt
from sklearn import metrics

def scale_array(A):
    A -= np.amin(A)
    A /= np.amax(A)
    A *= 100
    return A

# Generate tain and test data
df1 = pd.read_csv('data/height_weight.csv')
target = df1['FKSmoker'].values
df1.drop(['FKSmoker'], inplace=True, axis=1)
data1 = df1.values

df2 = pd.read_csv('data/age_salary.csv')
df2.drop(['FKSmoker'], inplace=True, axis=1)
data2 = df2.values

pca1 = PCA(n_components=1)
var1 = pca1.fit_transform(data1)
var1 = scale_array(var1)

pca2 = PCA(n_components=1)
var2 = pca2.fit_transform(data2)
var2 = scale_array(var2)

X_n = pd.DataFrame({'HW': pd.Series(list(var1)), 'AS': pd.Series(list(var2))})
X_n = X_n.values
#af = Birch(threshold=2, branching_factor=110, n_clusters=None)
af = KMeans(n_clusters=2)
af.fit(X_n)
cluster_centers = af.cluster_centers_
labels = af.labels_

# np.savetxt('Post_cluster_labels.txt', np.array(labels))

n_clusters_ = len(cluster_centers)

print('Estimated number of clusters: %d' % n_clusters_)
print("Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(X_n, labels, metric='sqeuclidean'))
      
# #############################################################################
# Plot result
import matplotlib.pyplot as plt
from itertools import cycle

fig = plt.figure()

colors = cycle(['#e6194b', '#3cb44b', '#ffe119', '#0082c8', '#f58231', '#911eb4', '#46f0f0', '#f032e6', '#d2f53c',
                '#fabebe', '#008080', '#e6beff', '#aa6e28', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1',
                '#000080', '#808080', '#000000'])

for k, col in zip(range(n_clusters_), colors):
    my_members = labels == k
    cluster_center = cluster_centers[k]
    for x in X_n[my_members]:
        plt.plot([cluster_center[0], x[0]], [cluster_center[1], x[1]], c=col)

    plt.scatter(cluster_center[0], cluster_center[1],c=col, edgecolor='k', s=100)

#    for x in X_n[my_members]:
#       plt.plot([cluster_center[0], x[0]], [cluster_center[1], x[1]], c=col)
        
plt.xlabel('HW')
plt.ylabel('AS')
plt.dist = 12

plt.show()