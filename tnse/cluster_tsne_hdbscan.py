from imblearn.over_sampling import SMOTE
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import AffinityPropagation, Birch, KMeans, DBSCAN, AgglomerativeClustering
import matplotlib.pyplot as plt
from sklearn import metrics
import hdbscan

X_n = np.genfromtxt('TSNE_Embedding_Exact.txt', delimiter=',')

df = pd.read_csv('data/datalab_persona_run1_with_scale.csv')

target = df['FKSmoker'].values

# af = Birch(threshold=2, branching_factor=110, n_clusters=None)
# af = KMeans(n_clusters=30)
# af = AgglomerativeClustering(n_clusters=5)
# af.fit(X_n)
# cluster_centers = af.cluster_centers_
# labels = af.labels_

clusterer = hdbscan.HDBSCAN(min_cluster_size=80, min_samples=1, metric='sqeuclidean')
clusterer.fit(X_n)
labels = clusterer.labels_

for lab in set(labels):

    cluster_smoke = target[[labels == lab]]

    print(100*len(cluster_smoke[cluster_smoke==0])/len(cluster_smoke))

# np.savetxt('Post_cluster_labels.txt', np.array(labels))

n_clusters_ = len(set(labels))

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

for k, col in zip(range(len(set(labels))), colors):
    my_members = labels == k
    # cluster_center = cluster_centers[k]
    # for x in X_n[my_members]:
        #plt.plot([cluster_center[0], x[0]], [cluster_center[1], x[1]], c=col)
    plt.scatter(X_n[my_members][:,0], X_n[my_members][:,1], c=col)

    # plt.scatter(cluster_center[0], cluster_center[1], c=col, edgecolor='k', s=100)

#    for x in X_n[my_members]:
#       plt.plot([cluster_center[0], x[0]], [cluster_center[1], x[1]], c=col)

# plt.xlabel('HW')
# plt.ylabel('AS')
plt.dist = 12

plt.show()