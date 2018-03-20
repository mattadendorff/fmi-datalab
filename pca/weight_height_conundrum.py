from imblearn.over_sampling import SMOTE
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Generate tain and test data
df = pd.read_csv('data/datalab_persona_run1_with_scale_cont.csv')

target = df['FKSmoker'].values

df.drop(['FKSmoker'], inplace=True, axis=1)

X = df.values
y_res = target


plt.figure()
colors = ['r' for x in y_res if x==0]
X1 = X[y_res==0]
plt.scatter(X1[:, 0], X1[:, 2], c=colors, alpha=.5, s=10)

colors = ['g' for x in y_res if x==1]
X2 = X[y_res==1]
plt.scatter(X2[:, 0], X2[:, 2], c=colors, alpha=.5, s=10)
plt.show()
