import numpy as np
import pandas as pd
import mca

df = pd.read_csv('data/datalab_persona_run1_with_scale_cat.csv')

target = df['FKSmoker'].values

target = np.array([target, -(target-1)]).T

df.drop(['FKSmoker'], inplace=True, axis=1)

cols = [x for x in df.columns.values if
        x not in ['Age Next at DOC', 'Height', 'Weight', 'Annual Salary', 'Travel %']]

df = pd.get_dummies(df, columns=cols)

X = df.values
ncols = len(df.columns.values)

mca_ben = mca.MCA(X, ncols=ncols)
mca_ind = mca.MCA(X, ncols=ncols, benzecri=False)
