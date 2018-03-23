import pandas as pd
import prince
import matplotlib.pyplot as plt

# Generate tain and test data

df = pd.read_csv('data/datalab_persona_run1_with_scale_cat_classless.csv')

df2 = pd.read_csv('data/datalab_persona_run1_with_scale_cat.csv')
df_class = df2['class'].values

cols = ['g' if x=='smoker' else 'b' for x in df_class]

# df = pd.read_csv('data/ogm.csv')

# cols = [x for x in df2.columns.values if
#         x not in ['Age Next at DOC', 'Height', 'Weight', 'Annual Salary', 'Travel %']]

# df = pd.get_dummies(df)

mca = prince.MCA(df, n_components=-1)

# Set the axes you want to examine below, i.e. which component pair you are interested in - (0, 1)

vals = mca.row_principal_coordinates

print(len(vals))

vals=vals.values

plt.scatter(vals[:,0], vals[:,1], c=cols)

mca = prince.MCA(df2, n_components=-1)

# Set the axes you want to examine below, i.e. which component pair you are interested in - (0, 1)

components = (0, 1)

mca.plot_rows(axes=components, color_by='class', ellipse_fill=True,)
# mca.plot_cumulative_inertia()
# mca.plot_inertia()
# mca.plot_rows_columns(show_column_labels=True)
# mca.plot_relationship_square()

plt.show()
