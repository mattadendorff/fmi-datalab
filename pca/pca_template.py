import pandas as pd
import prince
import matplotlib.pyplot as plt

# Generate tain and test data
# df = pd.read_csv('data/datalab_persona_run1_with_scale_cont.csv')
df = pd.read_csv('data/iris.csv')

pca = prince.PCA(df, n_components=-1)

# Set the axes you want to examine below, i.e. which component pair you are interested in - (0, 1)

components = (0, 1)

pca.plot_rows(axes=components, color_by='class', ellipse_fill=True)
pca.plot_correlation_circle(axes=components)
pca.plot_cumulative_inertia()
pca.plot_inertia()

plt.show()
