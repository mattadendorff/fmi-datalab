import pandas as pd
import prince
import matplotlib.pyplot as plt

df = pd.read_excel('BPE&INDdataForSurivivalModel.xlsx')

df = df[df['Duration'] >= 12]

df2 = df.loc[:, ['DISTRIBUTION CHANNEL', 'GENDER', 'SMOKER STATUS','PremiumPattern', 'BENEFITS TYPE',
                 'BROKER COMM', 'DEBITORDERPERIOD', 'PREM % EARNINGS BAND']]

mca = prince.MCA(df2, n_components=13)

# Set the axes you want to examine below, i.e. which component pair you are interested in - (0, 1)

components = (0, 1)

# mca.plot_rows(axes=components, color_by='class', ellipse_fill=True,)
mca.plot_cumulative_inertia()
mca.plot_inertia()
# mca.plot_rows_columns(show_column_labels=True)
mca.plot_relationship_square()

plt.show()
