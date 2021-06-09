import pandas as pd
from matplotlib import pyplot as plt

# Load training place-data-here using pandas
df_train = pd.read_csv('place-data-here/task1_train.csv')

# Split training place-data-here into input X and output y
X_train = df_train.iloc[:, :-1]

# Set index to 'id' column as insignificant as a feature
X_train = X_train.set_index('id')

# Plot histogram per numeric input feature
for col in X_train.columns.tolist():
    plt.hist(X_train[str(col)])
    plt.title(str(col))
    plt.show()
