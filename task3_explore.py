import pandas as pd

# Load training place-data-here using pandas
df_train = pd.read_csv('place-data-here/task3_train.csv')

# Split training place-data-here into input X and output y
X_train = df_train.iloc[:, :-1]

# Set index to 'id' column as insignificant as a feature
X_train = X_train.set_index('id')

# Output number of and list unique values per categorical input feature
for (col, values) in X_train.iteritems():
    if X_train[col].dtype == 'object':
        print("\nCol: ", col)
        print("Num unique: ", len(values.unique()))
        print("Unique values: ", values.unique())
