import pickle

import numpy as np
import pandas as pd
from pandas import to_datetime


def task2_predict():
    # Load no-label test place-data-here using pandas
    df_predict = pd.read_csv('place-data-here/task2_test_nolabels.csv')

    # Set index to 'id' column as insignificant as a feature
    X_predict = df_predict.set_index('id')

    # Transform date_recorded to categorical feature
    df_predict['date_recorded'] = to_datetime(df_predict['date_recorded'])
    df_predict['date_recorded'].apply(lambda x: x.strftime('%Y-%m-%d'))

    # Deal with missing values in binary features separately
    binary_features = ['permit', 'public_meeting']
    for feat in binary_features:
        val = df_predict[feat].value_counts()[:1].index.tolist()[0]
        df_predict[feat].replace(np.nan, val)

    # Deal with missing and group low-frequency values in categorical features
    for col in df_predict:
        if df_predict[str(col)].dtype == 'object' and col not in binary_features:
            df_predict[col] = df_predict[col].replace(np.nan, 'rare')
            freq = df_predict[col].value_counts(normalize=True, ascending=True)
            thresh = freq[(freq.cumsum() > 0.1).idxmax()]
            df_predict[col]\
                .mask(df_predict[col]
                      .map(df_predict[col]
                           .value_counts(normalize=True)) < thresh, 'rare')

    # Load saved classifier
    with open('saved-models/task2_model.pkl', 'rb') as f:
        clf = pickle.load(f)

    # Calculate predictions
    y_hat = clf.predict(X_predict)

    # Convert to pandas DataFrame
    df_yhat = pd.DataFrame(data=y_hat, columns=['status_group'])

    # Concatenate predictions with corresponding IDs
    df_concat = pd.concat([df_predict.iloc[:, :1], df_yhat], axis=1)

    # Query pumps needing repair, convert to sorted list and...
    ids_repair = sorted(
        df_concat.query("status_group == 'functional needs repair'")['id'].to_list()
    )
    # ...output to console
    print("The following pumps require repairs:")
    for i in ids_repair:
        print(i)
    print("-----\nTotal: {} pumps".format(len(ids_repair)))
