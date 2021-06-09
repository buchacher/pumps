import pickle

import pandas as pd


def task1_predict():
    # Load no-label test place-data-here using pandas
    df_predict = pd.read_csv('place-data-here/task1_test_nolabels.csv')

    # Set index to 'id' column as insignificant as a feature
    X_predict = df_predict.set_index('id')

    # Load saved classifier
    with open('saved-models/task1_model.pkl', 'rb') as f:
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
