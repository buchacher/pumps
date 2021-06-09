import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score


def task1_test():
    # Load test place-data-here using pandas
    df_test = pd.read_csv('place-data-here/task1_test.csv')

    # Split test place-data-here into input X and output y
    X_test = df_test.iloc[:, :-1]
    y_test = df_test.iloc[:, -1]

    # Set index to 'id' column as insignificant as a feature
    X_test = X_test.set_index('id')

    # Load saved classifier
    with open('saved-models/task1_model.pkl', 'rb') as f:
        clf = pickle.load(f)

    # Calculate predictions
    y_hat = clf.predict(X_test)

    # Output corresponding predictions to a text file
    np.savetxt('output/task1_test.out', y_hat, fmt='%s')

    # Output classification accuracy
    class_accuracy = accuracy_score(y_test, y_hat)
    print("Test set classification accuracy: %.3f" % class_accuracy)
