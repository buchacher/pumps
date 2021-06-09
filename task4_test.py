import pickle

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pandas import to_datetime
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, plot_precision_recall_curve, \
    f1_score


def task4_test():
    # Load test place-data-here using pandas
    df_test = pd.read_csv('place-data-here/task3_test.csv')

    # Split test place-data-here into input X and output y
    X_test = df_test.iloc[:, :-1]
    y_test = df_test.iloc[:, -1]

    # Set index to 'id' column as insignificant as a feature
    X_test = X_test.set_index('id')

    # Transform date_recorded to categorical feature
    X_test['date_recorded'] = to_datetime(X_test['date_recorded'])
    X_test['date_recorded'].apply(lambda x: x.strftime('%Y-%m-%d'))

    # Deal with missing values in binary features separately
    binary_features = ['permit', 'public_meeting']
    for feat in binary_features:
        val = X_test[feat].value_counts()[:1].index.tolist()[0]
        X_test[feat].replace(np.nan, val)

    # Deal with missing and group low-frequency values in categorical features
    for col in X_test:
        if X_test[str(col)].dtype == 'object' and col not in binary_features:
            X_test[col] = X_test[col].replace(np.nan, 'rare')
            freq = X_test[col].value_counts(normalize=True, ascending=True)
            thresh = freq[(freq.cumsum() > 0.1).idxmax()]
            X_test[col]\
                .mask(X_test[col]
                      .map(X_test[col]
                           .value_counts(normalize=True)) < thresh, 'rare')

    # Load saved classifier
    with open('saved-models/task4_model.pkl', 'rb') as f:
        clf = pickle.load(f)

    # Calculate predictions
    y_hat = clf.predict(X_test)

    # Output corresponding predictions to a text file
    np.savetxt('output/task4_test.out', y_hat, fmt='%s')

    # Output classification accuracy
    test_class_accuracy = accuracy_score(y_test, y_hat)
    print("Test set classification accuracy: %.3f" % test_class_accuracy)

    # Output balanced accuracy
    test_bal_accuracy = balanced_accuracy_score(y_test, y_hat)
    print("Test set balanced accuracy: %.3f" % test_bal_accuracy)

    # Output F1 score
    test_f1_score = f1_score(y_test, y_hat, average='binary', pos_label='functional needs repair')
    print("Test set F1 score: %.3f" % test_f1_score)

    # Output confusion matrix
    test_conf_matrix = confusion_matrix(y_test, y_hat)
    print("Test set confusion matrix:")
    print(test_conf_matrix)

    # Plot precision-recall-curve
    test_prc = plot_precision_recall_curve(clf, X_test, y_test, pos_label='functional needs repair')
    test_prc.ax_.set_title("Task 4 Test Precision-Recall Curve")
    test_prc.plot()
    plt.show()
