import pickle

import numpy as np
import pandas as pd
from pandas import to_datetime
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from config import global_random_state


def task2_train():
    # Load training place-data-here using pandas
    df_train = pd.read_csv('place-data-here/task2_train.csv')

    # Shuffle place-data-here
    df_train = df_train.sample(frac=1, random_state=global_random_state)

    # Split training place-data-here into input X and output y
    X_train = df_train.iloc[:, :-1]
    y_train = df_train.iloc[:, -1]

    # Set index to 'id' column as insignificant as a feature
    X_train = X_train.set_index('id')

    # Transform date_recorded to categorical feature
    X_train['date_recorded'] = to_datetime(X_train['date_recorded'])
    X_train['date_recorded'].apply(lambda x: x.strftime('%Y-%m-%d'))

    # Deal with missing values in binary features separately
    binary_features = ['permit', 'public_meeting']
    for feat in binary_features:
        val = X_train[feat].value_counts()[:1].index.tolist()[0]
        X_train[feat].replace(np.nan, val)

    # Deal with missing and group low-frequency values in categorical features
    for col in X_train:
        if X_train[str(col)].dtype == 'object' and col not in binary_features:
            X_train[col] = X_train[col].replace(np.nan, 'rare')
            freq = X_train[col].value_counts(normalize=True, ascending=True)
            thresh = freq[(freq.cumsum() > 0.1).idxmax()]
            X_train[col]\
                .mask(X_train[col]
                      .map(X_train[col]
                           .value_counts(normalize=True)) < thresh, 'rare')

    # Build transformer for numeric features
    num_transformer = Pipeline([
        ('scaler', StandardScaler())
    ])

    # Build transformer for categorical features
    cat_transformer = Pipeline([
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Combine transformers
    preprocessor = ColumnTransformer(transformers=[
        ('num', num_transformer, make_column_selector(dtype_include=['float64', 'int64'])),
        ('cat', cat_transformer, make_column_selector(dtype_include='object'))
    ])

    # Build a multi-layer perceptron classifier
    model = MLPClassifier(
        hidden_layer_sizes=[100, 100],
        batch_size=100,
        activation='relu',
        solver='sgd',
        learning_rate_init=0.1,
        momentum=0.9,
        early_stopping=True,
        n_iter_no_change=10,
        random_state=global_random_state
    )

    # Combine preprocessor and model into one pipeline
    clf = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', model)
    ])

    # Apply preprocessing and train classifier
    clf.fit(X_train, y_train)

    # Get predicted labels
    y_hat_train = clf.predict(X_train)

    # Output classification accuracy
    class_accuracy = accuracy_score(y_train, y_hat_train)
    print("Training set classification accuracy: %.3f" % class_accuracy)

    '''Uncomment below to save the built classifier (incl. preprocessor)'''
    with open('saved-models/task2_model.pkl', 'wb') as f:
        pickle.dump(clf, f)