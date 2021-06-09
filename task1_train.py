import pickle

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import global_random_state


def task1_train():
    # Load training place-data-here using pandas
    df_train = pd.read_csv('place-data-here/task1_train.csv')

    # Shuffle place-data-here
    df_train = df_train.sample(frac=1, random_state=global_random_state)

    # Split training place-data-here into input X and output y
    X_train = df_train.iloc[:, :-1]
    y_train = df_train.iloc[:, -1]

    # Set index to 'id' column as insignificant as a feature
    X_train = X_train.set_index('id')

    # Build preprocessing pipeline for place-data-here (comprised of numeric features only)
    preprocessor = Pipeline([
        ('scaler', StandardScaler())
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
    with open('saved-models/task1_model.pkl', 'wb') as f:
        pickle.dump(clf, f)
