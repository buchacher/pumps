import pickle

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTENC
from matplotlib import pyplot as plt
from pandas import to_datetime
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, plot_precision_recall_curve, \
    f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from config import global_random_state


def task4_train():
    # Load training place-data-here using pandas
    df_train = pd.read_csv('place-data-here/task3_train.csv')

    '''Uncomment below to show histogram of imbalanced dataset'''
    # plt.hist(df_train['status_group'])
    # plt.title("Class Distribution Before Resampling")
    # plt.show()

    # Over-sample - chosen technique for balancing the dataset
    num_dominant_examples = df_train['status_group'].value_counts().max()
    df_examples = [df_train]
    for idx, group in df_train.groupby('status_group'):
        df_examples.append(
            group.sample(num_dominant_examples - len(group), replace=True)
        )
    df_train = pd.concat(df_examples)

    '''Uncomment below to perform under-sampling'''
    # num_major_ex, num_minor_ex = df_train['status_group'].value_counts()
    # df_major = df_train[df_train['status_group'] == 'others']
    # df_minor = df_train[df_train['status_group'] == 'functional needs repair']
    # df_major_sample = df_major.sample(num_minor_ex)
    # df_train = pd.concat([df_major_sample, df_minor], axis=0)

    '''Uncomment below to show histogram of now balanced dataset'''
    # plt.hist(df_train['status_group'])
    # plt.title("Class Distribution After Resampling")
    # plt.show()

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

    '''Uncomment below to perform SMOTE-NC resampling'''
    # date_dict = {'date_recorded': str}
    # X_train = X_train.astype(date_dict)
    # permit_dict = {'permit': str}
    # X_train = X_train.astype(permit_dict)
    # public_dict = {'public_meeting': str}
    # X_train = X_train.astype(public_dict)
    #
    # cat_idx = []
    # for col in X_train:
    #     # Obtain column indexes for categorical features
    #     if X_train[str(col)].dtype == 'object':
    #         cat_idx.append(X_train.columns.get_loc(col))
    #
    # sm = SMOTENC(
    #     categorical_features=cat_idx,
    #     sampling_strategy='minority',
    #     random_state=global_random_state
    # )
    #
    # print("Started SMOTE-NC resampling ...")
    # X_train, y_train = sm.fit_resample(X_train, y_train)
    # print("Completed SMOTE-NC resampling ...")

    '''Uncomment below to show histogram of now balanced dataset for SMOTE-NC'''
    # plt.hist(y_train)
    # plt.title("Class Distribution After SMOTE-NC")
    # plt.show()

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
        hidden_layer_sizes=[25, 25],
        batch_size=100,
        activation='relu',
        solver='sgd',
        learning_rate_init=0.001,
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
    print("Started neural net training ...")
    clf.fit(X_train, y_train)
    print("Completed neural net training ...")

    # Get predicted labels
    y_hat_train = clf.predict(X_train)

    # Output classification accuracy
    train_class_accuracy = accuracy_score(y_train, y_hat_train)
    print("Training set classification accuracy: %.3f" % train_class_accuracy)

    # Output balanced accuracy
    train_bal_accuracy = balanced_accuracy_score(y_train, y_hat_train)
    print("Training set balanced accuracy: %.3f" % train_bal_accuracy)

    # Output F1 score
    train_f1_score = f1_score(y_train, y_hat_train, average='binary', pos_label='functional needs repair')
    print("Training set F1 score: %.3f" % train_f1_score)

    # Output confusion matrix
    train_conf_matrix = confusion_matrix(y_train, y_hat_train)
    print("Training set confusion matrix:")
    print(train_conf_matrix)

    # Plot precision-recall-curve
    train_prc = plot_precision_recall_curve(clf, X_train, y_train, pos_label='functional needs repair')
    train_prc.ax_.set_title("Task 4 Training Precision-Recall Curve")
    train_prc.plot()
    plt.show()

    # Save the built classifier (incl. preprocessor)
    with open('saved-models/task4_model.pkl', 'wb') as f:
        pickle.dump(clf, f)
