import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pandas import to_datetime
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, plot_precision_recall_curve, \
    f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from config import global_random_state

# Load place-data-here using pandas
df_train = pd.read_csv('place-data-here/task3_train.csv')

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

# Build a standard logistic regression classifier
log_reg = LogisticRegression(max_iter=1000, random_state=global_random_state)

# Combine preprocessor and model into one pipeline
baseline_clf = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('baseline', log_reg)
])

# Apply preprocessing and train classifier
baseline_clf.fit(X_train, y_train)

# Get predicted labels
y_train_hat = baseline_clf.predict(X_train)

# Output classification accuracy
train_class_accuracy = accuracy_score(y_train, y_train_hat)
print("Baseline training set classification accuracy: %.3f" % train_class_accuracy)

# Output balanced accuracy
train_bal_accuracy = balanced_accuracy_score(y_train, y_train_hat)
print("Baseline training set balanced accuracy: %.3f" % train_bal_accuracy)

# Output F1 score
train_f1_score = f1_score(y_train, y_train_hat, average='binary', pos_label='functional needs repair')
print("Baseline training set F1 score: %.3f" % train_f1_score)

# Output confusion matrix
train_conf_matrix = confusion_matrix(y_train, y_train_hat)
print("Baseline training set confusion matrix:")
print(train_conf_matrix)

# Plot precision-recall-curve
train_prc = plot_precision_recall_curve(baseline_clf, X_train, y_train, pos_label='functional needs repair')
train_prc.ax_.set_title("Task 3 Baseline Training Precision-Recall Curve")
train_prc.plot()
plt.show()

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

# Calculate predictions
y_hat = baseline_clf.predict(X_test)

# Output classification accuracy
test_class_accuracy = accuracy_score(y_test, y_hat)
print("Baseline test set classification accuracy: %.3f" % test_class_accuracy)

# Output balanced accuracy
test_bal_accuracy = balanced_accuracy_score(y_test, y_hat)
print("Baseline test set balanced accuracy: %.3f" % test_bal_accuracy)

# Output F1 score
test_f1_score = f1_score(y_test, y_hat, average='binary', pos_label='functional needs repair')
print("Baseline test set F1 score: %.3f" % test_f1_score)

# Output confusion matrix
test_conf_matrix = confusion_matrix(y_test, y_hat)
print("Baseline test set confusion matrix:")
print(test_conf_matrix)

# Plot precision-recall-curve
test_prc = plot_precision_recall_curve(baseline_clf, X_test, y_test, pos_label='functional needs repair')
test_prc.ax_.set_title("Task 3 Baseline Test Precision-Recall Curve")
test_prc.plot()
plt.show()
