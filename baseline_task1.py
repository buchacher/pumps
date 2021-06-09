import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import global_random_state

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

# Output training set classification accuracy
train_class_accuracy = accuracy_score(y_train, y_train_hat)
print("Baseline training set classification accuracy: %.3f" % train_class_accuracy)

# Load test place-data-here using pandas
df_test = pd.read_csv('place-data-here/task1_test.csv')

# Split test place-data-here into input X and output y
X_test = df_test.iloc[:, :-1]
y_test = df_test.iloc[:, -1]

# Set index to 'id' column as insignificant as a feature
X_test = X_test.set_index('id')

# Calculate predictions
y_hat = baseline_clf.predict(X_test)

# Output test set classification accuracy
test_class_accuracy = accuracy_score(y_test, y_hat)
print("Baseline test set classification accuracy: %.3f" % test_class_accuracy)
