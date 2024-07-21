#!/usr/bin/env python

"""
random forest regression model with hyper-parameter tuning

this script performs the following steps:
1. Data Import and Preprocessing:
    - Reads a CSV file containing training data with spectral features.
    - Splits the data into features (X) and target variable (y).
    - Splits the data into training and testing sets.

2. Baseline Random Forest Model:
    - Initializes a RandomForestRegressor with default parameters.
    - Trains the model on the training data.
    - Evaluates the model on the testing data using Mean Absolute Error (MAE), Mean Squared Error (MSE), and R2 score.

3. Hyperparameter Tuning:
    - Defines a parameter grid for hyperparameter tuning.
    - Initializes a RandomizedSearchCV with the RandomForestRegressor and parameter grid.
    - Performs the randomized search on the training data.
    - Prints the best parameters found and the time taken for the search.
    - Trains a RandomForestRegressor with the best parameters on the training data.
    - Evaluates the optimized model on the testing data using MAE, MSE, and R2 score.

Dependencies:
- pandas
- sklearn (scikit-learn)
- scipy
- time

Usage:
- Ensure the CSV file 'training_data_spectra_FINAL.csv' is available in the specified path.
- Run the script to perform Random Forest regression and hyperparameter tuning on the dataset.

Author: Isabel Hofmockel
Date: 21-06-24
"""

import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import randint
import time


### STEP 1: importing data

# reading in training dataset, which has a class (numeric) attribute, a label (string) attribute, and a column for each band reflectance
df = pd.read_csv('data/testing/training-data/training_data_spectra_FINAL.csv')

# assigning features and target variable
X = df.drop(columns=['Class', 'Label'])
y = df['Class']

# splitting into training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=54)

### STEP 2: Baseline Random Forest model

# Initialize the baseline RandomForestRegressor with default parameters
baseline_rf = RandomForestRegressor(random_state=21)

# Fit the model to the training data
baseline_rf.fit(X_train, y_train)

# Predict on the test data
y_pred_baseline = baseline_rf.predict(X_test)

# Evaluate the baseline model
baseline_mae = mean_absolute_error(y_test, y_pred_baseline)
baseline_mse = mean_squared_error(y_test, y_pred_baseline)
baseline_r2 = r2_score(y_test, y_pred_baseline)

print("Baseline Random Forest Model Performance:")
print(f"Mean absolute error: {baseline_mae:.4f}")
print(f"Mean squared error: {baseline_mse:.4f}")
print(f"R2 score: {baseline_r2:.4f}")
print("")

### STEP 3: Hyperparameter tuning

# Define the parameter grid
param_dist = {
    'n_estimators': randint(100, 1000),
    'max_depth': randint(10, 50),
    'min_samples_split': randint(2, 32),
    'min_samples_leaf': randint(1, 32),
    'max_features': ['sqrt', 'log2'],
    'criterion': ["squared_error", "absolute_error", "friedman_mse", "poisson"]
}

# Initialize the RandomForestRegressor
rf = RandomForestRegressor(random_state=21)

# Initialize the RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_dist,
    n_iter=100,
    scoring='neg_mean_squared_error',
    cv=3,
    random_state=42,
    verbose=1,
    n_jobs=-1
)

# Track the start time
start_time = time.time()

# Perform the search
random_search.fit(X_train, y_train)

# Track the end time
end_time = time.time()

# Print the time taken
print(f"RandomizedSearchCV took {end_time - start_time:.2f} seconds for 100 candidates parameter settings.")

# Print the best parameters found
print("Best parameters found: ", random_search.best_params_)

# Train the final model with the best parameters
best_rf = random_search.best_estimator_
best_rf.fit(X_train, y_train)

# Evaluate the final model on the test set
y_pred = best_rf.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Optimized Random Forest Model Performance:")
print(f"Mean absolute error: {mae:.4f}")
print(f"Mean squared error: {mse:.4f}")
print(f"R2 score: {r2:.4f}")

"""
OUTPUT:
Best parameters found:
'criterion': 'squared_error',
'max_depth': 48,
'max_features': 'log2',
'min_samples_leaf': 1,
'min_samples_split': 2,
'n_estimators': 476
"""
