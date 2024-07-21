import pandas as pd
import seaborn as sns
from sklearn import metrics
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import optuna
import time
import logging

### STEP 1: importing data

# reading in training dataset, which has a class (numeric) attribute, a label (string) attribute, and a column for each band reflectance
df = pd.read_csv('data/testing/training-data/training_data_spectra_FINAL.csv')

# assigning features and target variable
X = df.drop(columns=['Class', 'Label'])
y = df['Class']

# splitting into training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=54)

### STEP 2: hyperparameter tuning

# defining the hyperparameter suggestions, each range is inclusive
def objective(trial):
    start_time = time.time()
    
    n_estimators = trial.suggest_int('n_estimators', 100, 1000) # number of estimators in model construction
    max_depth = trial.suggest_int('max_depth', 10, 50) # maximum depth of individual trees
    min_samples_split = trial.suggest_int('min_samples_split', 2, 32) # min number of samples needed to split an internal node in a tree
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 32) # numum number of samples needed to be at a leaf node
    max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2']) # number of features to consider when looking for the best split
    criterion = trial.suggest_categorical('criterion', ["squared_error", "absolute_error", "friedman_mse", "poisson"]) # function to measure the quality of a split

    # initalising a model with the hyperparameters suggested by the tuning
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        criterion=criterion,
        random_state=21
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # metric  to optimize
    score = mean_squared_error(y_test, y_pred)
    
    end_time = time.time()
    trial_duration = end_time - start_time
    print(f"Trial {trial.number} finished in {trial_duration:.2f} seconds with score {score:.4f}")

    return score

# Set up logging
optuna.logging.set_verbosity(optuna.logging.INFO)
logging.getLogger().setLevel(logging.INFO)

study = optuna.create_study(direction='minimize', sampler=optuna.samplers.RandomSampler(seed=42))
study.optimize(objective, n_trials=200, callbacks=[lambda study, trial: print(f"Trial {trial.number} completed. Best score so far: {study.best_value:.4f}")])

# print the best parameters found 
print("Best trial:")
trial = study.best_trial

print("Value: {:.4f}".format(trial.value))

print("Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))
