#!/usr/bin/env python

"""
random forest model prediction

This script trains a Random Forest model using land cover data and applies the trained model to .tif files to predict land cover classes.

Functions:
- train_and_evaluate_random_forest: Trains a Random Forest model, evaluates it using accuracy and classification report metrics, and returns the trained model.
- apply_model_to_tifs: Applies the trained model to .tif files in a specified input folder and saves the predictions to an output folder.

Parameters:
- csv_file: Path to the CSV file containing training data with land cover classes and band reflectance data.
- input_folder: Path to the folder containing .tif files for prediction.
- output_folder: Path to the folder where the prediction results will be saved.
- n_estimators: Number of trees in the Random Forest.
- max_depth: Maximum depth of the trees in the Random Forest.
- min_samples_leaf: Minimum number of samples required to be at a leaf node.
- min_samples_split: Minimum number of samples required to split an internal node.
- max_features: Number of features to consider when looking for the best split.
- random_state: Random seed for reproducibility.

Usage:
- Train the model by running the script with the specified parameters.
- Apply the trained model to .tif files in the input folder and save the predictions to the output folder.

"""

import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import rasterio
from rasterio.plot import reshape_as_image, reshape_as_raster
from concurrent.futures import ProcessPoolExecutor

# Function to train and evaluate Random Forest model
def train_and_evaluate_random_forest(csv_file, n_estimators, max_depth, random_state, min_samples_leaf, min_samples_split, max_features):
    """
    Trains a Random Forest model and evaluates its performance.

    Parameters:
    csv_file (str): Path to the CSV file containing training data.
    n_estimators (int): Number of trees in the forest.
    max_depth (int): Maximum depth of the trees.
    random_state (int): Seed for the random number generator.
    min_samples_leaf (int): Minimum number of samples required to be at a leaf node.
    min_samples_split (int): Minimum number of samples required to split an internal node.
    max_features (str): Number of features to consider when looking for the best split.

    Returns:
    RandomForestClassifier: Trained Random Forest model.
    """
    # Load training data
    data = pd.read_csv(csv_file)
    
    # Assuming the first two columns are labels and the rest are features
    X = data.iloc[:, 2:].values  # Omitting the first two columns
    y = data.iloc[:, 0].values   # Assuming the first column is 'land_cover_class'
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
    
    # Train Random Forest model
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        min_samples_leaf=min_samples_leaf,
        min_samples_split=min_samples_split,
        max_features=max_features,
    )
    model.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    print(f"Model Accuracy: {accuracy}")
    print(f"Classification Report:\n{report}")
    
    return model

# Function to process a single .tif file
def process_tif_file(filename, input_folder, output_folder, model):
    filepath = os.path.join(input_folder, filename)
    with rasterio.open(filepath) as src:
        # Read the image data
        image = src.read()
        
        # Reshape the image data for prediction
        reshaped_image = reshape_as_image(image)
        n_rows, n_cols, n_bands = reshaped_image.shape
        
        reshaped_image = reshaped_image.reshape((n_rows * n_cols, n_bands))
        
        # Predict land cover classes
        predictions = model.predict(reshaped_image)
        
        # Reshape predictions back to original image shape
        reshaped_predictions = predictions.reshape((n_rows, n_cols))
        
        # Write the predictions to a new .tif file
        output_path = os.path.join(output_folder, f'predicted_{filename}')
        with rasterio.open(
            output_path,
            'w',
            driver='GTiff',
            height=n_rows,
            width=n_cols,
            count=1,
            dtype=rasterio.uint8,
            crs=src.crs,
            transform=src.transform
        ) as dst:
            dst.write(reshaped_predictions, 1)

# Function to apply the model to .tif files in a folder
def apply_model_to_tifs(model, input_folder, output_folder, max_workers=4):
    """
    Applies a trained model to .tif files in the input folder and saves the predictions to the output folder.

    Parameters:
    model (RandomForestClassifier): Trained Random Forest model.
    input_folder (str): Path to the folder containing .tif files.
    output_folder (str): Path to the folder where the prediction results will be saved.
    max_workers (int): Number of parallel processes to use.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    tif_files = [f for f in os.listdir(input_folder) if f.endswith('.tif')]
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_tif_file, filename, input_folder, output_folder, model) for filename in tif_files]
        
        for future in futures:
            future.result()  # Wait for all futures to complete

# Parameters
csv_file = 'data/testing/training_data_spectra_FINAL.csv'
input_folder = 'data/hs-tiled/flightline-1'
output_folder = 'data/outputs/RF'
n_estimators = 476  # Set your optimal number of trees
max_depth = 48      # Set your optimal max depth
min_samples_leaf = 1
min_samples_split = 2
max_features = 'log2'
random_state = 42   # Set your random state for reproducibility
max_workers = 4     # Number of parallel processes

# Train and evaluate the Random Forest model
model = train_and_evaluate_random_forest(csv_file, n_estimators, max_depth, random_state, min_samples_leaf, min_samples_split, max_features)

# Apply the model to .tif files in the input folder and save predictions
apply_model_to_tifs(model, input_folder, output_folder, max_workers)
