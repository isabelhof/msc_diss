#!/usr/bin/env python
# THIS WORKS 
"""
Random Forest Model Prediction

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
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

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
    
    # Assuming the first column is labels and the rest are features
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

# Function to process a single .tif file in chunks
def process_tif_file(args):
    filename, input_folder, output_folder, model = args
    filepath = os.path.join(input_folder, filename)
    output_path = os.path.join(output_folder, f'predicted_{filename}')
    
    if os.path.exists(output_path):
        return
    
    with rasterio.open(filepath) as src:
        transform = src.transform
        crs = src.crs
        dtype = src.dtypes[0]
        n_bands = src.count
        tile_size = 1000  # Size of the tile (chunk)
        
        # Calculate number of tiles in x and y directions
        width = src.width
        height = src.height
        
        for i in range(0, width, tile_size):
            for j in range(0, height, tile_size):
                # Define the window
                window = rasterio.windows.Window(i, j, min(tile_size, width - i), min(tile_size, height - j))
                
                # Read the chunk
                chunk = src.read(window=window)
                
                # Reshape the chunk for prediction
                chunk_reshaped = chunk.reshape((n_bands, -1)).T
                predictions = model.predict(chunk_reshaped)
                
                # Reshape predictions back to chunk shape
                predictions_reshaped = predictions.reshape((min(tile_size, height - j), min(tile_size, width - i)))
                
                # Write the predictions to the output file
                if i == 0 and j == 0:
                    # Create the output file
                    with rasterio.open(
                        output_path,
                        'w',
                        driver='GTiff',
                        height=height,
                        width=width,
                        count=1,
                        dtype=rasterio.uint8,
                        crs=crs,
                        transform=src.transform
                    ) as dst:
                        dst.write(predictions_reshaped, 1)
                else:
                    # Append to the output file
                    with rasterio.open(output_path, 'r+') as dst:
                        dst.write(predictions_reshaped, 1, window=window)

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
    
    for root, dirs, files in os.walk(input_folder):
        for dir in dirs:
            subfolder_path = os.path.join(root, dir)
            output_subfolder_path = subfolder_path.replace(input_folder, output_folder, 1)
            if not os.path.exists(output_subfolder_path):
                os.makedirs(output_subfolder_path)
        
        tif_files = [f for f in files if f.endswith('.img')]
        args = [(filename, root, root.replace(input_folder, output_folder, 1), model) for filename in tif_files]
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Use tqdm to show progress bar
            for _ in tqdm(executor.map(process_tif_file, args), total=len(args), desc=f"Processing files in {root}"):
                pass

# Parameters
csv_file = 'data/two_class_training_FINAL.csv'
input_folder = 'raw'
output_folder = 'data/outputs/RF_two_class'
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