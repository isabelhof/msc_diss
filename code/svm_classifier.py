import os
import glob
import rasterio
import geopandas as gpd
import numpy as np
from sklearn.svm import SVC
from shapely.geometry import Point
from collections import Counter
import random

# Step 1: Load the shapefile data
shapefile_path = 'subset-testing/training_samples.shp'

# Read shapefile
training_data = gpd.read_file(shapefile_path)

# Directory containing tiled .tif files
input_folder = 'subset-testing/tiled'
output_folder = 'subset-testing/classified/svm'

# Create output directory if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Get list of all .tif files in the input folder
tif_files = glob.glob(os.path.join(input_folder, '*.tif'))

"""
# OPTIONAL: define specific tiles to use for training

# List of specific .tif files to use for training
specific_tiles = [
    'subset-testing/tiled/tile1.tif',
    'subset-testing/tiled/tile2.tif',
    'subset-testing/tiled/tile3.tif',
    'subset-testing/tiled/tile4.tif',
    'subset-testing/tiled/tile5.tif'
]

# Verify that all specified tiles exist
for tile_path in specific_tiles:
    if not os.path.isfile(tile_path):
        raise FileNotFoundError(f"Specified tile {tile_path} does not exist.")
"""

# Step 2: Extract training data
def extract_training_data_from_all_tiles(tif_files, training_data):
    training_samples = []
    training_labels = []

    for raster_path in tif_files:
        with rasterio.open(raster_path) as src:
            raster = src.read()
            crs = src.crs
            
            # Convert training data CRS to match raster CRS
            training_data = training_data.to_crs(crs)
            
            for idx, row in training_data.iterrows():
                if isinstance(row.geometry, Point):
                    coord = (row.geometry.x, row.geometry.y)
                    row_geom = [coord]
                    try:
                        value = list(src.sample(row_geom))[0]
                        training_samples.append(value)
                        training_labels.append(row['class'])  # Assuming 'class' is the column with class info
                    except ValueError:
                        # Skip points that fall outside the tile
                        continue
                
    return np.array(training_samples), np.array(training_labels)

# Extract training data from all tiles
training_samples, training_labels = extract_training_data_from_all_tiles(tif_files, training_data)

class_counts = Counter(training_labels)
for cls, count in class_counts.items():
    print(f"Class {cls}: {count} samples")

# Step 3: Train the SVM classifier
clf = SVC(kernel='rbf', random_state=42)
clf.fit(training_samples, training_labels)

# Step 4: Classify each tile
for raster_path in tif_files:
    with rasterio.open(raster_path) as src:
        raster = src.read()
        raster_meta = src.meta
        rows, cols = raster.shape[1], raster.shape[2]
        X = raster.reshape(raster.shape[0], -1).T
        
        # Predict
        y_pred = clf.predict(X)
        
        # Reshape prediction to the original raster shape
        classified = y_pred.reshape(rows, cols)
        
        # Step 5: Save the classified raster
        tile_name = os.path.basename(raster_path).replace('spatial_subset_', 'svm_')
        output_filename = f"{tile_name}"
        output_path = os.path.join(output_folder, output_filename)
        raster_meta.update(dtype=rasterio.uint8, count=1)
        
        with rasterio.open(output_path, 'w', **raster_meta) as dst:
            dst.write(classified.astype(rasterio.uint8), 1)

    print(f"Classified tile saved to {output_path}")

print("Classification of individual tiles completed.")
