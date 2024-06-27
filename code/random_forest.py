import os
import glob
import rasterio
import geopandas as gpd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from shapely.geometry import Point

# Step 1: Load the shapefile data
shapefile_path = 'subset-testing/training_samples.shp'

# Read shapefile
training_data = gpd.read_file(shapefile_path)

# Directory containing tiled .tif files
input_folder = 'subset-testing/tiled'
output_folder = 'subset-testing/classified'

# Create output directory if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Get list of all .tif files in the input folder
tif_files = glob.glob(os.path.join(input_folder, '*.tif'))

# Step 2: Extract training data
def extract_training_data(raster_path, training_data):
    with rasterio.open(raster_path) as src:
        raster = src.read()
        crs = src.crs
        
        # Convert training data CRS to match raster CRS
        training_data = training_data.to_crs(crs)
        
        training_samples = []
        training_labels = []
        
        for idx, row in training_data.iterrows():
            if isinstance(row.geometry, Point):
                coord = (row.geometry.x, row.geometry.y)
                row_geom = [coord]
                value = list(src.sample(row_geom))[0]
                training_samples.append(value)
                training_labels.append(row['class'])  # Assuming 'ID' is the column with class info
                
    return np.array(training_samples), np.array(training_labels)

# Collect training samples from multiple tiles
training_samples_list = []
training_labels_list = []

# Specify the number of tiles to use for training
num_training_tiles = 5  # or any other number of tiles you want to use for training

for raster_path in tif_files[:num_training_tiles]:
    samples, labels = extract_training_data(raster_path, training_data)
    training_samples_list.append(samples)
    training_labels_list.append(labels)

# Concatenate all training samples and labels
training_samples = np.concatenate(training_samples_list)
training_labels = np.concatenate(training_labels_list)

# Step 3: Train the Random Forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
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
        output_path = os.path.join(output_folder, os.path.basename(raster_path))
        raster_meta.update(dtype=rasterio.uint8, count=1)
        
        with rasterio.open(output_path, 'w', **raster_meta) as dst:
            dst.write(classified.astype(rasterio.uint8), 1)

    print(f"Classified tile saved to {output_path}")

print("Classification of individual tiles completed.")
