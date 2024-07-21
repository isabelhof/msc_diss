import os
import glob
import rasterio
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point
from collections import Counter
import pandas as pd
import spectral

# Step 1: Load the shapefile data
shapefile_path = 'subset-testing/training_samples.shp'

# Read shapefile
training_data = gpd.read_file(shapefile_path)

# Directory containing tiled .tif files
input_folder = 'subset-testing/tiled'
output_folder = 'subset-testing/classified/rf'

# Create output directory if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Get list of all .tif files in the input folder
tif_files = glob.glob(os.path.join(input_folder, '*.tif'))

# Step 2: Extract training data
def extract_training_data_from_all_tiles(tif_files, training_data):
    training_samples = []
    training_labels = []
    tile_names = []

    for raster_path in tif_files:
        with rasterio.open(raster_path) as src:
            raster = src.read()
            crs = src.crs
            
            # Convert training data CRS to match raster CRS
            training_data = training_data.to_crs(crs)
            
            tile_name = os.path.basename(raster_path).replace('.tif', '')
            
            for idx, row in training_data.iterrows():
                if isinstance(row.geometry, Point):
                    coord = (row.geometry.x, row.geometry.y)
                    row_geom = [coord]
                    try:
                        value = list(src.sample(row_geom))[0]
                        training_samples.append(value)
                        training_labels.append(row['class'])  # Assuming 'class' is the column with class info
                        tile_names.append(tile_name)
                    except ValueError:
                        # Skip points that fall outside the tile
                        continue
                
    return np.array(training_samples), np.array(training_labels), tile_names

# Extract training data from all tiles
training_samples, training_labels, tile_names = extract_training_data_from_all_tiles(tif_files, training_data)

"""
# Step 3: Export training data to CSV
# Create a DataFrame with band values, labels, and tile names
columns = [f'band{i+1}' for i in range(training_samples.shape[1])] + ['class', 'tile_name']
df = pd.DataFrame(np.hstack((training_samples, training_labels[:, np.newaxis], np.array(tile_names)[:, np.newaxis])), columns=columns)

# Save to CSV
csv_output_path = os.path.join(output_folder, 'training_data.csv')
df.to_csv(csv_output_path, index=False)

print(f"Training data exported to {csv_output_path}")

# Specify the file path where you want to save the CSV
csv_file = 'training_samples.csv'

# Save training_samples to CSV
np.savetxt(csv_file, training_samples, delimiter=',')

"""
# Calculate unique classes
unique_classes = np.unique(training_labels)

class_counts = Counter(training_labels)
for cls, count in class_counts.items():
    print(f"Class {cls}: {count} samples")

# Spectral investigation: Mean spectral signatures
unique_classes = np.unique(training_labels)
mean_spectra = {}

for cls in unique_classes:
    class_samples = training_samples[training_labels == cls]
    mean_spectra[cls] = np.mean(class_samples, axis=0)

"""
# Plot mean spectral signatures
plt.figure(figsize=(10, 6))
for cls, mean_spectrum in mean_spectra.items():
    plt.plot(mean_spectrum, label=f'Class {cls}')
plt.xlabel('Band')
plt.ylabel('Mean Reflectance')
plt.title('Mean Spectral Signatures')
plt.legend()
plt.show()
"""


# Spectral investigation: Mean spectral signatures with ±1 standard deviation
unique_classes = np.unique(training_labels)
mean_spectra = {}
std_spectra = {}

for cls in unique_classes:
    class_samples = training_samples[training_labels == cls]
    mean_spectra[cls] = np.mean(class_samples, axis=0)
    std_spectra[cls] = np.std(class_samples, axis=0)

# Plot mean spectral signatures with shading for ±1 standard deviation
plt.figure(figsize=(10, 6))
for cls in unique_classes:
    mean_spectrum = mean_spectra[cls]
    std_spectrum = std_spectra[cls]
    plt.plot(mean_spectrum, label=f'Class {cls}')
    plt.fill_between(range(len(mean_spectrum)), mean_spectrum - std_spectrum, mean_spectrum + std_spectrum, alpha=0.2)

plt.xlabel('Band')
plt.ylabel('Reflectance')
plt.title('Mean Spectral Signatures with ±1 Standard Deviation')
plt.legend()
plt.show()

"""
# Spectral investigation: Scatter plot for band 50 vs band 70
band_50 = 49  # Adjusted for zero-based index
band_70 = 69  # Adjusted for zero-based index

plt.figure(figsize=(8, 8))
for cls in unique_classes:
    class_samples = training_samples[training_labels == cls]
    plt.scatter(class_samples[:, band_50], class_samples[:, band_70], label=f'Class {cls}', alpha=0.5)
plt.xlabel('Band 50 Reflectance')
plt.ylabel('Band 70 Reflectance')
plt.title('Scatter Plot: Band 50 vs Band 70')
plt.legend()
plt.show()
"""