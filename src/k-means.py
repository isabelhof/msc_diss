import rasterio
import numpy as np
from sklearn.cluster import KMeans

# Load the raster data
with rasterio.open('subset-testing/tiled/spatial_subset_1_1.tif') as src:
    raster_data = src.read()

# Get the dimensions of the raster
bands, rows, cols = raster_data.shape

# Reshape the data to a 2D array (pixels x bands)
X = raster_data.reshape((bands, rows * cols)).T

# Define the number of clusters
n_clusters = 8

# Perform KMeans clustering
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans.fit(X)

# Get the labels for each pixel
labels = kmeans.labels_

# Reshape the labels to the original raster shape (rows x cols)
classified_image = labels.reshape((rows, cols))

# Define the output file path
output_file = 'subset-testing/tiled/spatial_subset_1_1.tif'

# Get the transform and metadata from the source raster
transform = src.transform
metadata = src.meta

# Update the metadata to reflect the number of layers
metadata.update({'count': 1, 'dtype': 'int32'})

# Save the classified image
with rasterio.open(output_file, 'w', **metadata) as dst:
    dst.write(classified_image.astype('int32'), 1)
