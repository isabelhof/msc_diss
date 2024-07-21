import os
import rasterio

# Path to the folder containing .img files
folder_path = '/home/s1941095/scratch/msc_diss/data/hs-tiled/flightline-3'

# Desired projection
desired_projection = 'EPSG:27700'

# Iterate through all files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith('.tif'):
        filepath = os.path.join(folder_path, filename)
        with rasterio.open(filepath) as src:
            # Get the CRS (Coordinate Reference System)
            crs = src.crs
            if crs != desired_projection:
                print(f"File '{filename}' has projection {crs}, which is NOT {desired_projection}")
