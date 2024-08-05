import os
import numpy as np
import rasterio

# Define input and output directories
input_folder = '/home/s1941095/scratch/msc_diss/data/outputs/RF_full_class/resampled/survey-3'
output_folder = '/home/s1941095/scratch/msc_diss/data/outputs/RF_full_class/masked/survey-3'

# Ensure output folder exists
os.makedirs(output_folder, exist_ok=True)

# Iterate through all .tif files in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith('.tif'):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)
        
        with rasterio.open(input_path) as src:
            # Read the raster data
            data = src.read(1)  # Read the first band

            # Mask pixels with value 8 as nodata (0)
            data_masked = np.where(data == 8, 0, data)
            
            # Update metadata to reflect new nodata value
            meta = src.meta.copy()
            meta.update(dtype=rasterio.uint8, nodata=0)

            # Write the masked raster to the output folder
            with rasterio.open(output_path, 'w', **meta) as dst:
                dst.write(data_masked, 1)

print('Masking complete. Check the output folder for results.')
