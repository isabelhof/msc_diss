## STEP TWO OF MERGING OUTPUTS

import os
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject
import numpy as np

def resample_raster(input_path, output_path, target_resolution):
    with rasterio.open(input_path) as src:
        # Calculate new dimensions and transform
        transform, new_width, new_height = calculate_default_transform(
            src.crs, src.crs, src.width, src.height, *src.bounds, resolution=target_resolution
        )
        
        # Update metadata
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': src.crs,
            'transform': transform,
            'width': new_width,
            'height': new_height,
            'dtype': 'uint8'  # Use 'uint16' or other types if needed
        })
        
        # Create a new raster with the resampled data
        with rasterio.open(output_path, 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                data = src.read(i)
                resampled_data = np.empty((new_height, new_width), dtype=np.uint8)
                reproject(
                    source=data,
                    destination=resampled_data,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=src.crs,
                    resampling=Resampling.nearest
                )
                dst.write(resampled_data, i)

def resample_folder(input_folder, output_folder, target_resolution):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for filename in os.listdir(input_folder):
        if filename.endswith('.img'):
            input_path = os.path.join(input_folder, filename)
            output_filename = os.path.splitext(filename)[0] + '.tif'
            output_path = os.path.join(output_folder, output_filename)
            print(f'Resampling {input_path} to {output_path}')
            resample_raster(input_path, output_path, target_resolution)
    
    print('Resampling completed.')

# Define input and output folders and target resolution
input_folder = '/home/s1941095/scratch/msc_diss/data/outputs/RF_full_class/reprojected/flightline-3'
output_folder = '/home/s1941095/scratch/msc_diss/data/outputs/RF_full_class/resampled/flightline-3'
target_resolution = 0.35  # Target resolution in meters

resample_folder(input_folder, output_folder, target_resolution)
