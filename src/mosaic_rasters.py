## STEP FOUR OF MERGING OUTPUTS

import os
import rasterio
from rasterio.merge import merge
import numpy as np

def mosaic_rasters(input_folder, output_path):
    # List of raster files
    raster_files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith('.tif')]
    
    if not raster_files:
        raise ValueError("No .tif files found in the input folder.")

    # Open raster files
    rasters = [rasterio.open(rf) for rf in raster_files]
    
    # Merge rasters
    mosaic, out_transform = merge(rasters)
    
    # Metadata for the output raster
    out_meta = rasters[0].meta.copy()
    out_meta.update({
        "driver": "GTiff",
        "count": 1,
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "transform": out_transform
    })
    
    # Write the mosaic to a new file
    with rasterio.open(output_path, "w", **out_meta) as dest:
        dest.write(mosaic[0], 1)
    
    # Close all raster files
    for raster in rasters:
        raster.close()
    
    print(f'Mosaic completed and saved to {output_path}.')

# Define input folder and output file path
input_folder = '/home/s1941095/scratch/msc_diss/data/outputs/RF_two_class/compressed'
output_path = '/home/s1941095/scratch/msc_diss/data/outputs/RF_two_class/mosaic-full-area.tif'

mosaic_rasters(input_folder, output_path)
