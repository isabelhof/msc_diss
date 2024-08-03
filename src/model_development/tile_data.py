# create 1000 x 1000 pixel tiles from the original full-band hs data
# repeat for each of the three periods of aerial data collection

# import packages
import os
import subprocess

def create_tiles(input_dir, output_dir, tile_size):
    # generate a list of all .tif files in the directory
    img_files = [f for f in os.listdir(input_dir) if f.endswith('.img')]

    # tile each tif tile using gdal
    for input_file in img_files:
        input_path = os.path.join(input_dir, input_file)
        # label the output with the tile row and column, e.g., tile_1_2.tif or tile_2_4.tif
        output_pattern = os.path.join(output_dir, f"{os.path.splitext(input_file)[0]}_tile_%04d_%04d.tif")
        # use gdal to tile, specifying tile size, directory, and apply a compression
        subprocess.call([
            'gdal_retile.py',
            '-ps', str(tile_size), str(tile_size),
            '-targetDir', output_dir,
            '-s_srs', 'EPSG:27700',
            '-co', 'COMPRESS=LZW',
            input_path,
            '-of', 'GTiff'
        ])

# define variables
input_dir = '/home/s1941095/ds_s1941095/msc_dissertation/testing-data'
output_dir = '/home/s1941095/scratch/msc_diss/data/hs-tiled/flightline-1'
tile_size = 1000 

# run the create_tiles function
create_tiles(input_dir, output_dir, tile_size)

