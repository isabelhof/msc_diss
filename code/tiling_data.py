# tiling the flight lines into more manageable sizes to speed up processing using the create_tiles function

import os
import subprocess

def create_tiles(input_dir, output_dir, tile_size):
    # for each file in the input directory, define the path to the file and the pattern for naming the output file
    for input_file in os.listdir(input_dir):
        if input_file.endswith('.tif'):
            input_path = os.path.join(input_dir, input_file)
            # label the output with the tile row and column, e.g. tile_1_2.tif or tile_2_4.tif
            output_pattern = os.path.join(output_dir, f"{os.path.splitext(input_file)[0]}_tile_%04d_%04d.tif")
            # use gdal to tile, specifying tile size, directory, and apply a compression
            subprocess.call([
                'gdal_retile.py',
                '-ps', str(tile_size), str(tile_size),
                '-targetDir', output_dir,
                '-co', 'COMPRESS=LZW',
                input_path,
                '-of', 'GTiff'
            ])

# define variables
input_dir = 'subset-testing/mnf'
output_dir = 'subset-testing/mnf/tiled'
tile_size = 1000 

# run
create_tiles(input_dir, output_dir, tile_size)
