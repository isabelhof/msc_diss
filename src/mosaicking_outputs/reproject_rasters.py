## STEP ONE OF MERGING OUTPUTS

import os
import subprocess

# Define input and output directories
input_dir = "data/outputs/RF_full_class/survey-1"
output_dir = "data/outputs/RF_full_class/reprojected/survey-1"

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Loop over all .img files in the input directory
for filename in os.listdir(input_dir):
    if filename.endswith(".img"):
        input_file = os.path.join(input_dir, filename)
        output_file = os.path.join(output_dir, filename)

        # Use gdalwarp to reproject the file
        command = [
            "gdalwarp",
            "-s_srs", "EPSG:27700",
            "-t_srs", "EPSG:27700",
            input_file,
            output_file
        ]

        # Execute the command
        result = subprocess.run(command, capture_output=True, text=True)

        # Check if gdalwarp was successful
        if result.returncode == 0:
            print(f"Successfully reprojected: {filename}")
        else:
            print(f"Error reprojecting: {filename}")
            print(result.stderr)
