import os
import sys
import subprocess

def merge_tifs(input_folder, output_file, nodata_value=0):
    """
    Merges GeoTIFF files in a specified folder using gdal_merge.py with LZW compression.

    Parameters:
    input_folder (str): The directory containing input GeoTIFF files.
    output_file (str): The path for the output merged GeoTIFF file.
    nodata_value (int or float, optional): The value to treat as no-data. Default is 0.

    Raises:
    FileNotFoundError: If no GeoTIFF files are found in the input folder.
    RuntimeError: If the gdal_merge.py command fails.
    """
    # Gather all .tif files from the input folder
    tiff_files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith('.tif')]
    
    if not tiff_files:
        raise FileNotFoundError("No .tif files found in the input folder.")

    # Prepare the gdal_merge.py command
    command = [
        "gdal_merge.py", 
        "-o", output_file, 
        "-n", str(nodata_value), 
        "-a_nodata", str(nodata_value), 
        "-ot", "Byte", 
        "-co", "COMPRESS=LZW"
    ] + tiff_files

    # Execute the command
    try:
        subprocess.run(command, check=True)
        print(f"Merged file created successfully: {output_file}")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"gdal_merge.py failed: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python merge_tifs.py input_folder output_file")
    else:
        input_folder = sys.argv[1]
        output_file = sys.argv[2]
        merge_tifs(input_folder, output_file)

