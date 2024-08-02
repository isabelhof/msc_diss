from osgeo import gdal
import sys

def compress_tif_with_lzw(input_file, output_file):
    try:
        # Open the input TIFF file
        dataset = gdal.Open(input_file, gdal.GA_ReadOnly)
        
        if not dataset:
            raise FileNotFoundError(f"Cannot open file: {input_file}")
        
        # Get the image metadata
        driver = gdal.GetDriverByName('GTiff')
        if not driver:
            raise RuntimeError("GTiff driver is not available")

        # Define options for LZW compression
        options = ['COMPRESS=LZW']

        # Create a new dataset with LZW compression
        out_dataset = driver.CreateCopy(output_file, dataset, options=options)
        
        if not out_dataset:
            raise RuntimeError(f"Failed to create output file: {output_file}")

        # Cleanup
        dataset = None
        out_dataset = None
        
        print(f"File compressed successfully: {output_file}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python gdal_compresssion.py input_file output_file")
    else:
        input_file = sys.argv[1]
        output_file = sys.argv[2]
        compress_tif_with_lzw(input_file, output_file)
