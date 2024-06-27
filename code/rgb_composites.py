import os
import glob
from osgeo import gdal

# function to extract specific bands and create an RGB image
def extract_rgb(input_file, output_file, bands):
    # open the input file
    dataset = gdal.Open(input_file)
    band_data = []
    no_data_values = []
    
    # read the specific bands
    for band in bands:
        band_obj = dataset.GetRasterBand(band)
        band_data.append(band_obj.ReadAsArray())
        no_data_values.append(band_obj.GetNoDataValue())
    
    driver = gdal.GetDriverByName('GTiff')
    out_dataset = driver.Create(output_file, dataset.RasterXSize, dataset.RasterYSize, len(bands), gdal.GDT_Float32)
    
    # write to new image
    for i, (band, no_data) in enumerate(zip(band_data, no_data_values), 1):
        out_band = out_dataset.GetRasterBand(i)
        out_band.WriteArray(band)
        if no_data is not None:
            out_band.SetNoDataValue(no_data)
    
    out_dataset.SetGeoTransform(dataset.GetGeoTransform())
    out_dataset.SetProjection(dataset.GetProjection())
    out_dataset.FlushCache()
    
    print(f"RGB image created successfully: {output_file}")

# directory containing the input files
input_dir = 'testing-data'
output_dir = 'RGB-images'

# ensure the output directory exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# bands to extract for RGB
bands = [75, 46, 19]

# process each .i,g file in the input directory
for input_file in glob.glob(os.path.join(input_dir, '*.img')):
    filename = os.path.basename(input_file)
    output_file = os.path.join(output_dir, filename.replace('_VNIR_1800_SN00888_quac_specPol_rectGeRot.img', '_RGB.tif'))
    extract_rgb(input_file, output_file, bands)

print("Processing complete.")

