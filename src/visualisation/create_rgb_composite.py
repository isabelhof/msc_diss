import os
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

# input and output files
input_file = '/home/s1941095/scratch/dissfinal/raw/HBNX-1-20230616/HBNX-1-20230616-2_14_VNIR_1800_SN00888_quac_specPol_rectGeRot.img'
output_file = '/home/s1941095/scratch/dissfinal/data/rgb_2_14.tif'

# bands to extract for RGB
bands = [75, 46, 19]

# process the input .img file
extract_rgb(input_file, output_file, bands)

print("Processing complete.")
