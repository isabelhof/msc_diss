from osgeo import gdal

# Open the hyperspectral image
dataset = gdal.Open('subset-testing/spatial_subset.tif')

# Specify the bands you want to extract (GDAL band indices start from 1)
bands = [75, 46, 19]

# Read the bands
band_data = []
for band in bands:
    band_data.append(dataset.GetRasterBand(band).ReadAsArray())

# Create the output image
driver = gdal.GetDriverByName('GTiff')
out_dataset = driver.Create('spatial_subset_rgb.tif', dataset.RasterXSize, dataset.RasterYSize, len(bands), gdal.GDT_Float32)

# Write the bands to the output image
for i, band in enumerate(band_data, 1):
    out_band = out_dataset.GetRasterBand(i)
    out_band.WriteArray(band)

# Set the georeference and projection if they exist
out_dataset.SetGeoTransform(dataset.GetGeoTransform())
out_dataset.SetProjection(dataset.GetProjection())

# Flush data to disk
out_dataset.FlushCache()

print("RGB image created successfully")
