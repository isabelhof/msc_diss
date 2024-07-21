import pandas as pd
import numpy as np
from osgeo import gdal
import os
import glob

# Function to read raster data
def read_raster(raster_file):
    dataset = gdal.Open(raster_file, gdal.GA_ReadOnly)
    band = dataset.GetRasterBand(1)
    array = band.ReadAsArray()
    return array, dataset

# Function to read mean spectra from CSV
def read_mean_spectra_from_csv(csv_file):
    # Read CSV into pandas DataFrame
    df = pd.read_csv(csv_file)
    
    # Extract class labels
    class_labels = df['class'].values
    
    # Extract mean spectra (excluding 'class' column)
    mean_spectra = df.drop(columns=['class']).values
    
    return class_labels, mean_spectra

# Function to calculate spectral angle
def spectral_angle_mapper(raster_array, reference_spectrum):
    # Ensure both arrays have compatible shapes
    assert raster_array.shape == reference_spectrum.shape, f"Shape mismatch between raster_array {raster_array.shape} and reference_spectrum {reference_spectrum.shape}"
    
    # Calculate dot product
    dot_product = np.sum(raster_array * reference_spectrum)
    
    # Calculate norms
    norm_raster = np.linalg.norm(raster_array)
    norm_ref = np.linalg.norm(reference_spectrum)
    
    # Calculate spectral angle
    spectral_angle = np.arccos(dot_product / (norm_raster * norm_ref))
    
    return spectral_angle

# Function to perform SAM classification
def sam_classification(raster_file, class_labels, reference_spectra):
    # Read raster data
    raster_array, dataset = read_raster(raster_file)
    rows, cols = raster_array.shape
    
    # Initialize output array
    output_array = np.zeros_like(raster_array, dtype=np.uint8)  # Assuming class labels are integers
    
    # Iterate over each pixel
    for i in range(rows):
        for j in range(cols):
            min_angle = float('inf')
            classified_class = None
            
            # Compare spectral angle for each class
            for idx, class_label in enumerate(class_labels):
                reference_spectrum = reference_spectra[idx]
                
                # Ensure shapes match before computation
                assert raster_array.shape == reference_spectrum.shape, f"Shape mismatch between raster_array {raster_array.shape} and reference_spectrum {reference_spectrum.shape}"
                
                angle = spectral_angle_mapper(raster_array[i, j], reference_spectrum)
                
                # Find the minimum angle
                if angle < min_angle:
                    min_angle = angle
                    classified_class = class_label
            
            # Assign the class label with the minimum angle to the output array
            output_array[i, j] = classified_class
    
    return output_array, dataset

# Function to save classified raster to disk
def save_classified_raster(classified_array, dataset, output_filename):
    driver = gdal.GetDriverByName('GTiff')
    dataset_out = driver.Create(output_filename, classified_array.shape[1], classified_array.shape[0], 1, gdal.GDT_Byte)
    dataset_out.SetGeoTransform(dataset.GetGeoTransform())
    dataset_out.SetProjection(dataset.GetProjection())
    dataset_out.GetRasterBand(1).WriteArray(classified_array)
    dataset_out.FlushCache()
    dataset_out = None


if __name__ == '__main__':
    csv_file = 'subset-testing/training_band_data.csv'
    raster_folder = 'subset-testing/tiled'
    output_folder = 'subset-testing/classified/sam'
    
    # Read mean spectra from CSV
    class_labels, reference_spectra = read_mean_spectra_from_csv(csv_file)
    
    # Process each raster file in the folder
    for raster_file in glob.glob(os.path.join(raster_folder, '*.tif')):
        # Perform SAM classification
        classified_array, dataset = sam_classification(raster_file, class_labels, reference_spectra)
        
        # Save classified raster to disk
        output_filename = os.path.join(output_folder, os.path.basename(raster_file).replace('.tif', '_classified.tif'))
        save_classified_raster(classified_array, dataset, output_filename)
    
    print("SAM classification complete.")
