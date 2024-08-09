# MSc Dissertation: Monitoring Invasive Non-Native Plant Species using Hyperspectral Remote Sensing Data: A Case Study on the West Highland Way in Scotland

## Overview

This GitHub repository contains the code developed for my MSc dissertation, which aims to detect the invasive plant species rhododendron using hyperspectral remote sensing data. The repository is organized into three main folders within the `src` directory: `model_development`, `mosaicking_outputs`, and `visualisation`. Each folder contains specific Python scripts to train the models, process the output classifications, and visualise the results.

## Repository Structure

### 1. `model_development/`

This folder contains the scripts used to develop and train the machine learning models for rhododendron detection. The models utilised are Random Forest (RF) and Support Vector Machine (SVM).

- **`hyperparameter_tuning.py`**: This script is used for tuning the hyperparameters of the Random Forest model. It employs the RandomisedSearchCV function to find the optimal parameters.

- **`svm_classifier.py`**: This script trains a Support Vector Machine (SVM) classifier and applies the trained model to the hyperspectral data to make predictions.

- **`rf_classifier.py`**: Similar to the SVM classifier script, this file is responsible for training a Random Forest (RF) classifier on the data and making predictions.

- **`tile_data.py`**: This script tiles the input hyperspectral data into smaller, more manageable chunks, which can be used for training data collection or prediction processes.

### 2. `mosaicking_outputs/`

This folder includes scripts for processing and manipulating classification outputs.

- **`compress_rasters.py`**: This script compresses the raster files to reduce their size for storage and processing efficiency.

- **`mosaic_rasters.py`**: This script mosaics individual raster files into a single, continuous raster. This is useful for creating a cohesive output after conducting the classification.

- **`reclassify_rasters.py`**: This script reclassifies the classified raster outputs into specific classes and is used for masking shadow or merging non-rhododnedron classes to highlight presence.

- **`resample_rasters.py`**: This script resamples the raster data to a shared spatial resolution.

- **`reproject_rasters.py`**: This script reprojects the raster data to a different coordinate reference system (CRS) to ensure consistency across datasets.

### 3. `visualisation/`

This folder contains scripts for visualizing the hyperspectral data and the results of the classification models.

- **`create_rgb_composites.py`**: This script generates RGB composites from the hyperspectral data.

- **`figures.py`**: This script creates various figures to investigate the spectral separability of the training data. These figures are useful for understanding the spectral characteristics of Rhododendron and other land cover types.

## Usage

To use the scripts in this repository, follow these general steps:

1. **Data Preparation**: (optional) Use `tile_data.py` to tile the hyperspectral data into smaller segments.
2. **Model Training**: Run `hyperparameter_tuning.py` to optimise the Random Forest model parameters. Then, use `svm_classifier.py` or `rf_classifier.py` to train and apply the SVM or RF classifiers.
3. **Raster Processing**: After classification, use the scripts in the `mosaicking_outputs/` folder to process the resulting raster outputs as needed (e.g., reprojecting, resampling, compressing, mosaicking).
4. **Visualization**: Generate RGB composites using `create_rgb_composites.py` and create figures with `figures.py` to analyze and present the results.

## Contact

For any questions or further information, please contact Isabel Hofmockel at isabelhofmockel@gmail.com.
