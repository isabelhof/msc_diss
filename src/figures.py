import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file into a DataFrame
df1 = pd.read_csv('data/testing/training-data/subset-0616-spectra-cleaned.csv')
df2 = pd.read_csv('data/testing/training-data/subset-0831-spectra-cleaned.csv')
df3 = pd.read_csv('data/testing/training-data/subset-0902-spectra-cleaned.csv')

# Group by the class column and calculate the mean for each band
mean_reflectance1 = df1.groupby('Class').mean()
std_reflectance1 = df1.groupby('Class').std()
mean_reflectance2 = df2.groupby('Class').mean()
std_reflectance2 = df2.groupby('Class').std()
mean_reflectance3 = df3.groupby('Class').mean()
std_reflectance3 = df3.groupby('Class').std()

# Transpose the DataFrame for easier plotting
mean_reflectance1_T = mean_reflectance1.T
std_reflectance1_T = std_reflectance1.T
mean_reflectance2_T = mean_reflectance2.T
std_reflectance2_T = std_reflectance2.T
mean_reflectance3_T = mean_reflectance3.T
std_reflectance3_T = std_reflectance3.T

# Filter DataFrames to only include the "grass" class
df1_grass = df1[df1['Class'] == 'scrub']
df2_grass = df2[df2['Class'] == 'scrub']
df3_grass = df3[df3['Class'] == 'scrub']

# Calculate the mean and standard deviation for the "grass" class, excluding the 'class' column
mean_grass1 = df1_grass.loc[:, df1_grass.columns != 'Class'].mean()
std_grass1 = df1_grass.loc[:, df1_grass.columns != 'Class'].std()

mean_grass2 = df2_grass.loc[:, df2_grass.columns != 'Class'].mean()
std_grass2 = df2_grass.loc[:, df2_grass.columns != 'Class'].std()

mean_grass3 = df3_grass.loc[:, df3_grass.columns != 'Class'].mean()
std_grass3 = df3_grass.loc[:, df3_grass.columns != 'Class'].std()

# Plot the mean reflectance with shading for each CSV file
plt.figure(figsize=(12, 6))

# Plot for file1
plt.plot(mean_grass1.index, mean_grass1, label='Scrub - File 1')
plt.fill_between(mean_grass1.index, mean_grass1 - std_grass1, mean_grass1 + std_grass1, alpha=0.3)

# Plot for file2
plt.plot(mean_grass2.index, mean_grass2, label='Scrub - File 2')
plt.fill_between(mean_grass2.index, mean_grass2 - std_grass2, mean_grass2 + std_grass2, alpha=0.3)

# Plot for file3
plt.plot(mean_grass3.index, mean_grass3, label='Scrub - File 3')
plt.fill_between(mean_grass3.index, mean_grass3 - std_grass3, mean_grass3 + std_grass3, alpha=0.3)

plt.xlabel('Band')
plt.ylabel('Reflectance')
plt.title('Mean Reflectance with Standard Deviation for Grass Class Across Three Files')
plt.legend()
plt.show()

"""

# Plot the mean reflectance with shading for one standard deviation
plt.figure(figsize=(12, 6))
for class_name in mean_reflectance1_T.columns:
    mean_values = mean_reflectance1_T[class_name]
    std_values = std_reflectance1_T[class_name]
    
    plt.plot(mean_values.index, mean_values, label=f'Class {class_name}')
    plt.fill_between(mean_values.index, mean_values - std_values, mean_values + std_values, alpha=0.3)

plt.xlabel('Band')
plt.ylabel('Reflectance')
plt.title('Mean Reflectance with Standard Deviation for Each Class Across All Bands (0902)')
plt.legend()
plt.show()
"""