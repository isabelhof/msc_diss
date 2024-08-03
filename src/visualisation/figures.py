#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()
sns.set_style("whitegrid")
sns.set_style("ticks")


training_classes = pd.read_csv('full_class_training_FINAL.csv')
two_classes = pd.read_csv('two_class_training_FINAL.csv')

print(two_classes)

# Extract wavelength headers (assuming they are in the column names)
wavelengths = training_classes.columns[2:]  # Skip the first two columns (class and label)
# Convert the wavelength headers to a numpy array
x_axis = wavelengths.to_numpy()

# Group by the 'label' column and calculate mean reflectance
grouped_df = training_classes.groupby('Label').mean()

# Drop the first two columns (class and label) to get only reflectance data
mean_reflectances = grouped_df.drop(columns=['Class'])
mean_reflectances_d = mean_reflectances / 10000

# Transpose the mean reflectances DataFrame
mean_reflectances_T = mean_reflectances_d.T

# Specify the order of the classes and define colors
ordered_classes = ['Rhododendron', 'Grass', 'Scrub', 'Tree', 'Ground', 'Built', 'Water', 'Shadow']
colors = ['#FF9999', '#99FF99', '#C2C287', '#66C266', '#FAC898', '#C39BD3', '#B3CDE0', '#D3D3D3']

# Plot the mean reflectance with shading for one standard deviation
plt.figure(figsize=(12, 6))
for class_name, color in zip(ordered_classes, colors):
    if class_name in mean_reflectances_T.columns:
        mean_values = mean_reflectances_T[class_name]
        plt.plot(x_axis, mean_values, label=f'{class_name}', color=color, linewidth= 2.5)
        
    
plt.xlabel('Wavelength (nm)')
plt.ylabel('Mean Reflectance')
plt.xticks(np.arange(0, len(x_axis), 10), x_axis[::10])
plt.yticks()
plt.legend(title='Class Label', ncol = 2)
plt.savefig('mean_reflectance_full.jpg', bbox_inches='tight', dpi=300)
plt.show()


# Extract wavelength headers (assuming they are in the column names)
wavelengths = training_classes.columns[2:]  # Skip the first two columns (class and label)
# Convert the wavelength headers to a numpy array
x_axis = wavelengths.to_numpy()

# Group by the 'label' column and calculate mean reflectance
grouped_df_mean = training_classes.groupby('Label').mean()
grouped_df_std = training_classes.groupby('Label').std()

# Drop the first two columns (class and label) to get only reflectance data
mean_reflectances = grouped_df_mean.drop(columns=['Class'])
std_reflectances = grouped_df_std.drop(columns=['Class'])
mean_reflectances_d = mean_reflectances / 10000
std_reflectances_d = std_reflectances / 10000

# Transpose the mean reflectances DataFrame
mean_reflectances_T = mean_reflectances_d.T
std_reflectances_T = std_reflectances_d.T

# Define the specific classes to include and their colors
selected_classes = ['Rhododendron', 'Scrub', 'Tree', 'Grass']
selected_colors = ['red', 'gold', 'darkviolet', 'lawngreen']

# Ensure that only the selected classes are included in the mean_reflectances_T DataFrame
mean_reflectances_T = mean_reflectances_T[selected_classes]

# Plot the mean reflectance
plt.figure(figsize=(12, 6))
for class_name, color in zip(selected_classes, selected_colors):
    if class_name in mean_reflectances_T.columns:
        mean_values = mean_reflectances_T[class_name]
        std_values = std_reflectances_T[class_name]
        plt.plot(x_axis, mean_values, label=f'{class_name}', color=color, linewidth = 2.5, alpha=0.7)
        plt.plot(x_axis, mean_values - std_values, linestyle='--', color=color, alpha=0.7, label=f'{class_name} ± SD')
        plt.plot(x_axis, mean_values + std_values, linestyle='--', color=color, alpha=0.7)
        

plt.xlabel('Wavelength (nm)')
plt.ylabel('Mean Reflectance')
plt.xticks(np.arange(0, len(x_axis), 10), [int(float(val)) for val in x_axis[::10]])
plt.yticks()
plt.legend(title='Class Label', ncol= 2)
plt.savefig('mean_reflectance_veg.jpg', bbox_inches='tight', dpi=300)
plt.show()


# In[5]:


# Extract wavelength headers (assuming they are in the column names)
wavelengths = training_classes.columns[2:]  # Skip the first two columns (class and label)
# Convert the wavelength headers to a numpy array
x_axis = wavelengths.to_numpy()

# Group by the 'label' column and calculate mean reflectance
grouped_df_mean = two_classes.groupby('Label').mean()
grouped_df_std = two_classes.groupby('Label').std()

# Drop the first two columns (class and label) to get only reflectance data
mean_reflectances = grouped_df_mean.drop(columns=['Class'])
std_reflectances = grouped_df_std.drop(columns=['Class'])
mean_reflectances_d = mean_reflectances / 10000
std_reflectances_d = std_reflectances / 10000

# Transpose the mean reflectances DataFrame
mean_reflectances_T = mean_reflectances_d.T
std_reflectances_T = std_reflectances_d.T

# Specify the order of the classes and define colors
ordered_classes = ['Rhodie', 'Not Rhodie']
colors = ['#FF9999', '#D3D3D3']

# Create a mapping for legend labels
class_name_mapping = {
    'Rhodie': 'Rhododendron',
    'Not Rhodie': 'Not rhododendron'
}

# Plot the mean reflectance with shading for one standard deviation
plt.figure(figsize=(12, 6))
for class_name, color in zip(ordered_classes, colors):
    if class_name in mean_reflectances_T.columns:
        mean_values = mean_reflectances_T[class_name]
        std_values = std_reflectances_T[class_name]
        plt.plot(x_axis, mean_values, label=f'{class_name_mapping[class_name]}', color=color, linewidth= 2.5)
        plt.plot(x_axis, mean_values - std_values, linestyle='--', color=color, label=f'{class_name_mapping[class_name]} ± SD', alpha=0.7, )
        plt.plot(x_axis, mean_values + std_values, linestyle='--', color=color, alpha=0.7)
        plt.fill_between(x_axis, mean_values - std_values, mean_values + std_values, color=color, alpha=0.1)
        
    
plt.xlabel('Wavelength (nm)')
plt.ylabel('Mean Reflectance')
plt.xticks(np.arange(0, len(x_axis), 10), x_axis[::10])
plt.yticks()
plt.legend(title='Class Label')
plt.savefig('mean_reflectance_2_class.jpg', bbox_inches='tight', dpi=300)
plt.show()


"""

pixels = [3778, 4012, 3757, 4642, 2956, 2530, 1148, 1872]
labels = ['Rhododendron', 'Grass', 'Scrub', 'Tree', 'Bare ground', 'Built-up area', 'Water', 'Shadow']

"""

# Data for classes
labels_class = ['Rhododendron', 'Grass', 'Scrub', 'Tree', 'Bare ground', 'Built-up area', 'Water', 'Shadow']
sizes_class = [3392, 3343, 2685, 4642, 1483, 1806, 1148, 1872]
colors_class = ['#e45050', '#d5d5d5', '#b2b2b2', '#d5d5d5', '#b2b2b2', '#d5d5d5', '#b2b2b2', '#d5d5d5']

# Data for survey locations
labels_survey = ['Survey 1', 'Survey 2', 'Survey 3']
sizes_survey = [3392, 0, 0, 940, 1140, 1263, 1085, 685, 915, 2835, 1342, 465, 524, 494, 465, 400, 1006, 400, 0, 0, 1148, 426, 0, 1446]
colors_survey = [
    '#537bff', '#3da037', '#ff9939',  # Rhododendron (Survey 1, 2, 3)
    '#537bff', '#3da037', '#ff9939',  # Grass
    '#537bff', '#3da037', '#ff9939',  # Scrub
    '#537bff', '#3da037', '#ff9939',  # Tree
    '#537bff', '#3da037', '#ff9939',  # Bare ground
    '#537bff', '#3da037', '#ff9939',  # Built-up area
    '#537bff', '#3da037', '#ff9939',  # Water
    '#537bff', '#3da037', '#ff9939'   # Shadow
]

# Plot the outer pie (Classes)
plt.pie(sizes_class, labels=labels_class, colors=colors_class, startangle=90, frame=True)

# Plot the inner pie (Survey locations)
plt.pie(sizes_survey, colors=colors_survey, radius=0.75, startangle=90)

# Draw center circle for a donut chart
centre_circle = plt.Circle((0, 0), 0.5, color='black', fc='white', linewidth=0)
fig = plt.gcf()
fig.gca().add_artist(centre_circle)

# Create a legend for the survey locations
survey_patches = [plt.Line2D([0], [0], marker='o', color='w', label='Survey 1', markersize=10, markerfacecolor='#537bff'),
                  plt.Line2D([0], [0], marker='o', color='w', label='Survey 2', markersize=10, markerfacecolor='#3da037'),
                  plt.Line2D([0], [0], marker='o', color='w', label='Survey 3', markersize=10, markerfacecolor='#ff9939')]

# Add legend at the bottom
plt.legend(handles=survey_patches, title="Survey distribution", loc="lower center", bbox_to_anchor=(1, 0))

# Ensure aspect ratio is equal to make the pie chart a circle
plt.axis('equal')
plt.tight_layout()
plt.show()





