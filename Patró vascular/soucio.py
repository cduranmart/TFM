# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 10:47:25 2024

@author: cduran
"""

import warnings
warnings.filterwarnings("ignore")
import numpy as np
#import pandas as pd
import tensorflow as tf
import tensorflow.keras.layers as L
import tensorflow_addons as tfa
import glob, random, os, warnings
import matplotlib.pyplot as plt
import skimage
import cv2
from patchify import patchify, unpatchify
from skimage.transform import resize
from skimage.io import imread
from skimage.io import imsave
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from skimage.morphology import skeletonize
from skimage.morphology import dilation, square
from skan import Skeleton, summarize

#generar patro bascular skeletonitizat a partir de la predicció de la CNN

image=imread('vessel_prediction.jpg') #la procesamos directamente con la función 
# Comprobar si la imagen tiene tres canales (RGB)
plt.imshow(image, cmap='gray')


if image.ndim == 3 and image.shape[2] == 3:
    # Convertir a escala de grises si es necesario
    from skimage.color import rgb2gray
    gray_image = rgb2gray(image)
else:
    # Si la imagen ya está en escala de grises o es binaria, continuar
    gray_image = image

# Umbralizar la imagen si no está ya binarizada
if gray_image.max() > 1:
    thresh = threshold_otsu(gray_image)
    binary_image = gray_image > thresh
else:
    binary_image = gray_image
    
plt.imshow(binary_image, cmap='gray')

# Esqueletonizar
skeleton = skeletonize(binary_image)
skeleton_obj=Skeleton(skeleton)

summary = summarize(skeleton_obj)
summary['tortuosity'] = summary['branch-distance'] / summary['euclidean-distance']
tortuosity_threshold=0
summary_filtered= summary[summary['tortuosity'] > tortuosity_threshold]

# Assuming skeleton_obj is already created as shown in your question
num_paths = skeleton_obj.n_paths  

# Create a blank image for reconstruction, same size as the original binary_image
reconstructed_image = np.zeros_like(binary_image)

for path in summary['skeleton-id']:
    path_coords = skeleton_obj.path_coordinates(path)
    for point in path_coords:
        # Draw each point on the reconstructed_image; adjust drawing method as needed
        # This is a simple way; for better visualization, you might draw lines between consecutive points
        reconstructed_image[int(point[0]), int(point[1])] = 255  # Assuming white on black for the visualization

plt.imshow(reconstructed_image, cmap='gray')
plt.title('Reconstructed Image from Skeleton Paths')
plt.show()



# Assuming skeleton_obj is already created as shown in your question
num_paths = skeleton_obj.n_paths()  # This is hypothetical; adjust based on your actual object's methods

# Create a blank image for reconstruction, same size as the original binary_image
reconstructed_image = np.zeros_like(binary_image)

for index in range(num_paths):
    path_coords = skeleton_obj.path_coordinates(index)
    for point in path_coords:
        # Draw each point on the reconstructed_image; adjust drawing method as needed
        # This is a simple way; for better visualization, you might draw lines between consecutive points
        reconstructed_image[int(point[0]), int(point[1])] = 255  # Assuming white on black for the visualization

plt.imshow(reconstructed_image, cmap='gray')
plt.title('Reconstructed Image from Skeleton Paths')
plt.show()



# Initialize reconstructed_image with the correct type
reconstructed_image = np.zeros_like(binary_image, dtype=np.uint8)

for path in summary_filtered['skeleton-id']:  # Use the filtered summary
    path_coords = skeleton_obj.path_coordinates(path)
    
    # Check if path_coords is not empty
    if len(path_coords) > 0:
        # Draw lines between points for better visualization
        for i in range(len(path_coords) - 1):
            cv2.line(reconstructed_image,
                     pt1=(int(path_coords[i][1]), int(path_coords[i][0])),  # x, y coordinates for point 1
                     pt2=(int(path_coords[i+1][1]), int(path_coords[i+1][0])),  # x, y coordinates for point 2
                     color=(255),  # Color value for a white line
                     thickness=1)  # Line thickness

plt.imshow(reconstructed_image, cmap='gray')
plt.title('Reconstructed Image from Skeleton Paths with Tortuosity Filter')
plt.show()


for path in summary_filtered['skeleton-id']:  # Use the filtered summary
    path_coords = skeleton_obj.path_coordinates(path)
    print(path_coords)

path_coords = skeleton_obj.path_coordinates(0)
print(path_coords)


reconstructed_image = np.zeros_like(binary_image, dtype=np.uint8)

path_coords = skeleton_obj.path_coordinates(0)  # Example for the first path

# Check if path_coords is not empty
if len(path_coords) > 0:
    # Draw lines between points for better visualization
    for i in range(len(path_coords) - 1):
        # Note the reversal of order for x and y coordinates
        start_point = (path_coords[i][1], path_coords[i][0])
        end_point = (path_coords[i+1][1], path_coords[i+1][0])

        # Draw line on the reconstructed_image
        cv2.line(reconstructed_image, pt1=start_point, pt2=end_point, color=255, thickness=1)

# Display the image
plt.imshow(reconstructed_image, cmap='gray')
plt.title('Reconstructed Path from Skeleton')
plt.show()


reconstructed_image = np.zeros_like(binary_image, dtype=np.uint8)

# Use either summary or summary_filtered depending on whether you want all paths or filtered paths
for path_id in summary['skeleton-id']:  # Replace with summary_filtered if necessary
    path_coords = skeleton_obj.path_coordinates(path_id)

    # Check if path_coords is not empty
    if len(path_coords) > 0:
        # Draw lines between points for better visualization
        for i in range(len(path_coords) - 1):
            start_point = (path_coords[i][1], path_coords[i][0])
            end_point = (path_coords[i+1][1], path_coords[i+1][0])

            # Draw line on the reconstructed_image
            cv2.line(reconstructed_image, pt1=start_point, pt2=end_point, color=255, thickness=1)

# Display the image
plt.imshow(reconstructed_image, cmap='gray')
plt.title('Reconstructed Image from Skeleton Paths')
plt.show()



# Assuming binary_image is already defined and is of type np.uint8
# If not, you would need to make sure it is or cast it to np.uint8
reconstructed_image = np.zeros_like(binary_image, dtype=np.uint8)

for path_id in range(skeleton_obj.n_paths):  # Iterate over the number of paths
    path_coords = skeleton_obj.path_coordinates(path_id)

    print(f"Drawing path {path_id} with {len(path_coords)} points.")  # Debug output

    # Check if path_coords is not empty
    if len(path_coords) > 0:
        # Draw lines between points for better visualization
        for i in range(len(path_coords) - 1):
            start_point = (path_coords[i][1], path_coords[i][0])
            end_point = (path_coords[i+1][1], path_coords[i+1][0])

            # Draw line on the reconstructed_image
            cv2.line(reconstructed_image, pt1=start_point, pt2=end_point, color=255, thickness=1)

# Display the image after all paths are drawn
plt.imshow(reconstructed_image, cmap='gray')
plt.title('Reconstructed Image from Skeleton Paths')
plt.show()



reconstructed_image = np.zeros_like(binary_image, dtype=np.uint8)

for idx, row in summary_filtered.iterrows():
    path_coords = skeleton_obj.path_coordinates(idx)

    # Draw the path if coordinates are available
    if len(path_coords) > 0:
        for i in range(len(path_coords) - 1):
            start_point = (int(path_coords[i][1]), int(path_coords[i][0]))
            end_point = (int(path_coords[i+1][1]), int(path_coords[i+0][0]))
            cv2.line(reconstructed_image, start_point, end_point, color=255, thickness=3)

# Display the image
plt.imshow(reconstructed_image, cmap='gray')
plt.title('Reconstructed Image from Filtered Paths')
plt.show()
