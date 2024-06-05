#!/usr/bin/env python
# coding: utf-8

# https://github.com/arkanivasarkar/Retinal-Vessel-Segmentation-using-variants-of-UNET

# ## Carregar model

# In[1]:


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


# ## Model de predicció del patró vascular

# In[2]:


def IoU_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + 1.0)

def IoU_loss(y_true, y_pred):
    return -IoU_coef(y_true, y_pred)


# In[3]:


def clahe_equalized(imgs):    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))    
    imgs_equalized = clahe.apply(imgs)
    return imgs_equalized


# In[4]:


import pickle
from keras.models import load_model

filename = 'retina_Unet_150epochs'
model = load_model(filename+'.hdf5', custom_objects={'IoU_loss': IoU_loss, 'IoU_coef': IoU_coef})
        


# In[5]:


# function to resize image to the next multiple of patch_size
def resize_to_fit(image, target_size):
    height, width = image.shape[:2]
    new_height = ((height // target_size) + 1) * target_size
    new_width = ((width // target_size) + 1) * target_size
    resized_image = cv2.resize(image, (new_width, new_height))
    return resized_image


# In[6]:


#funció per retornat el patró vascular 
def vessel_pattern(imgpath):
    reconstructed_image = []
    test_img = skimage.io.imread(imgpath) #test image
    test_img = resize_to_fit(test_img, 512)
    patch_size = 512

    predicted_patches = []  

    test = test_img[:,:,1] #selecting green channel
    test = clahe_equalized(test) #applying CLAHE
    SIZE_X = (test_img.shape[1]//patch_size)*patch_size #getting size multiple of patch size
    SIZE_Y = (test_img.shape[0]//patch_size)*patch_size #getting size multiple of patch size
    test = cv2.resize(test, (SIZE_X, SIZE_Y))        
    test = np.array(test)
    patches = patchify(test, (patch_size, patch_size), step=patch_size) #create patches(patch_sizexpatch_sizex1)

    for i in range(patches.shape[0]):
        for j in range(patches.shape[1]):
            single_patch = patches[i,j,:,:]
            single_patch_norm = (single_patch.astype('float32')) / 255.
            single_patch_norm = np.expand_dims(np.array(single_patch_norm), axis=-1)
            single_patch_input = np.expand_dims(single_patch_norm, 0)
            single_patch_prediction = (model.predict(single_patch_input)[0,:,:,0] > 0.5).astype(np.uint8) #predict on single patch
            predicted_patches.append(single_patch_prediction)
    predicted_patches = np.array(predicted_patches)
    predicted_patches_reshaped = np.reshape(predicted_patches, (patches.shape[0], patches.shape[1], patch_size,patch_size) )
    reconstructed_image = unpatchify(predicted_patches_reshaped, test.shape) #join patches to form whole img
    return(reconstructed_image)


# ## Us de la biblioteca skan per a trobar turtuositats al patró vascular

# In[7]:


#encapsular-ho en una funció
def vessel_pattern_skeletonized(imgpath):
    image=vessel_pattern(imgpath) #la procesamos directamente con la función 
    # Comprobar si la imagen tiene tres canales (RGB)
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

    # Esqueletonizar
    skeleton = skeletonize(binary_image)
    return(skeleton)
    


# In[14]:


"""
# Crear el objeto Skeleton
skeleton_obj = Skeleton(skeleton)

# Sumarizar los datos del esqueleto
summary = summarize(skeleton_obj)
"""


# In[15]:


summary


# In[16]:


"""
# Calcular la tortuosidad: longitud del segmento / distancia euclidiana
# Puedes definir un umbral para la tortuosidad a partir del cual considerarás que un vaso es tortuoso
tortuosity_threshold = 1.05  # Este valor es un ejemplo, ajústalo según tus necesidades
summary['tortuosity'] = summary['branch-distance'] / summary['euclidean-distance']
tortuous_segments = summary[summary['tortuosity'] > tortuosity_threshold]

# Imprimir los segmentos que son tortuosos
print(tortuous_segments)
"""


# In[8]:


# Aumentar el grosor de los segmentos tortuosos
def thicken_line(coords, image_shape, thickness=3):
    # Inicializar una imagen en blanco con la misma forma que la original
    thick_line_image = np.zeros(image_shape)
    
    # Establecer los píxeles de la línea en 1
    thick_line_image[coords[:, 0], coords[:, 1]] = 255
    
    # Dilatar la línea para aumentar el grosor
    if thickness > 1:
        thick_line_image = dilation(thick_line_image, square(thickness))
    
    return thick_line_image


# In[12]:


#Encapsular-ho en una funció

def tortuosity(imgpath, tortuosity_threshold):
    skeleton=vessel_pattern_skeletonized(imgpath)
    # Crear el objeto Skeleton
    skeleton_obj = Skeleton(skeleton)

    # Sumarizar los datos del esqueleto
    summary = summarize(skeleton_obj)

    summary['tortuosity'] = summary['branch-distance'] / summary['euclidean-distance']
    tortuous_segments = summary[summary['tortuosity'] > tortuosity_threshold]
    
    # Crear una imagen para mostrar los resultados con mayor grosor
    image=skeleton
    results_image = np.zeros_like(image)
        
    
    # Dibujar solo los segmentos tortuosos con mayor grosor
    for segment in tortuous_segments['skeleton-id']:
        coords = skeleton_obj.path_coordinates(segment)
        segment_image = thicken_line(coords, image.shape, thickness=4)  # Ajusta el grosor aquí
        results_image = np.maximum(results_image, segment_image)
    
    #results_image = skeleton_obj.path_lengths()
    
    
    return(results_image)

imgpath='image.jpg'
image_preprocessed= skimage.io.imread(imgpath) #imatge preprocessada
image_preprocessed_resized = resize(           #redimensionamos para superposición en ax3
    image_preprocessed, 
    (512, 512), 
    preserve_range=True, 
    anti_aliasing=True
).astype(image_preprocessed.dtype)
skeleton=vessel_pattern_skeletonized(imgpath)  #patró vascular squeletonitzat
tortuositat= tortuosity(imgpath, 0 )          

fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(20,20))
ax1.axis('off')
ax2.axis('off')
ax3.axis('off')
ax1.set_title('Imatge preprocessada', fontsize=20)
ax2.set_title('Patró vascular', fontsize=20)
ax3.set_title('Segments tortuosos', fontsize=20)

ax1.imshow(image_preprocessed)
ax2.imshow(skeleton, cmap='gray')
#ax3.imshow(image_preprocessed_resized)
ax3.imshow(tortuositat, cmap='gray')#, alpha=0.5)

plt.show()


# In[11]:

skeleton=vessel_pattern_skeletonized(imgpath)
    # Crear el objeto Skeleton
skeleton=imread('vessel_prediction.jpg')
skeleton_obj = Skeleton(skeleton)

    # Sumarizar los datos del esqueleto
summary = summarize(skeleton_obj)
summary['tortuosity'] = summary['branch-distance'] / summary['euclidean-distance']

tortuosity_threshold = 1.1
tortuous_segments = summary[summary['tortuosity'] > tortuosity_threshold]


# In[20]:


results_image = skeleton_obj.path_label_image()
# Crear una máscara donde solo los segmentos con alta tortuosidad sean visibles
mask = np.isin(results_image, summary['skeleton-id'].unique())

# Aplicar la máscara a la imagen de etiquetas
filtered_image = np.where(mask, results_image, 0)

# Mostrar la imagen filtrada
plt.imshow(filtered_image, cmap='jet')
plt.axis('off')
plt.show()


# In[22]:


summary.shape


# In[13]:


tortuosity_threshold=1.05
tortuous_segments = summary[summary['tortuosity'] > tortuosity_threshold]
tortuous_segments.head()


# In[29]:


#generar predicció model 
vessel_prediction= vessel_pattern(imgpath)
plt.imshow(vessel_prediction, cmap='gray')


# In[33]:


vessel_prediction_skeletonized= vessel_pattern_skeletonized(imgpath)
plt.imshow(vessel_prediction_skeletonized, cmap='gray')


# In[37]:


from skimage import io

imsave('vessel_prediction_skeletonized.jpg', vessel_prediction_skeletonized)


# # Generar tots els patrons vasculars al conjunt de test

# In[18]:


"""
from vit_keras import utils, layers
import cv2
import matplotlib.cm as cm
"""

preprocessed_source_folder = 'test_preprocessed/DR'
destination_folder='test_patro_vascular/DR'
images = os.listdir(preprocessed_source_folder)
for image in images:
    imgpath = os.path.join(preprocessed_source_folder, image)
    image_preprocessed= skimage.io.imread(imgpath) #imatge preprocessada
    image_preprocessed_resized = resize(           #redimensionamos para superposición en ax3
    image_preprocessed, 
    (512, 512), 
    preserve_range=True, 
    anti_aliasing=True
    ).astype(image_preprocessed.dtype)
    skeleton=vessel_pattern_skeletonized(imgpath)  #patró vascular squeletonitzat
    tortuositat= tortuosity(imgpath, 1.05 )

   # Plot the results
    fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(20,20))
    ax1.axis('off')
    ax2.axis('off')
    ax3.axis('off')
    ax1.set_title('Imatge preprocessada', fontsize=20)
    ax2.set_title('Patró vascular', fontsize=20)
    ax3.set_title('Segments tortuosos', fontsize=20)

    ax1.imshow(image_preprocessed)
    ax2.imshow(skeleton, cmap='gray')
    ax3.imshow(image_preprocessed_resized)
    ax3.imshow(tortuositat, cmap='gray', alpha=0.5)
    # Guardar la figura completa en un archivo
    save_path = os.path.join(destination_folder, image)
    fig.savefig(save_path, bbox_inches='tight')
    print(f"Figure saved as {save_path}")
    plt.close(fig)  # Cierra la figura para liberar memoria


# In[ ]:

import numpy as np

# Generar la imagen de etiquetas de los caminos del esqueleto
results_image = skeleton_obj.path_label_image()

# Obtener todos los identificadores únicos de segmentos presentes en la imagen
unique_segment_ids_in_image = np.unique(results_image)

# Obtener todos los identificadores únicos de segmentos en el DataFrame summary
unique_segment_ids_in_summary = summary['skeleton-id'].unique()

# Verificar si todos los segmentos en la imagen están presentes en summary
all_segments_present = np.all(np.isin(unique_segment_ids_in_image, unique_segment_ids_in_summary))

print(f"Todos los segmentos están presentes en el summary: {all_segments_present}")




# In[ ]:




