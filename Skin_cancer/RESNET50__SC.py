# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 15:09:26 2024

@author: carle
"""

import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.layers as L
import tensorflow_addons as tfa
import glob, random, os, warnings
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import seaborn as sns

def seed_everything(seed = 42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'

seed_everything()

import warnings
warnings.filterwarnings("ignore")

image_size = 224
batch_size = 16
n_classes = 2
EPOCHS = 10

train_path = 'data/train'
#valid_path = 'Dataset/valid'
test_path = 'data/test'

################## Data augmentation ###########################################################

def data_augment(image):
    p_spatial = tf.random.uniform([], 0, 1.0, dtype=tf.float32)
    p_rotate = tf.random.uniform([], 0, 1.0, dtype=tf.float32)
    p_pixel_1 = tf.random.uniform([], 0, 1.0, dtype=tf.float32)
    p_pixel_2 = tf.random.uniform([], 0, 1.0, dtype=tf.float32)
    p_pixel_3 = tf.random.uniform([], 0, 1.0, dtype=tf.float32)

    # Flips
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)

    if p_spatial > 0.75:
        image = tf.image.transpose(image)

    # Rotates
    if p_rotate > 0.75:
        image = tf.image.rot90(image, k=3)  # rotate 270º
    elif p_rotate > 0.5:
        image = tf.image.rot90(image, k=2)  # rotate 180º
    elif p_rotate > 0.25:
        image = tf.image.rot90(image, k=1)  # rotate 90º

    # Pixel-level transforms
    if p_pixel_1 >= 0.4:
        # Manual contrast adjustment
        contrast_factor = 1.0
        image = (image - 0.5) * contrast_factor + 0.5
    if p_pixel_2 >= 0.4:
        # Manual brightness adjustment
        brightness_factor = 1.0
        image = image * brightness_factor
    if p_pixel_3 >= 0.4:
        # Manual saturation adjustment
        saturation_factor = 1.0
        image = (image - 0.5) * saturation_factor + 0.5

    return image

################ Data generation ###############################################################

datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale = 1./255,
                                                          samplewise_center = True,
                                                          samplewise_std_normalization = True,
                                                          validation_split = 0.2,
                                                          preprocessing_function = data_augment)

# set as training data

train_gen  = datagen.flow_from_directory(
    train_path,
    target_size=(224, 224),
    batch_size = batch_size,
    seed = 1,
    color_mode = 'rgb', #ojo era rgb
    shuffle = True,
    class_mode='categorical',
    subset='training') 

# same directory as training data

valid_gen  = datagen.flow_from_directory(
    train_path ,
    target_size=(224, 224),
    batch_size = batch_size,
    seed = 1,
    color_mode = 'rgb', #ojo era rgb
    shuffle = False,
    class_mode='categorical',
    subset='validation')

test_gen = datagen.flow_from_directory(test_path,
                                              target_size=(224, 224),
                                              batch_size=batch_size,
                                              seed=1,
                                              color_mode='rgb',
                                              shuffle=False,
                                              class_mode='categorical')

#####################Sample visualization ########################################
warnings.filterwarnings("ignore")

images = [train_gen[0][0][i] for i in range(4)]
fig, axes = plt.subplots(1, 4, figsize = (10, 10))

#axes = axes.flatten()

for img, ax in zip(images, axes):
    #ax.imshow((img.reshape(image_size, image_size, 3)*255).astype("uint8"))
    ax.imshow(img.reshape(image_size, image_size, 3))
    ax.axis('off')

plt.tight_layout()
plt.show()

################# Building the model ##################################################

CNN_model = tf.keras.Sequential()

base1= tf.keras.applications.ResNet50V2(include_top=False,
                   input_shape=(224,224,3),
                   pooling='avg',classes=2,
                   weights='imagenet')

CNN_model.add(base1)

CNN_model.add(tf.keras.layers.Flatten())
CNN_model.add(tf.keras.layers.BatchNormalization())
CNN_model.add(tf.keras.layers.Dense(128, activation = tfa.activations.gelu))
CNN_model.add(tf.keras.layers.BatchNormalization())
CNN_model.add(tf.keras.layers.Dense(64, activation = tfa.activations.gelu))
CNN_model.add(tf.keras.layers.Dense(32, activation = tfa.activations.gelu))
CNN_model.add(tf.keras.layers.Dense(n_classes, 'softmax'))

for layer in base1.layers:
    layer.trainable = False   

CNN_model.summary()

#################### Training the model ##################################################
warnings.filterwarnings("ignore")

learning_rate = 1e-5

optimizer = tfa.optimizers.RectifiedAdam(learning_rate = learning_rate)

CNN_model.compile(optimizer = optimizer, 
              loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing = 0.2), 
              metrics = ['accuracy'])

STEP_SIZE_TRAIN = train_gen.n // train_gen.batch_size
STEP_SIZE_VALID = valid_gen.n // valid_gen.batch_size



early_stopping_callbacks = tf.keras.callbacks.EarlyStopping(patience = 15, restore_best_weights = True, verbose = 1)

CNN_model.fit(x = train_gen,
          steps_per_epoch = STEP_SIZE_TRAIN,
          validation_data = valid_gen,
          validation_steps = STEP_SIZE_VALID,
          epochs = EPOCHS,
          callbacks = early_stopping_callbacks)

CNN_trainHistory = CNN_model.history

################# Salvar y cargar modelo ##################################################
import pickle
from keras.models import load_model

def save_trained_model(fileName, theModel, theHistory):
    # Check if the file already exists
    if not os.path.exists(fileName + '.h5'):
        # Save the model
        theModel.save(fileName + '.h5')

        # Save the history using pickle
        if isinstance(theHistory, tf.keras.callbacks.History):
            # If theHistory is a History object from model.fit --> 1st training
            history_dict = {
                'loss': theHistory.history['loss'],
                'accuracy': theHistory.history['accuracy'],
                'val_loss': theHistory.history['val_loss'],
                'val_accuracy': theHistory.history['val_accuracy'],
            }
        else:
            # If theHistory is a manually created dictionary --> additional trainings
            history_dict = theHistory

        with open(fileName + '_history.pkl', 'wb') as file:
            pickle.dump(history_dict, file)

        print("Model and history saved successfully.")
    else:
        print("The file already exists. No saving will be done.")   
        
def load_trained_model(fileName):
    
    # Provide the custom object when loading the model

    model = load_model(fileName+'.h5', custom_objects={'CNN_model': CNN_model})
   
    # Load the training history using pickle
    historyFile = fileName + '_history.pkl'
    if os.path.exists(historyFile):
        with open(historyFile, 'rb') as file:
            trainHistory = pickle.load(file)
        print("Model and trainHistory loaded successfully.")
    else:
        trainHistory = None
        print("History file not found.")

    return model, trainHistory

############### Cargar modelo ######################################################
loaded_model, previous_history = load_trained_model('RESNET50_model')


############### Nou entrenament ####################################################
import datetime
additional_epochs = 10
learning_rate = 1e-5 
optimizer = tfa.optimizers.RectifiedAdam(learning_rate=learning_rate) 
loaded_model.compile(
    optimizer=optimizer,
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.2),
    metrics=['accuracy']
)

STEP_SIZE_TRAIN = train_gen.n // train_gen.batch_size
STEP_SIZE_VALID = valid_gen.n // valid_gen.batch_size

early_stopping_callback = tf.keras.callbacks.EarlyStopping(patience = 15, restore_best_weights = True, verbose = 1)

# TensorBoard callback
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir='logs/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
    histogram_freq=1,  # Frequency (in epochs) at which to compute activation and weight histograms for the layers of the model.
    write_graph=True)

# Include both callbacks in a list
callbacks_list = [early_stopping_callback, tensorboard_callback]

new_history= loaded_model.fit(x = train_gen,
          steps_per_epoch = STEP_SIZE_TRAIN,
          validation_data = valid_gen,
          validation_steps = STEP_SIZE_VALID,
          epochs = additional_epochs)#,
          #callbacks = callbacks_list)
    
######## Gravar nou model ######################################################
save_trained_model('RESNET50_model', loaded_model, new_history)

######### Carregar nou model ###################################################
loaded_model, new_history = load_trained_model('RESNET50_model')


################## Concatenar històries ######################################
combined_history = {}
for key in previous_history.keys():
    combined_history[key] = previous_history[key] + new_history[key]
    
######################## Gravar model #######################################
save_trained_model('RESNET50_model', loaded_model, combined_history)

############### Graficar història #########################################
def plot_history(theHistory):
    fig, ax = plt.subplots(2, 1, figsize=(8, 12))
    
    # Create the first subplot for the evolution of training and validation loss
    ax[0].plot(theHistory['loss'], label='Training Loss')
    ax[0].plot(theHistory['val_loss'], label='Validation Loss')
    ax[0].set_title('Evolution of Loss')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Loss')
    ax[0].legend()
    ax[0].grid(True)
    
    # Create the second subplot for the evolution of training and validation accuracy
    ax[1].plot(theHistory['accuracy'], label='Training Accuracy')
    ax[1].plot(theHistory['val_accuracy'], label='Validation Accuracy')
    ax[1].set_title('Evolution of Accuracy')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Accuracy')
    ax[1].legend()
    ax[1].grid(True)
    
    plt.show()

plot_history(combined_history)

################### Resultats al conjunt de validació ##############################
predicted_classes = np.argmax(loaded_model.predict(valid_gen, steps = valid_gen.n // valid_gen.batch_size + 1), axis = 1)
true_classes = valid_gen.classes
class_labels = list(valid_gen.class_indices.keys())  

confusionmatrix = confusion_matrix(true_classes, predicted_classes)
plt.figure(figsize = (4, 4))
sns.heatmap(confusionmatrix, cmap = 'Blues', annot = True, cbar = True)

print(classification_report(true_classes, predicted_classes))


################## Resultats al conjunt de test####################################
predicted_classes = np.argmax(loaded_model.predict(test_gen, steps = test_gen.n // test_gen.batch_size + 1), axis = 1)
true_classes = test_gen.classes
class_labels = list(test_gen.class_indices.keys())  

confusionmatrix = confusion_matrix(true_classes, predicted_classes)
plt.figure(figsize = (4, 4))
sns.heatmap(confusionmatrix, cmap = 'Blues', annot = True, cbar = True)

print(classification_report(true_classes, predicted_classes))

######################## Inferència ###############################################
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the image file
img_path = 'data/test/malignant/8.jpg' 

img = image.load_img(img_path, target_size=(224, 224))  # Replace with your model's input size

# Preprocess the image
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
img_array /= 255.0  # Assuming your model expects images in [0, 1]

# Make prediction
predictions = loaded_model.predict(img_array)

# Interpret prediction
predicted_class = np.argmax(predictions, axis=1)
predicted_class_label = class_labels[predicted_class[0]]

print("Predicted class:", predicted_class_label)

#################################### Grad-cam ########################################
import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import keras

# Display
from IPython.display import Image, display
import matplotlib as mpl

#Nom de la darrera capa de convolució
last_conv_layer_name = "conv5_block3_3_conv"

################# Algoritme grad-cam ##################################################
def get_img_array(img_path, size):
    # `img` is a PIL image of size 299x299
    #img = keras.utils.load_img(img_path, target_size=size) #prob. compatibilitat versions
    img = tf.utils.load_img(img_path, target_size=size)
    # `array` is a float32 Numpy array of shape (299, 299, 3)
    array = keras.utils.img_to_array(img)
    # We add a dimension to transform our array into a "batch"
    # of size (1, 299, 299, 3)
    array = np.expand_dims(array, axis=0)
    return array


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = keras.models.Model(
        model.inputs, [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


################ Visualitzar heat-map ###########################################
# Make model
model = base1

# Remove last layer's softmax
model.layers[-1].activation = None

# Generate class activation heatmap
with tf.device('/cpu:0'): #desactivar GPU- Si no, dóna problemes
    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)
    
# Display heatmap
plt.matshow(heatmap)
plt.show()

######################### Funció grad-cam ########################################

def gradcam(img_path, model=base1, alpha=0.4):
    
    #Preprocessar la imatge
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Assuming your model expects images in [0, 1]
    
    # Calcular heatmap
    # Remove last layer's softmax
    model.layers[-1].activation = None
    with tf.device('/cpu:0'): #desactivar GPU
        heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)
    
    # Load the original image
    #img = keras.utils.load_img(img_path)
    img = tf.keras.utils.load_img(img_path)
    #img = keras.utils.img_to_array(img)
    img = tf.keras.utils.img_to_array(img)

    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = mpl.colormaps["jet"]

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    #jet_heatmap = keras.utils.array_to_img(jet_heatmap)
    jet_heatmap = tf.keras.utils.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    #jet_heatmap = keras.utils.img_to_array(jet_heatmap)
    jet_heatmap = tf.keras.utils.img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * alpha + img
    #superimposed_img = keras.utils.array_to_img(superimposed_img)
    superimposed_img = tf.keras.utils.array_to_img(superimposed_img)

    return superimposed_img

########### Visualitzar 10 imatges classificades correctament ######################

predictions = loaded_model.predict(valid_gen)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = valid_gen.classes
filenames = valid_gen.filenames

# Trobar els indexs dels elements classificats correctament
correct_classification_indices = [i for i, (true, pred) in enumerate(zip(true_classes, predicted_classes)) if (true == 0 and pred == 0) or (true==1 and pred==1)]

#seleccionar 10 indexs random
random_indices = np.random.choice(correct_classification_indices, 10, replace=False)

# Trobar els noms dels arxius
correct_class_filenames = [filenames[i] for i in random_indices]

for filename in correct_class_filenames:
    img_path = os.path.join(train_path, filename)
    img = utils.read(img_path, image_size)
    img_gradcam= gradcam(img_path)

    # Plot the original image and attention map
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 5))
    ax1.imshow(img)
    ax1.set_title('Original')
    ax1.axis('off')
    
    ax2.imshow(img_gradcam)
    ax2.set_title('Grad-cam')
    ax2.axis('off')
    
    plt.show()


################### Visualitzar 10 imatges classificades incorrectament ####################
#Elements de Classe 0 (Meningioma) classificats erròniament
predictions = loaded_model.predict(valid_gen)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = valid_gen.classes
filenames = valid_gen.filenames

# Trobar els indexs dels elements classificats correctament
incorrect_classification_indices = [i for i, (true, pred) in enumerate(zip(true_classes, predicted_classes)) if (true == 0 and pred!= 0) or (true==1 and pred!=1)]

#seleccionar 10 indexs random
random_indices = np.random.choice(correct_classification_indices, 10, replace=False)

# Trobar els noms dels arxius
correct_class_filenames = [filenames[i] for i in random_indices]

for filename in correct_class_filenames:
    img_path = os.path.join(train_path, filename)
    img = utils.read(img_path, image_size)
    img_gradcam= gradcam(img_path)

    # Plot the original image and attention map
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 5))
    ax1.imshow(img)
    ax1.set_title('Original')
    ax1.axis('off')
    
    ax2.imshow(img_gradcam)
    ax2.set_title('Grad-cam')
    ax2.axis('off')
    
    plt.show()
#print(count, ' elements')