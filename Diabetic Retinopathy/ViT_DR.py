# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 23:14:42 2024

@author: carle
"""

#Importar llibreries
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

#Establir model determinístic
def seed_everything(seed = 5):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'

seed_everything()

import warnings
warnings.filterwarnings("ignore")

#Definir hyper-paràmetres
image_size = 224
batch_size = 8
n_classes = 2
EPOCHS = 30

train_path = 'Dataset/train'
valid_path = 'Dataset/valid'
test_path = 'Dataset/test'

# Adjustar classes del diccionari
classes = {0: "DR", 1: "No_DR"}

#Augmanetació de dades
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
        contrast_factor = 0.5 #1.0
        image = (image - 0.5) * contrast_factor + 0.5
    if p_pixel_2 >= 0.4:
        # Manual brightness adjustment
        brightness_factor = 0.5 #1.0
        image = image * brightness_factor
    if p_pixel_3 >= 0.4:
        # Manual saturation adjustment
        saturation_factor = 0.5 #1.0
        image = (image - 0.5) * saturation_factor + 0.5

    return image

#Generació de dades
# Conjunt d'entrenamet
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255,
                                                                samplewise_center=True,
                                                                samplewise_std_normalization=True,
                                                                preprocessing_function=data_augment)

train_gen = train_datagen.flow_from_directory(train_path,
                                              target_size=(224, 224),
                                              batch_size=batch_size,
                                              seed=1,
                                              color_mode='rgb',
                                              shuffle=True,
                                              class_mode='categorical')

# Conjunt de valifdació
valid_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255,
                                                                samplewise_center=True,
                                                                samplewise_std_normalization=True)

valid_gen = valid_datagen.flow_from_directory(valid_path,
                                              target_size=(224, 224),
                                              batch_size=batch_size,
                                              seed=1,
                                              color_mode='rgb',
                                              shuffle=False,
                                              class_mode='categorical')
# Conjunt de test
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255,
                                                                samplewise_center=True,
                                                                samplewise_std_normalization=True)

test_gen = valid_datagen.flow_from_directory(test_path,
                                              target_size=(224, 224),
                                              batch_size=batch_size,
                                              seed=1,
                                              color_mode='rgb',
                                              shuffle=False,
                                              class_mode='categorical')

#Visualització d'un batch d'imatges
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

#Definició dels patches
class Patches(L.Layer):
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images = images,
            sizes = [1, self.patch_size, self.patch_size, 1],
            strides = [1, self.patch_size, self.patch_size, 1],
            rates = [1, 1, 1, 1],
            padding = 'VALID',
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches

#Visualitació dels patches en una imatge
plt.figure(figsize=(4, 4))
batch_size = 16
patch_size = 7  # Size of the patches to be extract from the input images
num_patches = (image_size // patch_size) ** 2

x = train_gen.next()
image = x[0][0]

plt.imshow(image.astype('uint8')) #ull 255
plt.axis('off')

resized_image = tf.image.resize(
    tf.convert_to_tensor([image]), size = (image_size, image_size)
)

patches = Patches(patch_size)(resized_image)
print(f'Image size: {image_size} X {image_size}')
print(f'Patch size: {patch_size} X {patch_size}')
print(f'Patches per image: {patches.shape[1]}')
print(f'Elements per patch: {patches.shape[-1]}')

n = int(np.sqrt(patches.shape[1]))
plt.figure(figsize=(4, 4))

for i, patch in enumerate(patches[0]):
    ax = plt.subplot(n, n, i + 1)
    patch_img = tf.reshape(patch, (patch_size, patch_size, 3))
    plt.imshow(patch_img.numpy().astype('uint8'))
    plt.axis('off')

#Model pre-entrenat
from vit_keras import vit
vit_model = vit.vit_b16(image_size=image_size,
                        activation='softmax',
                        pretrained=True,
                        include_top=False,
                        pretrained_top=False,
                        classes=n_classes)
vit_model.summary()

#Model complert amb les darreres capes per entrenar-les
model = tf.keras.Sequential([
    vit_model,
    tf.keras.layers.Flatten(),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(128, activation=tfa.activations.gelu),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(64, activation=tfa.activations.gelu),
    tf.keras.layers.Dense(32, activation=tfa.activations.gelu),
    tf.keras.layers.Dense(n_classes, activation='softmax')  # Adjusted for n_classes and softmax
], name='vision_transformer')

model.summary()

#Entrenar el model
warnings.filterwarnings("ignore")

learning_rate = 1e-4

optimizer = tfa.optimizers.RectifiedAdam(learning_rate = learning_rate)

model.compile(optimizer = optimizer, 
              loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing = 0.2), 
              metrics = ['accuracy'])
STEP_SIZE_TRAIN = train_gen.n // train_gen.batch_size
STEP_SIZE_VALID = valid_gen.n // valid_gen.batch_size
early_stopping_callbacks = tf.keras.callbacks.EarlyStopping(patience = 15, restore_best_weights = True, verbose = 1)
model.fit(x = train_gen,
          steps_per_epoch = STEP_SIZE_TRAIN,
          validation_data = valid_gen,
          validation_steps = STEP_SIZE_VALID,
          epochs = EPOCHS,
          callbacks = early_stopping_callbacks)
trainHistory = model.history 

#Salvar i carregar el model
import pickle
from keras.models import load_model

def save_trained_model(fileName, theModel, theHistory):
    # Check if the file already exists
    if not os.path.exists(fileName + '.h5'):
        # Save the model
        theModel.save(fileName + '.h5')
        # Save the history using pickle
        if isinstance(theHistory, tf.keras.callbacks.History):
            # Si theHistory és un objecte History d'un from model.fit --> D'un primer entrenament
            history_dict = {
                'loss': theHistory.history['loss'],
                'accuracy': theHistory.history['accuracy'],
                'val_loss': theHistory.history['val_loss'],
                'val_accuracy': theHistory.history['val_accuracy'],
            }
        else:
            # Si theHistory és un diccionari creat manualment --> entrenaments adicionals
            history_dict = theHistory

        with open(fileName + '_history.pkl', 'wb') as file:
            pickle.dump(history_dict, file)

        print("Model and history saved successfully.")
    else:
        print("The file already exists. No saving will be done.")   
        
def load_trained_model(fileName):
    
    # Per carregar el model s'ha de donar el custom_object 
    model = load_model(fileName+'.h5', custom_objects={'vit_model': vit_model})

    # Carregar història
    historyFile = fileName + '_history.pkl'
    if os.path.exists(historyFile):
        with open(historyFile, 'rb') as file:
            trainHistory = pickle.load(file)
        print("Model and trainHistory loaded successfully.")
    else:
        trainHistory = None
        print("History file not found.")

    return model, trainHistory

save_trained_model('ViT_model', model, trainHistory)

#Salvar el model
save_trained_model('ViT_model', model, trainHistory)

#Carrgar el model
loaded_model, previous_history = load_trained_model('ViT_model')

#En cas que es vulguin fer entrenaments adicionals
"""
additional_epochs = 5
learning_rate = 1e-4 
optimizer = tfa.optimizers.RectifiedAdam(learning_rate=learning_rate) 
loaded_model.compile(
    optimizer=optimizer,
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.2),
    metrics=['accuracy']
)

STEP_SIZE_TRAIN = train_gen.n // train_gen.batch_size
STEP_SIZE_VALID = valid_gen.n // valid_gen.batch_size

early_stopping_callbacks = tf.keras.callbacks.EarlyStopping(patience = 15, restore_best_weights = True, verbose = 1)

new_history= loaded_model.fit(x = train_gen,
          steps_per_epoch = STEP_SIZE_TRAIN,
          validation_data = valid_gen,
          validation_steps = STEP_SIZE_VALID,
          epochs = additional_epochs,
          callbacks = early_stopping_callbacks)

save_trained_model('ViT_model', loaded_model, new_history)
loaded_model, new_history = load_trained_model('ViT_model')

# Concatenatenar totes les claus del dccionari anterior i el nou
combined_history = {}
for key in previous_history.keys():
    combined_history[key] = previous_history[key] + new_history[key]
save_trained_model('ViT_model', loaded_model, combined_history)
"""

#Grficar història
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
    
plot_history(previous_history)

#Resultats al conjunt de validació
predicted_classes = np.argmax(loaded_model.predict(valid_gen, steps = valid_gen.n // valid_gen.batch_size + 1), axis = 1)
true_classes = valid_gen.classes
class_labels = list(valid_gen.class_indices.keys())  

confusionmatrix = confusion_matrix(true_classes, predicted_classes)
plt.figure(figsize = (4, 4))
sns.heatmap(confusionmatrix, cmap = 'Blues', annot = True, cbar = True)

print(classification_report(true_classes, predicted_classes))

#Resultats al conjut de test
#cheking result on test dataset
predicted_classes = np.argmax(loaded_model.predict(test_gen, steps = test_gen.n // test_gen.batch_size + 1), axis = 1)
true_classes = test_gen.classes
class_labels = list(test_gen.class_indices.keys())  

confusionmatrix = confusion_matrix(true_classes, predicted_classes)
plt.figure(figsize = (4, 4))
sns.heatmap(confusionmatrix, cmap = 'Blues', annot = True, cbar = True)

print(classification_report(true_classes, predicted_classes))

#Inferència
from tensorflow.keras.preprocessing import image
import numpy as np

# Carregar la imatge
#img_path = 'Dataset/test/DR/00cb6555d108_png.rf.29cca170969c6e9918ef9b9209abef8e.jpg' #DR
img_path = 'Dataset/test/NO_DR/851e40a21f81_png.rf.ea3c2c391c1bad72e2ca50db8cf2270c.jpg' # no DR
img = image.load_img(img_path, target_size=(224, 224))  # Replace with your model's input size

# Preprocessar la imatge
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
img_array /= 255.0  # Assuming your model expects images in [0, 1]

# Fer la predicció
predictions = loaded_model.predict(img_array)

# Assignar etiqueta a la predicció
predicted_class = np.argmax(predictions, axis=1)
predicted_class_label = class_labels[predicted_class[0]]

fig, ax = plt.subplots(figsize=(5,5))
ax.axis('off')
ax.imshow(img)

print("Predicted class:", predicted_class_label)

#Visualització del mapa d'atenció
from vit_keras import utils, visualize

# Carregar la imatge
img =utils.read(img_path, 224)

#Generar el mapa d'atenció
attention_map = visualize.attention_map(model=vit_model, image=img)

# Plot
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10,10))
ax1.axis('off')
ax2.axis('off')
ax1.set_title('Original')
ax2.set_title('Attention Map')
ax1.imshow(img)
ax2.imshow(attention_map)

#Mapes d'atenció de falsos negatius

predictions = loaded_model.predict(valid_gen)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = valid_gen.classes
filenames = valid_gen.filenames

# Trobar els indexs dels falsos negatius
false_negatives_indices = [i for i, (true, pred) in enumerate(zip(true_classes, predicted_classes)) if true == 0 and pred == 1]

# Trobar els noms dels arxius dels falsos negatius
false_negatives_filenames = [filenames[i] for i in false_negatives_indices]

for filename in false_negatives_filenames:
    img_path = os.path.join(valid_path, filename)
    img = utils.read(img_path, image_size)

    # Generar mapa d'atenció
    attention_map = visualize.attention_map(model=vit_model, image=img)

    # Plot imatge original i mapa d'atenció
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 5))
    ax1.imshow(img)
    ax1.set_title('Original')
    ax1.axis('off')
    ax2.imshow(attention_map)
    ax2.set_title('Attention Map')
    ax2.axis('off')
    plt.show()
    
#Mapes d'atenció en falsos positius
predictions = loaded_model.predict(valid_gen)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = valid_gen.classes
filenames = valid_gen.filenames

# Trobar els índexs dela falsos positius
false_positives_indices = [i for i, (true, pred) in enumerate(zip(true_classes, predicted_classes)) if true == 1 and pred == 0]

# Obtenir el nom dels arxius dels falsos positius 
false_positives_filenames = [filenames[i] for i in false_positives_indices]

for filename in false_positives_filenames:
    img_path = os.path.join(valid_path, filename)
    img = utils.read(img_path, image_size)

    attention_map = visualize.attention_map(model=vit_model, image=img)

    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 5))
    ax1.imshow(img)
    ax1.set_title('Original')
    ax1.axis('off')
    
    ax2.imshow(attention_map)
    ax2.set_title('Attention Map')
    ax2.axis('off')
    plt.show()
