#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


def seed_everything(seed = 5):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'

seed_everything()

import warnings
warnings.filterwarnings("ignore")


# In[5]:


image_size = 224
batch_size = 32
n_classes = 3
EPOCHS = 30

#train_path = 'brain-tumor_dataset'
train_path = 'brain-tumor_dataset_augmented'

"""
classes = {1 : "Meningioma",
           2 : "Glioma",
           3 : "Pituitary Tumor"}
"""


# ## Data augmentation

# In[11]:


#preaugmentació al disc dur
"""
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
"""    


# ## Data Generator

# In[4]:


datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale = 1./255,
                                                          samplewise_center = True,
                                                          samplewise_std_normalization = True,
                                                          validation_split = 0.2)#,
                                                          #preprocessing_function = data_augment)

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


# ## Sample image visualization

# In[7]:


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


# ## Building the model

# ## ViT B16 model

# In[8]:


from vit_keras import vit

vit_model = vit.vit_b16(image_size=image_size,
                        activation='softmax',
                        pretrained=True,
                        include_top=False,
                        pretrained_top=False,
                        classes=n_classes)


# ## ViT Model Architecture

# In[9]:


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

for layer in vit_model.layers:
    layer.trainable = False

model.summary()


# ## Training the Model

# In[8]:


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

trainHistory = model.history #guardamos historia


# ## Salvar y carregar model

# In[10]:


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

    model = load_model(fileName+'.h5', custom_objects={'vit_model': vit_model})
    #model = load_model(fileName+'.h5', custom_objects={'vit_model': vit.vit_b16})

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



# In[11]:


save_trained_model('ViT_model', model, trainHistory)


# ## Carregar model 

# In[11]:


loaded_model, previous_history = load_trained_model('ViT_model')


# ## Nou entrenament (opcional)

# In[9]:


additional_epochs = 10
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
        
     


# In[10]:


save_trained_model('ViT_model', loaded_model, new_history)


# In[11]:


loaded_model, new_history = load_trained_model('ViT_model')


# In[12]:


new_history['loss']


# In[13]:


# Concatenate all keys in the training history

combined_history = {}
for key in previous_history.keys():
    combined_history[key] = previous_history[key] + new_history[key]


# In[15]:


save_trained_model('ViT_model', loaded_model, combined_history)


# ## Graficar historia

# In[14]:


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


# ## Resultats

# In[15]:


predicted_classes = np.argmax(loaded_model.predict(valid_gen, steps = valid_gen.n // valid_gen.batch_size + 1), axis = 1)
true_classes = valid_gen.classes
class_labels = list(valid_gen.class_indices.keys())  

confusionmatrix = confusion_matrix(true_classes, predicted_classes)
plt.figure(figsize = (4, 4))
sns.heatmap(confusionmatrix, cmap = 'Blues', fmt='d', annot = True, cbar = True)

print(classification_report(true_classes, predicted_classes))


# ## Inferencia

# In[16]:


from tensorflow.keras.preprocessing import image
import numpy as np

# Load the image file
#img_path = 'brain-tumor_dataset/clase_1/image_3.png' 
img_path = 'brain-tumor_dataset/clase_2/image_567.png' #
img = image.load_img(img_path, target_size=(224, 224))  # Replace with your model's input size

# Preprocess the image
img_array = image.img_to_array(img)
img_array /= 255.0  # Normalize the image
img_array -= np.mean(img_array, keepdims=True)  #samplewise centering and normalization
img_array /= (np.std(img_array, keepdims=True) + 1e-7)
img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

# Make prediction
predictions = loaded_model.predict(img_array)

# Interpret prediction
predicted_class = np.argmax(predictions, axis=1)
predicted_class_label = class_labels[predicted_class[0]]

print("Predicted class:", predicted_class_label)


# ## Mapa d'atenció

# In[17]:


from vit_keras import utils, visualize

# Load the image file
img =utils.read(img_path, 224)

#Generate the attention map
attention_map = visualize.attention_map(model=vit_model, image=img)


# Plot the results
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10,10))
ax1.axis('off')
ax2.axis('off')
ax1.set_title('Original')
ax2.set_title('Attention Map')
ax1.imshow(img)
ax2.imshow(attention_map)


# ## Visualitzar 10 imatges classificades correctament i attention map

# In[27]:


#Elements de Classe 0 (Meningioma) classificats erròniament
predictions = loaded_model.predict(valid_gen)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = valid_gen.classes
filenames = valid_gen.filenames

# Trobar els indexs dels elements classificats correctament
correct_classification_indices = [i for i, (true, pred) in enumerate(zip(true_classes, predicted_classes)) if (true == 0 and pred == 0) or (true==1 and pred==1) or (true==2 and pred==2)]

#seleccionar 10 indexs random
random_indices = np.random.choice(correct_classification_indices, 10, replace=False)

# Trobar els noms dels arxius
correct_class_filenames = [filenames[i] for i in random_indices]

for filename in correct_class_filenames:
    img_path = os.path.join(train_path, filename)
    img = utils.read(img_path, image_size)

    # Generate attention map
    attention_map = visualize.attention_map(model=vit_model, image=img)

    # Plot the original image and attention map
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 5))
    ax1.imshow(img)
    ax1.set_title('Original')
    ax1.axis('off')
    
    ax2.imshow(attention_map)
    ax2.set_title('Attention Map')
    ax2.axis('off')
    
    plt.show()
#print(count, ' elements')


# ## Visualitzar 10 imatges classificades incorrectament i attention map

# In[28]:


#Elements de Classe 2 (Pituitaria Tumor) classificats erròniament
predictions = loaded_model.predict(valid_gen)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = valid_gen.classes
filenames = valid_gen.filenames

# Trobar els indexs dels elements classificats erròniament
incorrect_classification_indices = [i for i, (true, pred) in enumerate(zip(true_classes, predicted_classes)) if (true == 0 and pred != 0) or (true==1 and pred!=1) or (true==2 and pred!=2)]
#seleccionar 10 indexs random
random_indices = np.random.choice(incorrect_classification_indices, 10, replace=False)

# Trobar el nom dels arxius
incorrect_class_filenames = [filenames[i] for i in random_indices]

for filename in incorrect_class_filenames:
    img_path = os.path.join(train_path, filename)
    img = utils.read(img_path, image_size)

    # Generate attention map
    attention_map = visualize.attention_map(model=vit_model, image=img)

    # Plot the original image and attention map
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 5))
    ax1.imshow(img)
    ax1.set_title('Original')
    ax1.axis('off')
    
    ax2.imshow(attention_map)
    ax2.set_title('Attention Map')
    ax2.axis('off')
    
    plt.show()


# In[ ]:




