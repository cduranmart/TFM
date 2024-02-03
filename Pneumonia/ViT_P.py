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


# In[3]:


image_size = 224
batch_size = 32
n_classes = 2
EPOCHS = 40

#train_path = 'chest_xray/train'
train_path = 'chest_xray/augmented'
test_path = 'chest_xray/test'

classes = {0: "Normal", 1: "Pneumonia"}


# ## Data augmentation

# In[4]:


#Pre augmentació desde el disc
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
        contrast_factor = 0.8
        image = (image - 0.5) * contrast_factor + 0.5
    if p_pixel_2 >= 0.4:
        # Manual brightness adjustment
        brightness_factor = 0.8
        image = image * brightness_factor
    if p_pixel_3 >= 0.4:
        # Manual saturation adjustment
        saturation_factor = 0.8
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

test_gen  = datagen.flow_from_directory(
    test_path ,
    target_size=(224, 224),
    batch_size = batch_size,
    seed = 1,
    color_mode = 'rgb', #ojo era rgb
    shuffle = False,
    class_mode='categorical')


# ## Sample image visualization

# In[5]:


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

# In[6]:


from vit_keras import vit

vit_model = vit.vit_b16(image_size=image_size,
                        activation='softmax',
                        pretrained=True,
                        include_top=False,
                        pretrained_top=False,
                        classes=n_classes)


# ## ViT Model Architecture

# In[7]:


# Freeze all the layers in the loaded ViT model
for layer in vit_model.layers:
    layer.trainable = False

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

# In[10]:


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

# In[8]:


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



# In[13]:


save_trained_model('ViT_model', model, trainHistory)


# ## Carregar model 

# In[9]:


loaded_model, previous_history = load_trained_model('ViT_model')


# ## Nou entrenament (opcional)

# In[ ]:


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
        
      


# In[12]:


save_trained_model('ViT_model', loaded_model, new_history)


# In[13]:


loaded_model, new_history = load_trained_model('ViT_model')


# In[ ]:


new_history['val_accuracy']


# In[15]:


# Concatenate all keys in the training history

combined_history = {}
for key in previous_history.keys():
    combined_history[key] = previous_history[key] + new_history[key]


# In[16]:


save_trained_model('ViT_model', loaded_model, combined_history)


# ## Graficar historia

# In[10]:


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

# In[11]:


#conjunt de validació
predicted_classes = np.argmax(loaded_model.predict(valid_gen, steps = valid_gen.n // valid_gen.batch_size + 1), axis = 1)
true_classes = valid_gen.classes
class_labels = list(valid_gen.class_indices.keys())  

confusionmatrix = confusion_matrix(true_classes, predicted_classes)
plt.figure(figsize = (4, 4))
sns.heatmap(confusionmatrix, cmap='Blues', annot=True, fmt='d', cbar=True)

print(classification_report(true_classes, predicted_classes))


# In[16]:


#Conjunt de test
predicted_classes = np.argmax(loaded_model.predict(test_gen, steps = test_gen.n // test_gen.batch_size + 1), axis = 1)
true_classes = test_gen.classes
class_labels = list(test_gen.class_indices.keys())  

confusionmatrix = confusion_matrix(true_classes, predicted_classes)
plt.figure(figsize = (4, 4))
sns.heatmap(confusionmatrix, cmap='Blues', annot=True, fmt='d', cbar=True)

print(classification_report(true_classes, predicted_classes))


# ## Inferencia

# In[14]:


from tensorflow.keras.preprocessing import image
import numpy as np

# Load the image file
#img_path = 'chest_xray/test/PNEUMONIA/BACTERIA-227418-0002.jpeg' 
#img_path = 'chest_xray/test/NORMAL/NORMAL-1763721-0001.jpeg' 
img_path = 'chest_xray/test/NORMAL/NORMAL-3322209-0001.jpeg'
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


# In[15]:


print(train_gen.class_indices)
print(valid_gen.class_indices)
print(test_gen.class_indices)


# In[59]:


image_files = os.listdir('chest_xray/test/PNEUMONIA')
count_pneumonia=0
count_normal=0
for img_file in image_files:
    img_path = os.path.join('chest_xray/test/PNEUMONIA', img_file)
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array /= 255.0  # Normalize the image
    img_array -= np.mean(img_array, keepdims=True)  #samplewise centering and normalization
    img_array /= (np.std(img_array, keepdims=True) + 1e-7)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    predictions = loaded_model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)
    predicted_class_label = class_labels[predicted_class[0]]
    if predicted_class_label=='PNEUMONIA': 
        count_pneumonia+=1
    else: 
        count_normal+=1
    print("Predicted class:", predicted_class_label)

print('NORMAL: ', count_normal,' PNEUMONIA: ',count_pneumonia)
    
    


# ## Mapa d'atenció

# In[60]:


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


# ## Visualitzar 10 imatges classificades correctament i grad-cam

# In[61]:


predictions = loaded_model.predict(test_gen)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = test_gen.classes
filenames = test_gen.filenames

# Trobar els indexs dels elements classificats correctament
correct_classification_indices = [i for i, (true, pred) in enumerate(zip(true_classes, predicted_classes)) if (true == 0 and pred == 0) or (true==1 and pred==1)]

#seleccionar 10 indexs random
random_indices = np.random.choice(correct_classification_indices, 10, replace=False)

# Trobar els noms dels arxius
correct_class_filenames = [filenames[i] for i in random_indices]

for filename in correct_class_filenames:
    img_path = os.path.join(test_path, filename)
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


# ## Visualitzar 10 imatges classificades incorrectament i grad-cam

# In[62]:


#Elements de Classe 2 (Pituitaria Tumor) classificats erròniament
predictions = loaded_model.predict(test_gen)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = test_gen.classes
filenames = test_gen.filenames

# Trobar els indexs dels elements classificats erròniament
incorrect_classification_indices = [i for i, (true, pred) in enumerate(zip(true_classes, predicted_classes)) if (true == 0 and pred != 0) or (true==1 and pred!=1)]
#seleccionar 10 indexs random
random_indices = np.random.choice(incorrect_classification_indices, 10, replace=False)

# Trobar el nom dels arxius
incorrect_class_filenames = [filenames[i] for i in random_indices]

for filename in incorrect_class_filenames:
    img_path = os.path.join(test_path, filename)
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




