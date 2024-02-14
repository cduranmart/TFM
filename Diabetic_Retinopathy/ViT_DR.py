#!/usr/bin/env python
# coding: utf-8

# In[2]:


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


# In[3]:


def seed_everything(seed = 5):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'

seed_everything()


# In[4]:


image_size = 224
batch_size = 32
n_classes = 2
EPOCHS = 30

#train_path = 'Dataset/train'
train_path = 'Dataset/augmented'
valid_path = 'Dataset/valid'
test_path = 'Dataset/test'

#classes = {0: "DR", 1: "No_DR"}


# ## Data augmentation

# In[5]:


#Preaugmentació al disc
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

# In[5]:


# For training images
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255,
                                                                samplewise_center=True,
                                                                samplewise_std_normalization=True)#,
                                                                #preprocessing_function=data_augment)

train_gen = train_datagen.flow_from_directory(train_path,
                                              target_size=(224, 224),
                                              batch_size=batch_size,
                                              seed=1,
                                              color_mode='rgb',
                                              shuffle=True,
                                              class_mode='categorical')

# For validation images
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
# For test images
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


# ## Sample image visualization

# In[6]:


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

# In[7]:


from vit_keras import vit

vit_model = vit.vit_b16(image_size=image_size,
                        activation='softmax',
                        pretrained=True,
                        include_top=False,
                        pretrained_top=False,
                        classes=n_classes)
#vit_model.summary()


# ## ViT Model Architecture

# In[8]:


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

# In[9]:


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

# In[10]:


loaded_model, history = load_trained_model('ViT_model')


# ## Nou entrenament (opcional)

# In[ ]:


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
        
"""      


# In[ ]:


#save_trained_model('ViT_model', loaded_model, new_history)


# In[ ]:


#loaded_model, new_history = load_trained_model('ViT_model')


# In[ ]:


#new_history['loss']


# In[28]:


# Concatenate les entrades del diccionari anterior amb les noves
"""
combined_history = {}
for key in previous_history.keys():
    combined_history[key] = previous_history[key] + new_history[key]
"""


# In[ ]:


#save_trained_model('ViT_model', loaded_model, combined_history)


# ## Graficar historia

# In[11]:


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


plot_history(history)


# ## Resultats al conjunt de validació

# In[12]:


#Comprovem etiquetes i la seva consistència als tres conjunts d'imatges

print(train_gen.class_indices)
print(valid_gen.class_indices)
print(test_gen.class_indices)


# In[13]:


predicted_classes = np.argmax(loaded_model.predict(valid_gen, steps = valid_gen.n // valid_gen.batch_size + 1), axis = 1)
true_classes = valid_gen.classes
class_labels = list(valid_gen.class_indices.keys())  

confusionmatrix = confusion_matrix(true_classes, predicted_classes)
plt.figure(figsize = (4, 4))
sns.heatmap(confusionmatrix, annot=True, fmt='d', cmap='Blues', cbar=True,
            xticklabels=class_labels, yticklabels=class_labels) 

print(classification_report(true_classes, predicted_classes, target_names=class_labels))


# ## Resultats al conjunt de test

# In[15]:


#cheking result on test dataset
predicted_classes = np.argmax(loaded_model.predict(test_gen, steps = test_gen.n // test_gen.batch_size + 1), axis = 1)
true_classes = test_gen.classes
class_labels = list(test_gen.class_indices.keys())  

confusionmatrix = confusion_matrix(true_classes, predicted_classes)
plt.figure(figsize = (4, 4))
sns.heatmap(confusionmatrix, annot=True, fmt='d', cmap='Blues', cbar=True,
            xticklabels=class_labels, yticklabels=class_labels) 

print(classification_report(true_classes, predicted_classes, target_names=class_labels))


# ## Inferència

# In[17]:


from tensorflow.keras.preprocessing import image
import numpy as np

# Load the image file
img_path = 'Dataset/test/DR/test_074.jpg' #DR
#img_path = 'Dataset/test/NO_DR/851e40a21f81_png.rf.ea3c2c391c1bad72e2ca50db8cf2270c.jpg' # no DR
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

fig, ax = plt.subplots(figsize=(5,5))
ax.axis('off')
ax.set_title(img_path)
ax.imshow(img)

print("Predicted class:", predicted_class_label)


# ## Visualització del mapa d'atenció

# In[20]:


from vit_keras import utils, visualize_customized

# Load the image file
img =utils.read(img_path, 224)

#Generate the attention map
attention_map = visualize_customized.attention_map(model=vit_model, image=img)

# Plot the results
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10,10))
ax1.axis('off')
ax2.axis('off')
ax1.set_title(img_path+' Original')
ax2.set_title(img_path+' Attention Map')
ax1.imshow(img)
ax2.imshow(attention_map)


# ### Retinografies amb RD classificades erròniament com a sanes

# In[22]:


predictions = loaded_model.predict(valid_gen)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = valid_gen.classes
filenames = valid_gen.filenames

# Find the indices of false negatives
false_negatives_indices = [i for i, (true, pred) in enumerate(zip(true_classes, predicted_classes)) if true == 0 and pred == 1]

# Retrieve the filenames of the false negative images
false_negatives_filenames = [filenames[i] for i in false_negatives_indices]

# Now you can loop through these filenames, load the images, and visualize the attention maps
for filename in false_negatives_filenames:
    img_path = os.path.join(valid_path, filename)
    img = utils.read(img_path, image_size)

    # Generate attention map
    attention_map = visualize_customized.attention_map(model=vit_model, image=img)

    # Plot the original image and attention map
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 5))
    ax1.imshow(img)
    ax1.set_title(filename+' Original')
    ax1.axis('off')
    
    ax2.imshow(attention_map)
    ax2.set_title(filename+' Attention Map')
    ax2.axis('off')
    
    plt.show()


# ### Retinografies sanes classificades erròniament com a RD

# In[23]:


predictions = loaded_model.predict(valid_gen)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = valid_gen.classes
filenames = valid_gen.filenames

# Find the indices of false negatives
false_positives_indices = [i for i, (true, pred) in enumerate(zip(true_classes, predicted_classes)) if true == 1 and pred == 0]

# Retrieve the filenames of the false negative images
false_positives_filenames = [filenames[i] for i in false_positives_indices]

# Now you can loop through these filenames, load the images, and visualize the attention maps
for filename in false_positives_filenames:
    img_path = os.path.join(valid_path, filename)
    img = utils.read(img_path, image_size)

    # Generate attention map
    attention_map = visualize_customized.attention_map(model=vit_model, image=img)

    # Plot the original image and attention map
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 5))
    ax1.imshow(img)
    ax1.set_title(filename+' Original')
    ax1.axis('off')
    
    ax2.imshow(attention_map)
    ax2.set_title(filename+' Attention Map')
    ax2.axis('off')
    
    plt.show()


# ## Mostra aleatòria de 10 Retinografies amb RD classificades correctament 

# In[26]:


predictions = loaded_model.predict(valid_gen)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = valid_gen.classes
filenames = valid_gen.filenames

# Trobar els indexs dels elements amb RD classificats correctament
indices = [i for i, (true, pred) in enumerate(zip(true_classes, predicted_classes)) if (true == 0 and pred == 0)]

#seleccionar 10 indexs random
random_indices = np.random.choice(indices, 10, replace=False)

# Trobar els noms dels arxius
filenames = [filenames[i] for i in random_indices]

for filename in filenames:
    img_path = os.path.join(valid_path, filename)
    img = utils.read(img_path, image_size)

    # Generate attention map
    attention_map = visualize_customized.attention_map(model=vit_model, image=img)

    # Plot the original image and attention map
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 5))
    ax1.imshow(img)
    ax1.set_title(filename +' Original')
    ax1.axis('off')
    
    ax2.imshow(attention_map)
    ax2.set_title(filename +' Attention Map')
    ax2.axis('off')
    
    plt.show()


# ## Mostra aleatòria de 10 Retinografies sanes classificades correctament 

# In[27]:


predictions = loaded_model.predict(valid_gen)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = valid_gen.classes
filenames = valid_gen.filenames

# Trobar els indexs dels elements amb RD classificats correctament
indices = [i for i, (true, pred) in enumerate(zip(true_classes, predicted_classes)) if (true == 1 and pred == 1)]

#seleccionar 10 indexs random
random_indices = np.random.choice(indices, 10, replace=False)

# Trobar els noms dels arxius
filenames = [filenames[i] for i in random_indices]

for filename in filenames:
    img_path = os.path.join(valid_path, filename)
    img = utils.read(img_path, image_size)

    # Generate attention map
    attention_map = visualize_customized.attention_map(model=vit_model, image=img)

    # Plot the original image and attention map
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 5))
    ax1.imshow(img)
    ax1.set_title(filename +' Original')
    ax1.axis('off')
    
    ax2.imshow(attention_map)
    ax2.set_title(filename +' Attention Map')
    ax2.axis('off')
    
    plt.show()


# In[ ]:




