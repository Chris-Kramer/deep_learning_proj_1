#!/usr/bin/python3
###########
# Imports #
###########
# Model
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

# Data wrangling and preprocessing
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt

from os import listdir
from math import ceil

import warnings

warnings.filterwarnings("ignore")

##############################
# Part one - Data generators #
##############################

def get_test_data():
    print("MOCK FUNCTION")
    print("Step 1: read folder")
    print("step 2: scale to correct size")
    print("step 3: Profit!")

# ------ Training data ------
train_datagen = ImageDataGenerator(
    rescale=1./255, # Scale between 0 and 1
    rotation_range=40, # Random rotations of 40 degree 
    width_shift_range=0.2, # Randomly Shift pixels between 0% and 20% in width
    height_shift_range=0.2, # Randomly shift pixels between 0% and 20% in height
    shear_range=0.2, # Shear angle in counter-clockwise direction in degrees
    zoom_range=0.2, # Zoom in between 0 and 20 %
    horizontal_flip=True,
    vertical_flip = True)

train_generator = train_datagen.flow_from_directory(
    directory = r"catdog_data/train",
    target_size=(64, 64),
    color_mode = "rgb",
    batch_size=32,
    class_mode='binary',
    shuffle = True) # Hypotese: Giver mere generaliserbar model, hvis shuffle er true

# ------ Test data ------
test_val_datagen = ImageDataGenerator(rescale=1./255) # Used for test and val, since we do not want to augment them
test_generator = test_val_datagen.flow_from_directory(
    directory="catdog_data/test",
    target_size=(64, 64), # Optimal solution = 224x224 eller 112x112
    color_mode="rgb",
    batch_size = 32,
    class_mode= "binary",
    shuffle = False,   
)

# ------ Validation data ------
val_generator = test_val_datagen.flow_from_directory( 
    directory = "catdog_data/validation",
    target_size=(64, 64),
    color_mode = "rgb",
    batch_size=32,
    class_mode='binary')

####################
# Part two - Model # 
####################

# ------ Specifying model ------
model = models.Sequential()

# Deep hidden layers
model.add(layers.Conv2D(filters = 64,
                        kernel_size = (3,3),
                        activation = "relu",
                        input_shape = (64, 64, 3)))
model.add(layers.MaxPooling2D((2, 2)))

#Flatten and output layers
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

# Compiling model
model.compile(loss = 'binary_crossentropy',
                      optimizer = optimizers.Adam(learning_rate = 0.01),
                      metrics = ['accuracy'])

# ------ Fitting model ------

def calc_steps_epoch(path_to_datadir: str = "catdog_data/train", batch_size: int = 32) -> int:
    """
    Return how many steps per epoch (samples) hyperparameter
    """
    return ceil(len(listdir(path_to_datadir)) / batch_size) 

history = model.fit(
    train_generator,
    steps_per_epoch =  calc_steps_epoch("catdog_data/train", 32), # How ma
    epochs = 20,
    validation_data = val_generator,
    validation_steps = calc_steps_epoch("catdog_data/validation", 32),
    verbose = True
)

##############################
# Part three - Visualization #
##############################
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'b', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()


""""
#Function for plotting the models performance
def plot_history(H, epochs, output = "performance.png"):
    # visualize performance
    plt.style.use("fivethirtyeight")
    fig = plt.figure()
    plt.plot(np.arange(0, epochs), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, epochs), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, epochs), H.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, epochs), H.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.show()
    fig.savefig(output)

plot_history(history, len(list(epochs)))
"""