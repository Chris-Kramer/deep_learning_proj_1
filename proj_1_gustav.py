import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers
from tensorflow.keras import regularizers as regu

from utils import generate_data
from utils import calc_steps_epoch
from utils import plot_hist

from os import listdir
from math import ceil

######################
# Image Augmentation #
######################
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ImageDataGenerator rotation
datagen = ImageDataGenerator(
    rotation_range  = 40,
    fill_mode       = 'nearest',
    shear_range     = 0.2,
    zoom_range      = 0.2,
    horizontal_flip = True
    )

fnames = [os.path.join("catdog_data/train/cats/", fname) for
     fname in os.listdir("catdog_data/train/cats/")]

img_path = fnames[3]

img = tf.keras.utils.load_img(img_path, target_size=(150, 150))

x = tf.keras.utils.img_to_array(img)
x = x.reshape((1,) + x.shape)

i = 0
for batch in datagen.flow(x, batch_size=1):
    #plt.figure(i)
    ax = plt.subplot(2, 2, i + 1)
    imgplot = plt.imshow(tf.keras.utils.img_to_array(batch[0]).astype(np.uint8))
    i += 1
    if i % 4 == 0:
        break

plt.show()

##############################
# Part one - Data generators #
##############################
input_shape = (150, 150, 3)
batch_size = 32
train_generator, val_generator, test_generator, full_train_generator = generate_data(targ_size = (input_shape[0], input_shape[1]), batch_size = batch_size)

####################
# Part two - Model #
####################
model = models.Sequential()

# Conv 1
model.add(layers.Conv2D(filters = 32,
                        kernel_size = (3, 3),
                        kernel_regularizer = regu.l2(0.0001),
                        #bias_regularizer = regu.l2(0.0001),
                        #activity_regularizer = regu.l2(0.0001),
                        activation = 'relu',
                        input_shape = input_shape))
model.add(layers.MaxPooling2D(pool_size = 2))

model.add(layers.Conv2D(filters = 64,
                        kernel_size = (3, 3),
                        kernel_regularizer = regu.l2(0.0001),
                        #bias_regularizer = regu.l2(0.0001),
                        #activity_regularizer = regu.l2(0.0001),
                        activation = 'relu'))
model.add(layers.MaxPooling2D(pool_size = 2))

model.add(layers.Conv2D(filters = 128,
                        kernel_size = (3, 3),
                        kernel_regularizer = regu.l2(0.0001),
                        #bias_regularizer = regu.l2(0.0001),
                        #activity_regularizer = regu.l2(0.0001),
                        activation = 'relu'))
model.add(layers.MaxPooling2D(pool_size = 2))

model.add(layers.Conv2D(filters = 256,
                        kernel_size = (3, 3),
                        kernel_regularizer = regu.l2(0.0001),
                        #bias_regularizer = regu.l2(0.0001),
                        #activity_regularizer = regu.l2(0.0001),
                        activation = 'relu'))
model.add(layers.MaxPooling2D(pool_size = 2))

model.add(layers.Flatten())
model.add(layers.Dense(248,
                        kernel_regularizer = regu.l2(0.0001),
                        #bias_regularizer = regu.l2(0.0001),
                        #activity_regularizer = regu.l2(0.0001),
                       activation='relu'))

layers.Dropout(0.5)

model.add(layers.Dense(1, activation='sigmoid'))

model.compile(loss = 'binary_crossentropy',
                      optimizer = optimizers.Adam(learning_rate = 0.001),
                      metrics = ['accuracy'])

model.summary()

"""
model = models.Sequential()

# Conv 1
model.add(layers.Conv2D(filters = 32,
                        kernel_size = (3, 3),
                        kernel_regularizer = regu.l1_l2(l1=1e-5,l2=1e-4),
                        bias_regularizer = regu.l2(1e-4),
                        activity_regularizer = regu.l2(1e-5),
                        activation = 'relu',
                        input_shape = input_shape))


model.add(layers.MaxPooling2D(pool_size = 2))
# Conv 1
model.add(layers.Conv2D(filters = 64,
                        kernel_size = (3, 3),
                        kernel_regularizer = regu.l1_l2(l1=1e-5,l2=1e-4),
                        bias_regularizer = regu.l2(1e-4),
                        activity_regularizer = regu.l2(1e-5),
                        activation = 'relu'))


# Max pool 1
model.add(layers.MaxPooling2D(pool_size = 2))

# COnv 2
model.add(layers.Conv2D(filters = 64,
                        kernel_size = (3, 3),
                        kernel_regularizer = regu.l1_l2(l1=1e-5,l2=1e-4),
                        bias_regularizer = regu.l2(1e-4),
                        activity_regularizer = regu.l2(1e-5),
                        activation = 'relu'))


model.add(layers.MaxPooling2D(pool_size = 2))

# Conv 3
model.add(layers.Conv2D(filters = 64,
                        kernel_size = (3, 3),
                        kernel_regularizer = regu.l1_l2(l1=1e-5,l2=1e-4),
                        bias_regularizer = regu.l2(1e-4),
                        activity_regularizer = regu.l2(1e-5),
                        activation = 'relu'))


model.add(layers.MaxPooling2D(pool_size = 2))


model.add(layers.Flatten())
model.add(layers.Dense(112,
                        kernel_regularizer = regu.l1_l2(l1=1e-5,l2=1e-4),
                        bias_regularizer = regu.l2(1e-4),
                        activity_regularizer = regu.l2(1e-5),
                       activation='relu'))

layers.Dropout(.2)

model.add(layers.Dense(1, activation='sigmoid'))

model.compile(loss = 'binary_crossentropy',
                      optimizer = "Adam", #optimizers.Adam(learning_rate = 0.001),
                      metrics = ['accuracy'])

"""
#callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=7)
history = model.fit(
    train_generator,
    #steps_per_epoch =  calc_steps_epoch("catdog_data/train", batch_size), # How ma
    batch_size=batch_size,
    epochs = 50,
    validation_data = val_generator,
    #validation_steps = calc_steps_epoch("catdog_data/validation", batch_size),
    verbose = True#,
    #callbacks=[callback]
)

model.evaluate(test_generator, steps=calc_steps_epoch("catdog_data/test", batch_size, 1))

plot_hist(history)

#################
# Visualization #
#################

# ------ Filters ------
for layer in model.layers:
    if 'conv' not in layer.name:
        continue    
    filters , bias = layer.get_weights()
    print(layer.name , filters.shape)

filters , bias = model.layers[2].get_weights() # retrieve weights from the third hidden layer

f_min, f_max = filters.min(), filters.max() # normalize filter values to 0-1 so we can visualize them
filters = (filters - f_min) / (f_max - f_min)

import matplotlib.pyplot as plt

n_filters = 6
ix = 1
fig = plt.figure(figsize=(8,8))
for i in range(n_filters):
    # get the filters
    f = filters[:,:,:,i]
    for j in range(3):
        # subplot for 6 filters and 3 channels
        plt.subplot(n_filters,3,ix)
        plt.imshow(f[:,:,j] ,cmap='gray')
        ix+=1

# plot the filters 
plt.show()

# ------ Feature Maps ------
for i in range(len(model.layers)):
    layer = model.layers[i]
    if 'conv' not in layer.name:
        continue    
    print(i , layer.name , layer.output.shape)

from tensorflow.keras.models import Model

model = Model(inputs=model.inputs , outputs=model.layers[5].output)

from tensorflow.keras.preprocessing.image import load_img

image = load_img("catdog_data/train/cats/cat.1.jpg" , target_size=(150,150))

# convert the image to an array
from tensorflow.keras.preprocessing.image import img_to_array
image = img_to_array(image)

# expand dimensions so that it represents a single 'sample'
from numpy import expand_dims
image = expand_dims(image, axis=0)

# X
from tensorflow.keras.applications.vgg16 import preprocess_input
image = preprocess_input(image)

#calculating features_map
features = model.predict(image)

fig = plt.figure(figsize=(8,8))
for i in range(1,features.shape[3]+1):

    plt.subplot(8,8,i)
    plt.imshow(features[0,:,:,i-1] , cmap='gray')
    
plt.show()