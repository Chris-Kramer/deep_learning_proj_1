import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers
from tensorflow.keras.regularizers import l2

from utils import generate_data
from utils import calc_steps_epoch
from utils import plot_hist

from os import listdir
from math import ceil

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
##############################
# Part one - Data generators #
##############################
input_shape = (224, 224, 3)
batch_size = 24
train_generator, val_generator, test_generator, full_train_generator = generate_data(targ_size = (input_shape[0], input_shape[1]), batch_size = batch_size)

####################
# Part two - Model #
####################

model = models.Sequential()
# Conv 1
model.add(layers.Conv2D(filters = 16,
                        kernel_size = (3, 3),
                        activation = 'relu',
                        input_shape = input_shape))


# Max pool 1
model.add(layers.MaxPooling2D(pool_size = 2, strides = 2))

# COnv 2
model.add(layers.Conv2D(filters = 64,
                        kernel_size = (5, 5),
                        activation = 'relu'))

# Conv 3
model.add(layers.Conv2D(filters = 64,
                        kernel_size = (3, 3),
                        activation = 'relu'))

# Max pool 2
model.add(layers.MaxPooling2D(pool_size =(3, 3), strides = 1))

# Conv 4
model.add(layers.Conv2D(filters = 128,
                        kernel_size = (3, 3),
                        activation = 'relu'))

# Conv 5 
model.add(layers.Conv2D(filters = 256,
                        kernel_size = (3, 3),
                        activation = 'relu'))



# Max pool 3
model.add(layers.MaxPooling2D(pool_size = (3, 3)))



l2_regu = l2(0.0005)
model.add(layers.Flatten())
model.add(layers.Dense(512,
                       kernel_regularizer = l2_regu,
                       activation='relu'))

layers.Dropout(.5)

model.add(layers.Dense(512,
                       kernel_regularizer = l2_regu,
                       activation='relu'))

model.add(layers.Dense(1, activation='sigmoid'))
# Compile model

model.compile(loss = 'binary_crossentropy',
                      optimizer = optimizers.Adam(learning_rate = 0.00001),
                      metrics = ['accuracy'])

"""
model.add(layers.Conv2D(8,(7,7), activation = 'relu', input_shape = input_shape))
model.add(layers.Conv2D(8,(7,7), activation = 'relu'))
model.add(layers.MaxPooling2D(pool_size = (2,2)))

model.add(layers.Conv2D(16,(7,7), activation = 'relu'))
model.add(layers.Conv2D(16,(7,7), activation = 'relu'))
model.add(layers.MaxPooling2D(pool_size = (2,2)))

# Adds a densely-connected layer with 64 units to the model:
model.add(layers.Conv2D(32,(5,5), activation = 'relu'))
model.add(layers.Conv2D(32,(5,5), activation = 'relu'))
model.add(layers.MaxPooling2D(pool_size = (2,2)))

model.add(layers.Conv2D(64,(3,3), activation = 'relu'))
model.add(layers.Conv2D(64,(3,3), activation = 'relu'))
model.add(layers.MaxPooling2D(pool_size = (2,2)))

model.add(layers.Conv2D(128,(2,2), activation = 'relu'))
model.add(layers.Conv2D(128,(2,2), activation = 'relu'))
model.add(layers.MaxPooling2D(pool_size = (2,2)))

l2_regu = l2(0.0005)
model.add(layers.Flatten())
model.add(layers.Dense(512,
                       #kernel_regularizer = l2_regu,
                       activation='relu'))
                    
#layers.Dropout(.5)
model.add(layers.Dense(1, activation='sigmoid'))
# Compile model
model.compile(loss = 'binary_crossentropy',
                      optimizer = optimizers.Adam(learning_rate = 0.0001),
                      metrics = ['accuracy'])

"""
callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=7)
history = model.fit(
    train_generator,
    steps_per_epoch =  calc_steps_epoch("catdog_data/train", batch_size), # How ma
    batch_size=batch_size,
    epochs = 400,
    validation_data = val_generator,
    validation_steps = calc_steps_epoch("catdog_data/validation", batch_size),
    verbose = True,
    callbacks=[callback]
)

model.evaluate(test_generator, steps=calc_steps_epoch("catdog_data/test", batch_size, 1))


plot_hist(history)
