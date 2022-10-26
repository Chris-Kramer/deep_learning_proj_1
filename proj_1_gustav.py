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

model.add(layers.Conv2D(filters = 64,
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

layers.Dropout(.2)

model.add(layers.Dense(1, activation='sigmoid'))

model.compile(loss = 'binary_crossentropy',
                      optimizer = optimizers.Adam(learning_rate = 0.001),
                      metrics = ['accuracy'])
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
    epochs = 100,
    validation_data = val_generator,
    #validation_steps = calc_steps_epoch("catdog_data/validation", batch_size),
    verbose = True#,
    #callbacks=[callback]
)

model.evaluate(test_generator, steps=calc_steps_epoch("catdog_data/test", batch_size, 1))


plot_hist(history)