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

INPUT_SHAPE = (112, 112, 3)
BATCH_SIZE = 64

# Calcualte length of data 
n_train_samples = len(listdir("catdog_data/train/cats")) + len(listdir("catdog_data/train/dogs")) # Number of training data
n_val_samples = len(listdir("catdog_data/validation/cats")) + len(listdir("catdog_data/validation/dogs")) # Number of validaiton data
n_test_samples = len(listdir("catdog_data/test/cats")) + len(listdir("catdog_data/test/dogs")) # number of test data
n_full_train_samples = n_train_samples + n_val_samples # Number of full data

TRAIN_STEPS = calc_steps_epoch(n_samples = n_train_samples,
                               batch_size = BATCH_SIZE,
                               factor = 1)
VAL_STEPS = calc_steps_epoch(n_samples = n_val_samples,
                               batch_size = BATCH_SIZE,
                               factor = 1)
TEST_STEPS = calc_steps_epoch(n_samples = n_test_samples,
                               batch_size = BATCH_SIZE,
                               factor = 1)
FULL_TRAIN_STEPS = calc_steps_epoch(n_samples = n_full_train_samples,
                               batch_size = BATCH_SIZE,
                               factor = 1)
train_generator, val_generator, test_generator, full_train_generator = generate_data(targ_size = (INPUT_SHAPE[0], INPUT_SHAPE[1]),
                                                                                     batch_size = BATCH_SIZE)

####################
# Part two - Model #
####################

model = models.Sequential()
# Conv 1
model.add(layers.Conv2D(filters = 16,
                        kernel_size = (3, 3),
                        activation = 'relu',
                        input_shape = INPUT_SHAPE))


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
    steps_per_epoch =  TRAIN_STEPS,
    epochs = 20,
    validation_data = val_generator,
    validation_steps = VAL_STEPS,
    verbose = True,
    callbacks=[callback]
)

model.evaluate(test_generator, steps = TEST_STEPS)


plot_hist(history)
