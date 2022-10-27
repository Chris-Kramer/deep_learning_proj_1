import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers
from tensorflow.keras import regularizers as regu

from utils import generate_data
from utils import calc_steps_epoch
from utils import plot_hist
from os import listdir


# ----- Model 2.0 -----
# Desc:
# Hyperparams:
#   - kernels = (3, 3)
#   - Activation = relu
#   - LR = 0.0001
# Architecture
# Conv(32) -> MaxPool -->
# Conv(64) -> MaxPool -->
# Conv(64) -> MaxPool -->
# Conv(64) -> MaxPool -->
# Flatten -> Dense(248) -->
# Dropout(0.2) -> Sigmoid

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

TEST_STEPS = calc_steps_epoch(n_samples = n_test_samples,
                               batch_size = BATCH_SIZE,
                               factor = 1)
FULL_TRAIN_STEPS = calc_steps_epoch(n_samples = n_full_train_samples,
                               batch_size = BATCH_SIZE,
                               factor = 1)
train_generator, val_generator, test_generator, full_train_generator = generate_data(targ_size = (INPUT_SHAPE[0], INPUT_SHAPE[1]),
                                                                                     batch_size = BATCH_SIZE)
# Model 2
model = models.Sequential()

# Conv(32) -> MaxPool -->
model.add(layers.Conv2D(filters = 32,
                        kernel_size = (3, 3),
                        input_shape = INPUT_SHAPE))
model.add(layers.MaxPooling2D(pool_size = 2))

# Conv(64) -> MaxPool -->
model.add(layers.Conv2D(filters = 64,
                        kernel_size = (3, 3),
                        activation = 'relu'))
model.add(layers.MaxPooling2D(pool_size = 2))

# Conv(64) -> MaxPool -->
model.add(layers.Conv2D(filters = 64,
                        kernel_size = (3, 3),
                        activation = 'relu'))
model.add(layers.MaxPooling2D(pool_size = 2))

# Conv(64) -> MaxPool -->
model.add(layers.Conv2D(filters = 64,
                        kernel_size = (3, 3),
                        activation = 'relu'))
model.add(layers.MaxPooling2D(pool_size = 2))

# Flatten -> Dense(248) -->
model.add(layers.Flatten())
model.add(layers.Dense(248, activation='relu'))

# Dropout(0.2) -> Sigmoid
layers.Dropout(.2)
model.add(layers.Dense(1, activation='sigmoid'))


model.compile(loss = 'binary_crossentropy',
                      optimizer = optimizers.Adam(learning_rate = 0.0001),
                      metrics = ['accuracy'])

callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=7)
history = model.fit(
    full_train_generator,
    steps_per_epoch =  FULL_TRAIN_STEPS,
    epochs = 100,
    verbose = True,
    callbacks=[callback]
)
model.evaluate(test_generator, steps = TEST_STEPS) # Acc: 77,25 % 
model.save("models/final_model.h5")
