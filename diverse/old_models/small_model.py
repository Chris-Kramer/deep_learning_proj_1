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

# ########
# Test 1 #
# ########
# MODEL 1.0
# 32 batcth size 
# 20  epochs
# input_shape (112, 112)
# data augmentering TRUE
# 73 % acc

# ########
# Test 2 #
##########
# MODEL 1.0
# 64 batcth size 
# 20  epochs
# input_shape (112, 112)
# data augmentering TRUE
# 75%  acc (stort set samme som test 1)

##########
# Test 3 #
##########
# Model 1.1
# 64 batcth size 
# 20  epochs
# input_shape (112, 112)
# data augmentering TRUE
# Acc: 69

##########
# Test 4 #
##########
# Model 2.0
# 64 batcth size 
# 20  epochs
# input_shape (112, 112)
# data augmentering TRUE
# Acc: 73.75 (Lader til at have mere potentiale)


##########
# Test 5 #
##########
# Model 3.0
# 64 batcth size 
# 20  epochs
# input_shape (112, 112)
# data augmentering TRUE
# Acc: 73.5 (Lader til at have mere potentiale)

##########
# Test 6 #
##########
# Model 2.0
# 32 batcth size 
# 20  epochs
# input_shape (112, 112)
# data augmentering TRUE
# Acc: 72.00 (Lader til at overfitte)

##########
# Test 7 #
##########
# Model 2.0
# 64 batcth size 
# 20  epochs
# input_shape (112, 112)
# data augmentering TRUE
# NY LR på 0.0001
# Acc: 72.50 (Lader til at være den bedste, da den ikke overfitter)

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

print(train_generator)

####################
# Part two - Model #
####################

#------ Model 1.0 ------
# Hyperparams:
#   - kernels = (3, 3)
#   - Activation = relu
#   - LR = 0.001
# Architecture
# Conv(32) -> MaxPool -->
# Conv(64) -> MaxPool -->
# Conv(64) -> MaxPool -->
# Flatten -> Dense(64) -->
# Dropout(0.2) -> Sigmoid


    # Model 1.1
    # Desc: Same as model 1 but no dropout at the end
    # Hyperparams:
    #   - kernels = (3, 3)
    #   - Activation = relu
    #   - LR = 0.001
    # Architecture
    # Conv(32) -> MaxPool -->
    # Conv(64) -> MaxPool -->
    # Conv(64) -> MaxPool -->
    # Flatten -> Dense(64) -> Sigmoid


# ----- Model 2.0 -----
# Desc:
# Hyperparams:
#   - kernels = (3, 3)
#   - Activation = relu
#   - LR = 0.001
# Architecture
# Conv(32) -> MaxPool -->
# Conv(64) -> MaxPool -->
# Conv(64) -> MaxPool -->
# Conv(64) -> MaxPool -->
# Flatten -> Dense(248) -->
# Dropout(0.2) -> Sigmoid


# ----- Model 3.0 -----
# Hyperparams:
#   - kernels = (3, 3)
#   - Activation = relu
#   - LR = 0.001
# Architecture
# Conv(32) -> MaxPool -->
# Conv(64) -> MaxPool -->
# Conv(128) -> MaxPool -->
# Flatten -> Dense(128) -->
# Dropout(0.2) -> Sigmoid

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

history = model.fit(
    train_generator,
    steps_per_epoch =  TRAIN_STEPS, #calc_steps_epoch("catdog_data/train", batch_size), # How ma
    epochs = 20,
    validation_data = val_generator,
    validation_steps =  VAL_STEPS, #calc_steps_epoch("catdog_data/validation", batch_size),
    verbose = True
)
model.evaluate(test_generator, steps = TEST_STEPS)


plot_hist(history)