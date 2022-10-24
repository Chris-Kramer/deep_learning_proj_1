import numpy as np
from os import listdir
import tensorflow as tf
import matplotlib.image as mpimg
from matplotlib import pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

###############
# Load Model #
###############
model_reloaded = tf.keras.models.load_model('models/gustav_chris_model.h5')
model_reloaded.summary()

#################
# Visualization #
#################

# ------ Display Example Image ------
fig, ax = plt.subplots(2, 2, sharex=False, sharey=False, figsize=(6,6)) # create a grid of 2x2 images

for i in range(2):
    for j in range(2):
        ax[i][j].imshow(mpimg.imread('catdog_data/train/cats/cat.' + str(i*3+j) + '.jpg'))

plt.show() # # display plot

# ------ Data Augmentation Example ------

# ------ Filters ------

# ------ Activation Maps ------

# ------ Training/Validation Accuracy and Loss ------
fig = plt.figure(figsize=plt.figaspect(0.3))

ax = fig.add_subplot(1, 2, 1)
ax.plot(history.history['loss'], label='Training loss')
ax.plot(history.history['val_loss'], label='Test loss')
plt.title("Training and validation loss")
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

ax = fig.add_subplot(1, 2, 2)
ax.plot(history.history['accuracy'], label='Training accuracy')
ax.plot(history.history['val_accuracy'], label='Test accuracy')
plt.title("Training and validation accuracy")
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.show()