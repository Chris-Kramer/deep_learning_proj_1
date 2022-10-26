##############################
# Part one - Data generators #
##############################
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from os import listdir

def generate_data(targ_size: tuple[int, int], batch_size: int) -> ImageDataGenerator:
    """
    Creates datagenerators with the specified target dimensions and batchsizes
    """
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
        directory = "catdog_data/train",
        target_size = targ_size,
        color_mode = "rgb",
        batch_size = batch_size,
        class_mode='binary',
        shuffle = True) # Hypotese: Giver mere generaliserbar model, hvis shuffle er true

    # ------ Test data ------
    test_val_datagen = ImageDataGenerator(rescale=1./255) # Used for test and val, since we do not want to augment them
    test_generator = test_val_datagen.flow_from_directory(
        directory="catdog_data/test",
        target_size = targ_size, # Optimal solution = 224x224 eller 112x112
        color_mode="rgb",
        batch_size = batch_size,
        class_mode= "binary",
        shuffle = False,   
    )

    # ------ Validation data ------
    val_generator = test_val_datagen.flow_from_directory( 
        directory = "catdog_data/validation",
        target_size = targ_size,
        color_mode = "rgb",
        batch_size= batch_size,
        class_mode='binary')

    # Full training data set
    full_train_generator = train_datagen.flow_from_directory(
        directory = "catdog_data/train_val",
        target_size = targ_size,
        color_mode = "rgb",
        batch_size = batch_size,
        class_mode='binary',
        shuffle = True)

    return train_generator, val_generator, test_generator, full_train_generator

input_shape = (224, 224, 3)
batch_size = 32
train_generator, val_generator, test_generator, full_train_generator = generate_data(targ_size = (input_shape[0], input_shape[1]), batch_size = batch_size)

# ------ Convnet ------
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu',
                        input_shape=input_shape))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.summary()

# ------ Configuration ------
model.compile(
    loss = "binary_crossentropy",
    optimizer = optimizers.RMSprop(lr=1e-4),
    metrics = ["accuracy"])

# ------ Training the model ------
from math import ceil
def calc_steps_epoch(path_to_datadir: str = "catdog_data/train",
                     batch_size: int = 32,
                     factor: int | float = 1) -> int:
    """
    Return how many batches to go through in each epoch (is neccesarry since we use augmented data)
    E.g. if there is 300 images in our directory and a batch size of 32, and we have it as a factor of 1
    then we will go through celing( (1 * 300) / 32) = 10 batches
    But we can raise the factor so we might end up going through more than the lenght of the directory (if there is little data)
    """
    return int(ceil( (factor * len(listdir(path_to_datadir)))/ batch_size)) 

history = model.fit(
      train_generator,
      steps_per_epoch=50,
      epochs=50,
      validation_data=val_generator,
      validation_steps=20,
      verbose = True)

# ------ Loss and Accuracy ------
import matplotlib.pyplot as plt

accuracy = history.history["accuracy"]
val_accuracy = history.history["val_accuracy"]
loss = history.history["loss"]
val_loss = history.history["val_loss"]
epochs = range(1, len(accuracy) + 1)
plt.plot(epochs, accuracy, "bo", label="Training accuracy")
plt.plot(epochs, val_accuracy, "b", label="Validation accuracy")
plt.title("Training and validation accuracy")
plt.legend()
plt.figure()
plt.plot(epochs, loss, "bo", label="Training loss")
plt.plot(epochs, val_loss, "b", label="Validation loss")
plt.title("Training and validation loss")
plt.legend()
plt.show()

# ------ Evaluating model on test set ------
loss, accuracy = model.evaluate(test_generator, steps=calc_steps_epoch("catdog_data/test", batch_size, 1))
print(f"Test accuracy: {accuracy:.3f}")

#################
# Visualization #
#################1

# ------ Data Augmentation Example ------
import tensorflow
size = (300, 300)

elephants = tensorflow.keras.utils.load_img("catdog_data/train/cats/cat.1.jpg", target_size=size)
elephants = tensorflow.keras.utils.img_to_array(elephants)
imshow(elephants)

# ------ Filters ------

# ------ Activation Maps ------
