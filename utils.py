from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from math import ceil



##############################
# Part one - Data generators #
##############################
def generate_data(targ_size: tuple[int, int], batch_size: int) -> ImageDataGenerator:
    """
    Creates datagenerators with the specified target dimensions and batchsizes
    """
    # ------ Training data ------
    train_datagen = ImageDataGenerator(
        rescale=1./255, # Scale between 0 and 1
        rotation_range = 20, # Random rotations of 20 degree 
        width_shift_range = 0.20, # Randomly Shift pixels between 0% and 20% in width
        height_shift_range = 0.20, # Randomly shift pixels between 0% and 20% in height
        shear_range=0.10, # Shear angle in counter-clockwise direction in degrees
        zoom_range=0.20, # Zoom in between 0 and 20 %
        horizontal_flip=True)

    train_generator = train_datagen.flow_from_directory(
        directory = r"catdog_data/train",
        target_size = targ_size,
        color_mode = "rgb",
        batch_size = batch_size,
        class_mode='binary',
        shuffle = True) # Hypotese: Giver mere generaliserbar model, hvis shuffle er true

    # ------ Test data ------
    test_val_datagen = ImageDataGenerator(rescale=1./255) # Used for test and val, since we do not want to augment them
    test_generator = test_val_datagen.flow_from_directory(
        directory= r"catdog_data/test",
        target_size = targ_size, # Optimal solution = 224x224 eller 112x112
        color_mode="rgb",
        batch_size = batch_size,
        class_mode= "binary")

    # ------ Validation data ------
    val_generator = test_val_datagen.flow_from_directory( 
        directory = r"catdog_data/validation",
        target_size = targ_size,
        color_mode = "rgb",
        batch_size= batch_size,
        class_mode='binary')

    # Full training data set
    full_train_generator = train_datagen.flow_from_directory(
        directory = r"catdog_data/train_val",
        target_size = targ_size,
        color_mode = "rgb",
        batch_size = batch_size,
        class_mode='binary',
        shuffle = True)

    return train_generator, val_generator, test_generator, full_train_generator


def calc_steps_epoch(n_samples: int,
                     batch_size: int = 32,
                     factor: int | float = 1) -> int:
    """
    Return how many batches to go through in each epoch (is neccesarry since we use augmented data)
    E.g. if there is 300 samples a batch size of 32, and we have it as a factor of 1
    then we will go through celing( (1 * 300) / 32) = 10 batches
    But we can raise the factor so we might end up going through more than the lenght of the directory (if there is little data and we use augmentation)
    """
    return int(ceil( (factor * n_samples) / batch_size)) 

def plot_hist(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)

    plt.plot(epochs, acc, 'b', label='Training accuracy')
    plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.legend()

    plt.figure()

    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.show()