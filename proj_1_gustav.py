import keras
import tensorflow as tf
from matplotlib import pyplot as plt

###############
# Load Model #
###############
model_reloaded = tf.keras.models.load_model('models/gustav_chris_model.h5')
model_reloaded.summary()



#################
# Visualization #
#################

# ------ Data Augmentation ------
plt.figure(figsize=(10, 10))
for images, _ in train_dataset.take(1):
    for i in range(9):
        augmented_images = data_augmentation(images)
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(augmented_images[0].numpy().astype("uint8"))
        plt.axis("off")


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