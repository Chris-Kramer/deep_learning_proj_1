
import numpy as np
from keras.datasets import mnist
from keras.utils import to_categorical
import matplotlib.pyplot as plt

# Loads the data
(train_data, train_labels), (test_data, test_labels) = mnist.load_data()

# Plots a single digit from the data
plt.imshow(train_data[1])
plt.show()

num_classes = len(np.unique(train_labels))

train_labels = to_categorical(train_labels, num_classes)
test_labels = to_categorical(test_labels, num_classes)

# Exercise 1

def get_number_indexes(labels):
	from collections import OrderedDict
	indx = {}
	for i, f in enumerate(labels):
		if len(indx) == 10:
			break
		if f not in indx:
			indx[f] = i
	return list(OrderedDict(sorted(indx.items())).values())

indices = get_number_indexes(mnist.load_data()[0][1])
fig, ((ax1, ax3, ax5, ax7, ax9), (ax2, ax4, ax6, ax8, ax10)) = plt.subplots(ncols=5, nrows=2, figsize=(15, 8))
for ax, indx in zip([ax1, ax3, ax5, ax7, ax9, ax2, ax4, ax6, ax8, ax10], indices):
	ax.imshow(train_data[indx])

plt.show()

# Reshape tensors

print('Training data before reshaping:', train_data.shape, train_labels.shape, '\nTesting data before reshaping:', test_data.shape, test_labels.shape)
train_data = train_data.astype('float32') / 255
train_data = train_data.reshape((60000, 28, 28, 1))
test_data = test_data.astype('float32') / 255
test_data = test_data.reshape((10000, 28, 28, 1))
print('Training data after reshaping:', train_data.shape, train_labels.shape, '\nTesting data after reshaping:', test_data.shape, test_labels.shape)


# Exercise 2

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, Dropout, LeakyReLU
from keras.models import Model

inputs = Input((28, 28, 1))
x = Conv2D(64, (3, 3), activation='relu')(inputs)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(32, (3, 3), activation="relu")(x)
x = MaxPooling2D((2, 2))(x)
x = Flatten()(x)
x = Dense(64, activation='relu')(x)
output = Dense(10, activation='softmax')(x)
model = Model(inputs, output)
model.summary()
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
model.save('pre-model.h5')

# Normal

model.fit(train_data, train_labels, epochs=5, batch_size=64, validation_data=(test_data, test_labels))

# Datagenerator

from keras.preprocessing.image import ImageDataGenerator

train_gen = ImageDataGenerator(
	rotation_range=20,
	width_shift_range=0.2,
	height_shift_range=0.2)

model.fit(train_gen.flow(train_data, train_labels, batch_size=64), epochs=5, validation_data=(test_data, test_labels))

model.save('post-model.h5')


# Exercise 3

# Incorrectly labelled datapoints:
predictions = model.predict(test_data).argmax(axis=-1)
(train_data, train_labels), (test_data, test_labels) = mnist.load_data()
wrong_data = test_data[test_labels != predictions]
wrong_labels = predictions[test_labels != predictions]
correct_labels = test_labels[test_labels != predictions]

indices = get_number_indexes(wrong_labels)
fig, ((ax1, ax3, ax5, ax7, ax9), (ax2, ax4, ax6, ax8, ax10)) = plt.subplots(ncols=5, nrows=2, figsize=(15, 8))
for ax, indx in zip([ax1, ax3, ax5, ax7, ax9, ax2, ax4, ax6, ax8, ax10], indices):
	ax.imshow(wrong_data[indx])
	ax.set_title(f'Prediction: {wrong_labels[indx]}. True label: {correct_labels[indx]}')
	
plt.show()

# a

from keras.models import load_model

def viz_kernels(model):
	kernels = [f for f in model.layers if 'conv' in f.name][0].get_weights()[0][:, :, 0, :]
	for i in range(0, 32):
		plt.subplot(4, 8, i+1)
		plt.imshow(kernels[:, :, i], interpolation="nearest", cmap="gray")
	plt.show()

viz_kernels(load_model('pre-model.h5'))
viz_kernels(load_model('post-model.h5'))

# b

model = load_model('post-model.h5')
model = Model(model.inputs, model.layers[1].output)
for f in get_number_indexes(test_labels):
	feature_map_data = test_data[f].reshape(1, 28, 28, 1)
	feature_maps = model.predict(feature_map_data)
	square = 5
	for ix in range(np.square(square)):
		ax = plt.subplot(square, square, ix+1)
		ax.set_xticks([])
		ax.set_yticks([])
		# plot filter channel in grayscale
		plt.imshow(feature_maps[0, :, :, ix], cmap='gray')
	plt.show()


# c

import keras.backend as K
import tensorflow as tf
tf.compat.v1.disable_eager_execution()  # Error supression

model = load_model('post-model.h5')
layer_dict = dict([(layer.name, layer) for layer in model.layers[1:]])
input_img = model.inputs[0]

layer_name = "conv2d"
output_layer = layer_dict[layer_name]
layer_output = output_layer.output

filter_index = 3
loss = K.mean(layer_output[:, :, :, filter_index])

grads = K.gradients(loss, input_img)[0]

# normalization trick: we normalize the gradient
grads /= (K.sqrt(K.mean(K.square(grads))) + K.epsilon())

# this function returns the loss and grads given the input picture
iterate = K.function([input_img], [loss, grads])

output_dim = (28, 28)

input_img_data = np.random.random((1, output_dim[0], output_dim[1], 1))
# input_img_data = (input_img_data - 0.5) * 20 + 128

for _ in range(20):
	loss_value, grads_value = iterate([input_img_data])
	input_img_data += grads_value * 1

input_img_data -= input_img_data.mean()
input_img_data /= (input_img_data.std() + K.epsilon())
input_img_data *= 0.25

# clip to [0, 1]
input_img_data += 0.5
input_img_data = np.clip(input_img_data, 0, 1)

# convert to RGB array
input_img_data *= 255
input_img_data = np.clip(input_img_data, 0, 255).astype('uint8')


def _draw_filters(filters, n=None):
	"""Draw the best filters in a nxn grid.
	# Arguments
		filters: A List of generated images and their corresponding losses
				 for each processed filter.
		n: dimension of the grid.
		   If none, the largest possible square will be used
	"""
	if n is None:
		n = int(np.floor(np.sqrt(len(filters))))
	# the filters that have the highest loss are assumed to be better-looking.
	# we will only keep the top n*n filters.
	filters.sort(key=lambda x: x[1], reverse=True)
	filters = filters[:n * n]
	# build a black picture with enough space for
	# e.g. our 8 x 8 filters of size 412 x 412, with a 5px margin in between
	MARGIN = 5
	width = n * output_dim[0] + (n - 1) * MARGIN
	height = n * output_dim[1] + (n - 1) * MARGIN
	stitched_filters = np.zeros((width, height, 3), dtype='uint8')
	# fill the picture with our saved filters
	for i in range(n):
		for j in range(n):
			img, _ = filters[i * n + j]
			width_margin = (output_dim[0] + MARGIN) * i
			height_margin = (output_dim[1] + MARGIN) * j
			stitched_filters[
			width_margin: width_margin + output_dim[0],
			height_margin: height_margin + output_dim[1], :] = img
	# save the result to disk
	from tensorflow.keras.preprocessing.image import save_img
	save_img('visualize_filter.png'.format(layer_name, n), stitched_filters)

_draw_filters([(input_img_data, 0.1)])