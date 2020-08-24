import matplotlib.pyplot as plt
import numpy as np
import math

## Import the keras API
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import InputLayer, Input
from tensorflow.python.keras.layers import Reshape, MaxPooling2D
from tensorflow.python.keras.layers import Conv2D, Dense, Flatten

from tensorflow.python.keras.models import load_model
from tensorflow.python.keras import backend as K

#----------------------------------------------------------------------------------------------------------------------

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
data = input_data.read_data_sets('data/MNIST/', one_hot=True)
data.test.cls = np.argmax(data.test.labels, axis=1)

#----------------------------------------------------------------------------------------------------------------------


# We know that MNIST images are 28 pixels in each dimension.
img_size = 28

# Images are stored in one-dimensional arrays of this length.
img_size_flat = img_size * img_size

# Tuple with height and width of images used to reshape arrays.
# This is used for plotting the images.
img_shape = (img_size, img_size)

# Tuple with height, width and depth used to reshape arrays.
# This is used for reshaping in Keras.
img_shape_full = (img_size, img_size, 1)

# Number of colour channels for the images: 1 channel for gray-scale.
num_channels = 1

# Number of classes, one class for each of 10 digits.
num_classes = 10

#----------------------------------------------------------------------------------------------------------------------

def plot_images(images, cls_true, cls_pred=None):
    assert len(images) == len(cls_true) == 9

    # Create figure with 3x3 sub-plots.
    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        # Plot image.
        ax.imshow(images[i].reshape(img_shape), cmap='binary')

        # Show true and predicted classes.
        if cls_pred is None:
            xlabel = "True: {0}".format(cls_true[i])
        else:
            xlabel = "True: {0}, Pred: {1}".format(cls_true[i], cls_pred[i])

        # Show the classes as the label on the x-axis.
        ax.set_xlabel(xlabel)

        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()

#----------------------------------------------------------------------------------------------------------------------

# Get the first images from the test-set.
images = data.test.images[0:9]

# Get the true classes for those images.
cls_true = data.test.cls[0:9]

# Plot the images and labels using our helper-function above.
plot_images(images=images, cls_true=cls_true)

#----------------------------------------------------------------------------------------------------------------------

def plot_example_errors(cls_pred):
    # cls_pred is an array of the predicted class-number for
    # all images in the test-set.

    # Boolean array whether the predicted class is incorrect.
    incorrect = (cls_pred != data.test.cls)

    # Get the images from the test-set that have been
    # incorrectly classified.
    images = data.test.images[incorrect]

    # Get the predicted classes for those images.
    cls_pred = cls_pred[incorrect]

    # Get the true classes for those images.
    cls_true = data.test.cls[incorrect]

    # Plot the first 9 images.
    plot_images(images=images[0:9],
                cls_true=cls_true[0:9],
                cls_pred=cls_pred[0:9])

#----------------------------------------------------------------------------------------------------------------------

# Comienza la construcción del modelo Keras Sequential.
model = Sequential()

# Agrega una capa de entrada que es similar a un feed_dict en TensorFlow.
# Tenga en cuenta que la forma de entrada debe ser una tupla que contenga el tamaño de la imagen.
model.add(InputLayer(input_shape=(img_size_flat,)))

# La entrada es una matriz aplanada con 784 elementos (img_size * img_size),
# pero las capas convolucionales esperan imágenes con forma (28, 28, 1), por tanto hacemos un reshape
model.add(Reshape(img_shape_full))

# Primera capa convolucional con ReLU-activation y max-pooling.
model.add(Conv2D(kernel_size=5, strides=1, filters=16, padding='same',
                 activation='relu', name='layer_conv1'))
model.add(MaxPooling2D(pool_size=2, strides=2))

# Segunda capa convolucional con ReLU-activation y max-pooling.
model.add(Conv2D(kernel_size=5, strides=1, filters=36, padding='same',
                 activation='relu', name='layer_conv2'))
model.add(MaxPooling2D(pool_size=2, strides=2))

# Aplanar la salida de 4 niveles de las capas convolucionales
# a 2-rank que se puede ingresar a una capa totalmente conectada
model.add(Flatten())

# Primera capa completamente conectada  con ReLU-activation.
model.add(Dense(128, activation='relu'))

# Última capa totalmente conectada con activación de softmax
# para usar en la clasificación.
model.add(Dense(num_classes, activation='softmax'))

#----------------------------------------------------------------------------------------------------------------------

from tensorflow.python.keras.optimizers import Adam
optimizer = Adam(lr=1e-3)

#----------------------------------------------------------------------------------------------------------------------

model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

#----------------------------------------------------------------------------------------------------------------------

# entrenamos el modelo...
model.fit(x=data.train.images,
          y=data.train.labels,
          epochs=1, batch_size=128)

#----------------------------------------------------------------------------------------------------------------------

result = model.evaluate(x=data.test.images,
                        y=data.test.labels)

#----------------------------------------------------------------------------------------------------------------------

for name, value in zip(model.metrics_names, result):
    print(name, value)

#----------------------------------------------------------------------------------------------------------------------

print("{0}: {1:.2%}".format(model.metrics_names[1], result[1]))

#----------------------------------------------------------------------------------------------------------------------

images = data.test.images[0:9]

#----------------------------------------------------------------------------------------------------------------------

cls_true = data.test.cls[0:9]

#----------------------------------------------------------------------------------------------------------------------

y_pred = model.predict(x=images)

#----------------------------------------------------------------------------------------------------------------------

cls_pred = np.argmax(y_pred,axis=1)
print(cls_pred)
print(cls_true)

#----------------------------------------------------------------------------------------------------------------------

plot_images(images=images,
            cls_true=cls_true,
            cls_pred=cls_pred)

#----------------------------------------------------------------------------------------------------------------------

# Crea una capa de entrada que es similar a un feed_dict en TensorFlow.
# Tenga en cuenta que la forma de entrada debe ser una tupla que contenga el tamaño de la imagen.
inputs = Input(shape=(img_size_flat,))

# Variable utilizada para construir la red neuronal
net = inputs

# La entrada es una imagen como una matriz aplanada con 784 elementos.
# Pero las capas convolucionales esperan imágenes con forma (28, 28, 1)
net = Reshape(img_shape_full)(net)

# Primera capa convolucional con ReLU-activation y max-pooling.
net = Conv2D(kernel_size=5, strides=1, filters=16, padding='same',
             activation='relu', name='layer_conv1')(net)
net = MaxPooling2D(pool_size=2, strides=2)(net)

# Segunda capa convolucional con ReLU-activation y max-pooling
net = Conv2D(kernel_size=5, strides=1, filters=36, padding='same',
             activation='relu', name='layer_conv2')(net)
net = MaxPooling2D(pool_size=2, strides=2)(net)

# Aplanar la salida de la capa conv de 4-dim a 2-dim.
net = Flatten()(net)

# Primera capa completamente conectada  con ReLU-activation.
net = Dense(128, activation='relu')(net)

# Última capa totalmente conectada  con activación de softmax
# por lo que se puede utilizar para la clasificación.
net = Dense(num_classes, activation='softmax')(net)

# Salida de la red neuronal
outputs = net

#----------------------------------------------------------------------------------------------------------------------

from tensorflow.python.keras.models import Model

#----------------------------------------------------------------------------------------------------------------------

model2 = Model(inputs=inputs, outputs=outputs)

#----------------------------------------------------------------------------------------------------------------------

model2.compile(optimizer='rmsprop',
               loss='categorical_crossentropy',
               metrics=['accuracy'])

#----------------------------------------------------------------------------------------------------------------------

model2.fit(x=data.train.images,
           y=data.train.labels,
           epochs=3, batch_size=128)

#----------------------------------------------------------------------------------------------------------------------

result = model2.evaluate(x=data.test.images,
                         y=data.test.labels)

#----------------------------------------------------------------------------------------------------------------------

for name, value in zip(model2.metrics_names, result):
    print(name, value)

#----------------------------------------------------------------------------------------------------------------------

path_model = 'model.keras'

#----------------------------------------------------------------------------------------------------------------------

model2.save(path_model)

#----------------------------------------------------------------------------------------------------------------------

del model2

#----------------------------------------------------------------------------------------------------------------------

model3 = load_model(path_model)

#----------------------------------------------------------------------------------------------------------------------

images = data.test.images[0:9]
cls_true = data.test.cls[0:9]

#----------------------------------------------------------------------------------------------------------------------

y_pred = model3.predict(x=images)

#----------------------------------------------------------------------------------------------------------------------

cls_pred = np.argmax(y_pred, axis=1)
print(cls_pred)
print(cls_true)

#----------------------------------------------------------------------------------------------------------------------

plot_images(images=images,
            cls_pred=cls_pred,
            cls_true=cls_true)

#----------------------------------------------------------------------------------------------------------------------

model3.summary()

#----------------------------------------------------------------------------------------------------------------------

# La capa de entrada tiene el índice 0.
layer_input = model3.layers[0]
print(layer_input)


# La primera capa convolucional tiene el índice 1.
layer_conv1 = model3.layers[2]
print(layer_conv1)

# La segunda capa convolucional tiene el índice 1.
layer_conv2 = model3.layers[4]

#----------------------------------------------------------------------------------------------------------------------

def plot_conv_weights(weights, input_channel=0):
    # Get the lowest and highest values for the weights.
    # This is used to correct the colour intensity across
    # the images so they can be compared with each other.
    w_min = np.min(weights)
    w_max = np.max(weights)

    # Number of filters used in the conv. layer.
    num_filters = weights.shape[3]

    # Number of grids to plot.
    # Rounded-up, square-root of the number of filters.
    num_grids = math.ceil(math.sqrt(num_filters))

    # Create figure with a grid of sub-plots.
    fig, axes = plt.subplots(num_grids, num_grids)

    # Plot all the filter-weights.
    for i, ax in enumerate(axes.flat):
        # Only plot the valid filter-weights.
        if i<num_filters:
            # Get the weights for the i'th filter of the input channel.
            # See new_conv_layer() for details on the format
            # of this 4-dim tensor.
            img = weights[:, :, input_channel, i]

            # Plot image.
            ax.imshow(img, vmin=w_min, vmax=w_max,
                      interpolation='nearest', cmap='seismic')

        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()

#----------------------------------------------------------------------------------------------------------------------

weights_conv1 = layer_conv1.get_weights()[0]
print(weights_conv1.shape)

#----------------------------------------------------------------------------------------------------------------------

plot_conv_weights(weights=weights_conv1, input_channel=0)

#----------------------------------------------------------------------------------------------------------------------

weights_conv2 = layer_conv2.get_weights()[0]
print(weights_conv2.shape)
plot_conv_weights(weights=weights_conv2, input_channel=0)

#----------------------------------------------------------------------------------------------------------------------

def plot_conv_output(values):
    # Number of filters used in the conv. layer.
    num_filters = values.shape[3]

    # Number of grids to plot.
    # Rounded-up, square-root of the number of filters.
    num_grids = math.ceil(math.sqrt(num_filters))

    # Create figure with a grid of sub-plots.
    fig, axes = plt.subplots(num_grids, num_grids)

    # Plot the output images of all the filters.
    for i, ax in enumerate(axes.flat):
        # Only plot the images for valid filters.
        if i<num_filters:
            # Get the output image of using the i'th filter.
            img = values[0, :, :, i]

            # Plot image.
            ax.imshow(img, interpolation='nearest', cmap='binary')

        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()

#----------------------------------------------------------------------------------------------------------------------

def plot_image(image):
    plt.imshow(image.reshape(img_shape),
               interpolation='nearest',
               cmap='binary')

    plt.show()

#----------------------------------------------------------------------------------------------------------------------

image1 = data.test.images[0]
plot_image(image1)

#----------------------------------------------------------------------------------------------------------------------

output_conv2 = Model(inputs=layer_input.input,
                     outputs=layer_conv2.output)

#----------------------------------------------------------------------------------------------------------------------

layer_output2 = output_conv2.predict(np.array([image1]))
print(layer_output2.shape)

#----------------------------------------------------------------------------------------------------------------------

plot_conv_output(values=layer_output2)

#----------------------------------------------------------------------------------------------------------------------
