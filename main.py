import numpy as np
from keras.datasets import mnist
from keras.layers import BatchNormalization, Conv2D, Conv2DTranspose, Dense, Flatten, Reshape
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential

from model import RegularizedGAN


def discriminator():
    network = Sequential(name='discriminator')
    network.add(Conv2D(df_dim, (kernel_height, kernel_width), padding='same', strides=(strides_height, strides_width), input_shape=image_shape))
    network.add(LeakyReLU(alpha=0.2))
    network.add(Conv2D(df_dim * 2, (kernel_height, kernel_width), padding='same', strides=(strides_height, strides_width)))
    network.add(LeakyReLU(alpha=0.2))
    network.add(BatchNormalization())
    network.add(Conv2D(df_dim * 4, (kernel_height, kernel_width), padding='same', strides=(strides_height, strides_width)))
    network.add(LeakyReLU(alpha=0.2))
    network.add(BatchNormalization())
    network.add(Conv2D(df_dim * 8, (kernel_height, kernel_width), padding='same', strides=(strides_height, strides_width)))
    network.add(LeakyReLU(alpha=0.2))
    network.add(BatchNormalization())
    network.add(Flatten())
    network.add(Dense(1, activation='linear'))  # Sigmoid will be applied by RegGAN later
    # network.summary()
    return network


def infer_shape():
    shape = image_dims
    for layer in range(4):
        shape = tuple(int(np.ceil(shape[i] / deconvolution_shape[i])) for i in range(len(shape)))
    return shape + (gf_dim * 8,)


def generator():
    projection_shape = infer_shape()
    network = Sequential(name='generator')
    network.add(Dense(np.prod(projection_shape), activation='relu', input_dim=latent_dims))
    network.add(Reshape(projection_shape))
    network.add(BatchNormalization())
    network.add(Conv2DTranspose(gf_dim * 4, (kernel_height, kernel_width), padding='same', output_padding=1, strides=(deconvolution_height, deconvolution_width), activation='relu'))
    network.add(BatchNormalization())
    network.add(Conv2DTranspose(gf_dim * 2, (kernel_height, kernel_width), padding='same', output_padding=0, strides=(deconvolution_height, deconvolution_width), activation='relu'))
    network.add(BatchNormalization())
    network.add(Conv2DTranspose(gf_dim * 1, (kernel_height, kernel_width), padding='same', output_padding=1, strides=(deconvolution_height, deconvolution_width), activation='relu'))
    network.add(BatchNormalization())
    network.add(Conv2DTranspose(image_channels, (kernel_height, kernel_width), padding='same', output_padding=1, strides=(deconvolution_height, deconvolution_width), activation='tanh'))
    # network.summary()
    return network


if __name__ == '__main__':
    # Variables
    (training_images, _), (_, _) = mnist.load_data()
    training_images = (training_images - 127.5) / 127.5
    image_dims = (28, 28)
    image_channels = 1
    image_shape = image_dims + (image_channels,)
    latent_dims = 100
    # DCGAN Hyper-parameters
    df_dim = 64  # Dimension of discriminator filters in first convolutional layer
    gf_dim = 64  # Dimension of generator filters in last convolutional layer
    kernel_height = 5
    kernel_width = 5
    strides_height = 2
    strides_width = 2
    deconvolution_height = 2
    deconvolution_width = 2
    deconvolution_shape = (deconvolution_height, deconvolution_width)
    # Model Definition and Training
    framework = RegularizedGAN(image_shape, latent_dims, discriminator(), generator())
    training_images = np.expand_dims(training_images, -1)  # Add in the channel dimension
    framework.train(training_images, 10, initial_gamma=0.01)
