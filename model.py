from keras.models import Model
from keras.layers import Activation, Input, Layer
from keras.optimizers import Adam
from keras.activations import sigmoid
from keras.backend import variable, gradients, shape, square, reshape, set_value
from tensorflow import norm
from matplotlib import pyplot as plt
import numpy as np


class Regularizer(Layer):

    def __init__(self, label, **kwargs):
        self.discrimination_label = label
        super().__init__(**kwargs)

    def call(self, inputs, **kwargs):
        data_input, data_logits = inputs
        batch_size = shape(data_input)[0]
        data_gradient = gradients(data_logits, data_input)[0]
        gradient_norm = norm(reshape(data_gradient, [batch_size, -1]), axis=1, keep_dims=True)
        data_label = sigmoid(data_logits)
        return [data_label, square(gradient_norm) * square(data_label - self.discrimination_label)]

    def compute_output_shape(self, input_shape):
        return [input_shape[1], input_shape[1]]


class RegularizedGAN:

    def __init__(self, data_shape, latent_dims, discriminator, generator, gaussian_prior=True):
        self.data_shape = data_shape
        self.latent_dims = latent_dims
        self.optimizer = Adam(lr=0.0002, beta_1=0.5)
        self.gamma = variable(0)
        self.gaussian_prior = gaussian_prior
        self.disc = discriminator
        self.discriminator = None
        self.generator = generator
        self.stacked_model = None
        self.build_graph()

    def build_graph(self):
        # Discriminator Model with Regularization
        real_sample = Input(shape=self.data_shape)
        fake_sample = Input(shape=self.data_shape)
        real_sample_logits = self.disc(real_sample)
        fake_sample_logits = self.disc(fake_sample)
        discriminator_outputs_for_real_sample = Regularizer(1)([real_sample, real_sample_logits])
        discriminator_outputs_for_fake_sample = Regularizer(0)([fake_sample, fake_sample_logits])
        self.discriminator = Model([real_sample, fake_sample], discriminator_outputs_for_real_sample + discriminator_outputs_for_fake_sample)
        self.discriminator.compile(optimizer=self.optimizer, loss=['binary_crossentropy', 'mae', 'binary_crossentropy', 'mae'], loss_weights=[1, self.gamma / 2, 1, self.gamma / 2])
        # Generator Model
        random_noise = Input(batch_shape=(None, self.latent_dims))
        generated_sample = self.generator(random_noise)
        self.disc.trainable = False
        sample_logits = self.disc(generated_sample)
        sample_label = Activation(activation='sigmoid')(sample_logits)
        self.stacked_model = Model(random_noise, sample_label)
        self.stacked_model.compile(optimizer=self.optimizer, loss='binary_crossentropy')

    def train(self, training_data, epochs, batch_size=100, discriminator_steps=1, annealing=True, initial_gamma=0.1, decay_factor=0.01):
        real = np.ones(shape=(batch_size, 1))
        fake = np.zeros(shape=(batch_size, 1))
        gradient = np.zeros(shape=(batch_size, 1))
        for epoch in range(epochs):
            print('\nEpoch {}/{}:'.format(epoch + 1, epochs))
            np.random.shuffle(training_data)
            data_batches = training_data.reshape(-1, discriminator_steps, batch_size, *self.data_shape)
            for step, step_batches in enumerate(data_batches):
                set_value(
                    self.gamma, initial_gamma * decay_factor ** ((epoch * len(data_batches) + step) / (epochs * len(data_batches))) if annealing else initial_gamma)
                discriminator_loss = 0
                for real_batch in step_batches:
                    random_noise = np.random.normal(size=(batch_size, self.latent_dims)) if self.gaussian_prior else np.random.uniform(-1, 1, (batch_size, self.latent_dims))
                    fake_batch = self.generator.predict(random_noise)
                    step_loss = self.discriminator.train_on_batch([real_batch, fake_batch], [real, gradient, fake, gradient])
                    discriminator_loss += step_loss[0] / discriminator_steps
                random_noise = np.random.normal(size=(batch_size, self.latent_dims)) if self.gaussian_prior else np.random.uniform(-1, 1, (batch_size, self.latent_dims))
                generator_loss = self.stacked_model.train_on_batch(random_noise, real)
                print('\rStep {}/{} Discriminator Total Loss: {:.3f} Generator Loss: {:.3f}'.format(step+1, len(data_batches), discriminator_loss, generator_loss), end='')
            self.display()

    def display(self, size=4):
        random_noise = np.random.normal(size=(size**2, self.latent_dims)) if self.gaussian_prior else np.random.uniform(-1, 1, (size**2, self.latent_dims))
        generated_samples = self.generator.predict(random_noise)
        figure, axes = plt.subplots(size, size, sharex=True, sharey=True)
        for i in range(size):
            for j in range(size):
                axes[i, j].imshow(generated_samples[i * size + j, :, :, 0])
        plt.show()
