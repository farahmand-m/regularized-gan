# Regularized GAN
This repository is an implementation of the NIPS17 paper [Stabilizing Training of Generative Adversarial Networks through Regularization](https://arxiv.org/abs/1705.09367) in Keras library with TensorFlow backend in Python.

# Usage
To apply this technique to your already-existing GAN models remove the **Sigmoid Activation** at the end of your discriminator network and attach the **Regularizer** layer to it which you can find below. This layer takes in the input tensor of the discriminator network along with the output of its last layer (a.k.a. the logits) and returns two tensors which are the activated value of the logits (labels) and the gradient penalties for the corresponding samples.

```
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
```

When compiling your model use the **Binary Cross-entropy** loss for the first output and the **Mean Absolute Error** loss for the second. The second loss output requires a dynamic weight which can be created thorough a **variable** tensor that you can find in the backend module of Keras library.

If you haven't implemented the GAN framework, you could simply pass your discriminator and generator architectures to the **RegularizedGAN** class that you can find in `models.py` and the class will build up the computational graph along with the training loop for you.

# Example
You can find the famous DCGAN architecture trained using this regularization technique within `main.py`.