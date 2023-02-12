import tensorflow as tf


class DeepNetworkModel(tf.keras.Model):
    def __init__(self, input_layer, hidden_layers, n_outputs):
        super(DeepNetworkModel, self).__init__()
        self.input_layer = input_layer
        self.hidden_layers = hidden_layers
        self.output_layer = tf.keras.layers.Dense(
            n_outputs, activation='linear', kernel_initializer='RandomNormal')

    def call(self, inputs):
        z = self.input_layer(inputs)
        for layer in self.hidden_layers:
            z = layer(z)
        output = self.output_layer(z)
        return output
