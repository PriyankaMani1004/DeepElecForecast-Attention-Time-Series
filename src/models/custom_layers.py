import tensorflow as tf
from keras.layers import Layer

class Attention(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(
            name="att_weight",
            shape=(input_shape[-1], 1),
            initializer="glorot_uniform",
            trainable=True
        )
        super().build(input_shape)

    def call(self, inputs):
        # inputs: (batch, timesteps, features)
        scores = tf.matmul(inputs, self.W)          # (batch, timesteps, 1)
        weights = tf.nn.softmax(scores, axis=1)     # attention weights
        context = tf.reduce_sum(inputs * weights, axis=1)
        return context
