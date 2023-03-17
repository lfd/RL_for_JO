import tensorflow as tf
from tensorflow import keras

class SingleScale(keras.layers.Layer):

    def __init__(self, name=None):
        super(SingleScale, self).__init__(name=name)


    def build(self, input_shape):

        self.factor = self.add_weight(
            name='factor',
            shape=(1),
            initializer=keras.initializers.Constant(1.),
            trainable=True,
            dtype=keras.backend.floatx()
        )


    def call(self, inputs):
        return self.factor * inputs
