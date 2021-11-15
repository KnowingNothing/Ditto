from tensorflow.python.keras import activations
from tensorflow.python.keras.utils import tf_utils
import tensorflow as tf

import numpy as np

class MILSTMCell(tf.keras.layers.Layer):
    def __init__(self,
                 units,
                 n_class,
                 **kwargs):
        super(MILSTMCell, self).__init__(**kwargs)
        self.units = units

        self.kernel = tf.keras.layers.Dense(self.units * 4)
        self.recurrent_kernel = tf.keras.layers.Dense(self.units * 4)

        self.alpha = self.add_weight(
            shape=(1, self.units * 4),
            name='alpha')
        self.beta1 = self.add_weight(
            shape=(1, self.units * 4),
            name='beta1')
        self.beta2 = self.add_weight(
            shape=(1, self.units * 4),
            name='beta2')

        self.classifier = tf.keras.layers.Dense(n_class)

    @tf_utils.shape_type_conversion
    def build(self, input_shape):
        input_dim = input_shape[-1]

    def call(self, inputs, states):
        h_tm1 = states[0]  # previous memory state
        c_tm1 = states[1]  # previous carry state

        x = self.kernel(inputs)
        h_tmp = self.recurrent_kernel(h_tm1)

        xh_alpha = self.alpha * x * h_tmp
        x_beta1 = self.beta1 * x
        h_beta2 = self.beta2 * h_tmp

        i, f, c, o = tf.split(xh_alpha + x_beta1 + h_beta2,
                              num_or_size_splits=4, axis=1)

        i = activations.sigmoid(i)
        f = activations.sigmoid(f)
        o = activations.sigmoid(o)
        c = activations.tanh(c) * i + c_tm1 * f

        h = activations.tanh(c) * o
        return h, c
    
    
class MILSTM(tf.keras.layers.Layer):
    def __init__(self,
                 steps,
                 n_class,
                 **kwargs):
        super(MILSTM, self).__init__(**kwargs)
        self.steps = steps

        self.recurrent_kernel = MILSTMCell(1024, n_class)
        self.classifier = tf.keras.layers.Dense(n_class)

    @tf_utils.shape_type_conversion
    def build(self, input_shape):
        input_dim = input_shape[-1]

    def call(self, inputs):
        batch, steps, seq = inputs.shape
        assert steps == self.steps
        h = tf.convert_to_tensor(np.random.randn(batch, 1024), inputs.dtype)
        c = tf.convert_to_tensor(np.random.randn(batch, 1024), inputs.dtype)
        for step in range(self.steps):
            x = tf.slice(inputs, [0, step, 0], [batch, 1, seq])
            x = tf.squeeze(x, [1])
            h, c = self.recurrent_kernel(x, [h, c])
        out = self.classifier(h)
        return out, [h, c]