from tensorflow.python.keras import activations
from tensorflow.python.keras.utils import tf_utils
import tensorflow as tf
import numpy as np

import numpy as np

import time
import argparse


class subLSTMCell(tf.keras.layers.Layer):
    """Cell class for the subLSTM layer.

    Arguments:
      units: Positive integer, dimensionality of the output space.
      activation: Activation function to use.
      recurrent_activation: Activation function to use
        for the recurrent step.

    Call arguments:
      inputs: A 2D tensor.
      states: List of state tensors corresponding to the previous timestep.
    """

    def __init__(self,
                 units,
                 output_size,
                 **kwargs):
        super(subLSTMCell, self).__init__(**kwargs)
        self.units = units
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units * 4),
            name='recurrent_kernel')

        self.bias = self.add_weight(
            shape=(1, self.units * 4),
            name='bias')
        self.classifer = tf.keras.layers.Dense(output_size)

    @tf_utils.shape_type_conversion
    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.kernel = self.add_weight(
            shape=(input_dim, self.units * 4),
            name='kernel')

    def call(self, inputs, states):
        h_tm1, c_tm1 = states

        x = inputs @ self.kernel + self.bias
        h = h_tm1 @ self.recurrent_kernel
        i, f, c, o = tf.split(activations.sigmoid(
            x + h), num_or_size_splits=4, axis=1)

        c = f * c_tm1 + c - i
        h = activations.sigmoid(c) - o
        outputs = self.classifer(h)
        return outputs, [h, c]


def test_train_perf(batch_size):
    model = subLSTMCell(128, 10)

    data = tf.convert_to_tensor(
        np.random.randn(batch_size, 28 * 28), np.float32)
    old_h = tf.convert_to_tensor(np.random.randn(batch_size, 128), np.float32)
    old_c = tf.convert_to_tensor(np.random.randn(batch_size, 128), np.float32)
    labels = tf.convert_to_tensor(np.random.randn(batch_size, 10), np.float32)

    @tf.function(experimental_compile=USE_XLA)
    def model_loss(data, states, labels):
        logits, _ = model(data, states)
        loss = tf.nn.softmax_cross_entropy_with_logits(
            labels=labels, logits=logits)
        return loss

    model_loss(data, [old_h, old_c], labels)

    number = 10
    repeats = 10

    optimizer = tf.optimizers.SGD(learning_rate=0.002)
    for i in range(number):
        records = []
        for j in range(repeats):
            with tf.GradientTape() as tape:
                loss = model_loss(data, [old_h, old_c], labels)

            start_time = time.time()
            gradients = tape.gradient(loss, model.trainable_variables)
            end_time = time.time()

            optimizer.apply_gradients(
                zip(gradients, model.trainable_variables))
            records.append(end_time - start_time)
        print("Average training latency {} ms".format(1000. * np.mean(records)))
        print("Median training latency {} ms".format(1000. * np.median(records)))
    print("batch = ", batch_size)


def test_infer_perf(batch_size):
    model = subLSTMCell(128, 10)
    data = tf.convert_to_tensor(
        np.random.randn(batch_size, 28 * 28), np.float32)
    old_h = tf.convert_to_tensor(np.random.randn(batch_size, 128), np.float32)
    old_c = tf.convert_to_tensor(np.random.randn(batch_size, 128), np.float32)

    @tf.function(experimental_compile=USE_XLA)
    def model_func(data, states):
        out = model(data, states)
        return out
    model_func(data, [old_h, old_c])

    number = 10
    repeats = 10
    for i in range(repeats):
        records = []
        for j in range(number):

            start_time = time.time()
            logits, _ = model_func(data, [old_h, old_c])
            end_time = time.time()

            records.append(end_time - start_time)
        print("Average inference latency {} ms".format(1000. * np.mean(records)))
        print("Median inference latency {} ms".format(
            1000. * np.median(records)))
    print("batch = ", batch_size)


if __name__ == "__main__":
    device = 0
    for xla in [True, False]:
        for batch in [1, 16, 32, 64]:
            USE_XLA = xla
            with tf.device('GPU:'+str(device)):
                test_train_perf(batch)
                test_infer_perf(batch)
                print("use XLA:", xla)
                print()
