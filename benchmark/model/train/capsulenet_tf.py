# coding: utf-8
import numpy as np
import tensorflow as tf
from tensorflow import nn as F
from tensorflow.keras import layers as nn
import time
import os


class ConvLayer(nn.Layer):

    def __init__(self, in_channels=1, out_channels=256):
        super(ConvLayer, self).__init__()
        self.conv = nn.Conv2D(filters=out_channels, kernel_size=9, padding="valid",
                              strides=1, data_format="channels_first")

    def call(self, x):
        conved = self.conv(x)
        features = F.relu(conved)
        return features


class PrimaryCaps(nn.Layer):

    def __init__(self, num_capsules=8, in_channels=256, out_channels=32):
        super(PrimaryCaps, self).__init__()
        self.capsules = [
            nn.Conv2D(filters=out_channels, kernel_size=9, strides=2,
                      padding="valid", data_format="channels_first")
            for _ in range(num_capsules)]

    def call(self, x):
        # get batch size of inputs
        batch_size = x.shape[0]
        u = [tf.reshape(capsule(x), [batch_size, 32 * 6 * 6, 1])
             for capsule in self.capsules]
        u = tf.concat(u, axis=-1)
        u_squash = self.squash(u)
        return u_squash

    def squash(self, input_tensor):

        squared_norm = tf.reduce_sum(
            (input_tensor ** 2), axis=-1, keepdims=True)
        scale = squared_norm / (1 + squared_norm)  # normalization coeff
        output_tensor = scale * input_tensor / tf.sqrt(squared_norm)
        return output_tensor


def softmax(input_tensor, dim=2):
    perm = list(range(len(input_tensor.shape)))
    perm[dim], perm[-1] = perm[-1], perm[dim]
    transposed_input = tf.transpose(input_tensor, perm=perm)
    softmaxed_output = F.softmax(
        tf.reshape(transposed_input, [-1, transposed_input.shape[-1]]), axis=-1)
    return tf.transpose(tf.reshape(softmaxed_output, transposed_input.shape), perm=perm)


# dynamic routing
def dynamic_routing(b_ij, u_hat, squash, routing_iterations=3):
    for iteration in range(routing_iterations):
        c_ij = softmax(b_ij, dim=2)
        s_j = tf.reduce_sum((c_ij * u_hat), axis=2, keepdims=True)
        v_j = squash(s_j)
        if iteration < routing_iterations - 1:
            a_ij = tf.reduce_sum((u_hat * v_j), axis=-1, keepdims=True)
            b_ij = b_ij + a_ij
    return v_j


class DigitCaps(nn.Layer):

    def __init__(self, num_capsules=10, previous_layer_nodes=32*6*6,
                 in_channels=8, out_channels=16):
        super(DigitCaps, self).__init__()
        self.num_capsules = num_capsules
        self.previous_layer_nodes = previous_layer_nodes
        self.in_channels = in_channels

        self.W = tf.Variable(np.random.randn(num_capsules, previous_layer_nodes,
                                             in_channels, out_channels).astype("float32"))

    def call(self, u):
        u = u[None, :, :, None, :]
        W = self.W[:, None, :, :, :]
        u_hat = tf.matmul(u, W)
        b_ij = tf.zeros_like(u_hat)
        # if TRAIN_ON_GPU:
        #     b_ij = b_ij.cuda()

        v_j = dynamic_routing(b_ij, u_hat, self.squash, routing_iterations=3)

        return v_j

    def squash(self, input_tensor):
        squared_norm = tf.reduce_sum(
            (input_tensor ** 2), axis=-1, keepdims=True)
        scale = squared_norm / (1 + squared_norm)
        output_tensor = scale * input_tensor / tf.sqrt(squared_norm)
        return output_tensor


class CapsuleNetwork(nn.Layer):

    def __init__(self):
        super(CapsuleNetwork, self).__init__()
        self.conv_layer = ConvLayer()
        self.primary_capsules = PrimaryCaps()
        self.digit_capsules = DigitCaps()

    def call(self, images):
        primary_caps_output = self.primary_capsules(self.conv_layer(images))
        caps_output = tf.squeeze(self.digit_capsules(primary_caps_output))
        caps_output = tf.einsum("ij... -> ji...", caps_output)
        # squeeze can will delete all 1 in dims, which is unexpected
        if batch == 1:
            caps_output = tf.reshape(caps_output, [batch, 10, 16])
        return caps_output


class CapsuleLoss(nn.Layer):

    def __init__(self):
        super(CapsuleLoss, self).__init__()

    def call(self, x, labels, images=None, reconstructions=None):
        batch_size = x.shape[0]
        v_c = tf.sqrt(tf.reduce_sum((x**2), axis=2))
        left = tf.reshape(F.relu(0.9 - v_c), [batch_size, -1])
        right = tf.reshape(F.relu(v_c - 0.1), [batch_size, -1])
        margin_loss = labels * left + 0.5 * (1. - labels) * right
        margin_loss = tf.reduce_sum(margin_loss)
        return margin_loss


def train_perf():
    in_channel = 1
    model = CapsuleNetwork()
    criterion = CapsuleLoss()
    optimizer = tf.optimizers.SGD(learning_rate=0.002)
    dtype = "float32"
    img = np.random.uniform(-1, 1, [batch, in_channel, 28, 28]).astype(dtype)
    img_tensor = tf.convert_to_tensor(img)
    label_tensor = tf.convert_to_tensor(
        np.random.rand(batch, 10).astype(dtype))
    number = 10
    repeats = 10

    @tf.function(experimental_compile=USE_XLA)
    def model_loss(img_tensor):
        caps_output = model(img_tensor)
        images, reconstructions = 0, 0
        loss = criterion(caps_output, label_tensor,
                         images, reconstructions)

        return loss

    for i in range(number):
        time_record = []
        for j in range(repeats):
            with tf.GradientTape() as tape:
                loss = model_loss(img_tensor)

            start = time.time()
            gradients = tape.gradient(loss, model.trainable_variables)
            stop = time.time()
            total = (stop - start) * 1000.

            optimizer.apply_gradients(
                zip(gradients, model.trainable_variables))

            time_record.append(total)
        print("Average training latency", np.mean(time_record))
        print("Median training latency", np.median(time_record))
    print("batch = ", batch)


def inference_perf():
    in_channel = 1
    model = CapsuleNetwork()
    dtype = "float32"
    img = np.random.uniform(-1, 1, [batch, in_channel, 28, 28]).astype(dtype)
    img_tensor = tf.convert_to_tensor(img)
    label_tensor = tf.convert_to_tensor(np.random.rand(batch, 10))
    number = 10
    repeats = 10

    @tf.function(experimental_compile=USE_XLA)
    def model_func(img_tensor):
        caps_output = model(img_tensor)
        return caps_output

    for i in range(number):
        time_record = []
        for j in range(repeats):

            start = time.time()
            caps_output = model_func(img_tensor)
            stop = time.time()
            total = (stop - start) * 1000.

            time_record.append(total)
        print("Average inference latency", np.mean(time_record))
        print("Median inference latency", np.median(time_record))
    print("batch = ", batch)


if __name__ == "__main__":
    device = 0
    for xla in [True, False]:
        for batch in [1, 16, 32, 64]:
            USE_XLA = xla
            with tf.device('GPU:'+str(device)):
                train_perf()
                inference_perf()
                print("use XLA:", xla)
                print()
