import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.keras import layers
import numpy as np
import torch

import time
import os


def channel_shuffle(inputs, num_groups):

    n, h, w, c = inputs.shape
    x_reshaped = tf.reshape(inputs, [-1, h, w, num_groups, c // num_groups])
    x_transposed = tf.transpose(x_reshaped, [0, 1, 2, 4, 3])
    output = tf.reshape(x_transposed, [-1, h, w, c])

    return output


def group_conv(inputs, filters, kernel, strides, num_groups):

    conv_side_layers_tmp = tf.split(inputs, num_groups, axis=3)
    conv_side_layers = []
    for layer in conv_side_layers_tmp:
        conv_side_layers.append(
            tf.keras.layers.Conv2D(
                filters // num_groups, kernel, strides, padding="same"
            )(layer)
        )
    x = tf.concat(conv_side_layers, axis=-1)

    return x


class GroupConv(layers.Layer):
    def __init__(self, filters, kernel, strides, num_groups, **kwargs):
        super().__init__(**kwargs)
        self.conv = layers.Conv2D(
            filters // num_groups, kernel, strides, padding="same"
        )

        def fwd(inputs):
            conv_side_layers_tmp = tf.split(inputs, num_groups, axis=3)
            conv_side_layers = []
            for layer in conv_side_layers_tmp:
                conv_side_layers.append(self.conv(layer))
            x = tf.concat(conv_side_layers, axis=-1)
            return x

        self.fwd = fwd

    def call(self, inputs):
        return self.fwd(inputs)


class Conv(layers.Layer):
    def __init__(self, filters, kernel_size, stride=1, activation=False, **kwargs):
        super().__init__(**kwargs)
        self.conv = layers.Conv2D(filters, kernel_size, stride, padding="same")
        self.bn = layers.BatchNormalization()

        def fwd(x):
            x = self.conv(x)
            x = self.bn(x)
            if activation:
                x = tf.nn.relu(x)
            return x

        self.fwd = fwd

    def call(self, x):
        return self.fwd(x)


def conv(inputs, filters, kernel_size, stride=1, activation=False):

    x = tf.keras.layers.Conv2D(filters, kernel_size, stride, padding="same")(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    if activation:
        x = tf.keras.tf.nn.relu(x)

    return x


class DepthwiseConvBn(layers.Layer):
    def __init__(self, kernel_size, stride=1, **kwargs):
        super().__init__(**kwargs)
        self.conv = layers.DepthwiseConv2D(
            kernel_size=kernel_size, strides=stride, padding="same"
        )
        self.bn = layers.BatchNormalization()

        def fwd(x):
            x = self.conv(x)
            x = self.bn(x)
            return x

        self.fwd = fwd

    def call(self, x):
        return self.fwd(x)


def depthwise_conv_bn(inputs, kernel_size, stride=1):

    x = tf.keras.layers.DepthwiseConv2D(
        kernel_size=kernel_size, strides=stride, padding="same"
    )(inputs)
    x = tf.keras.layers.BatchNormalization()(x)

    return x


class ShuffleNetUnitA(layers.Layer):
    def __init__(self, num_groups):
        super(ShuffleNetUnitA, self).__init__()
        self.num_groups = num_groups

    @tf_utils.shape_type_conversion
    def build(self, input_shape):
        self.in_channels = input_shape[-1]
        self.out_channels = self.in_channels
        self.bottleneck_channels = self.out_channels // 4

        self.group_conv1 = GroupConv(
            self.bottleneck_channels, kernel=1, strides=1, num_groups=self.num_groups
        )
        self.group_conv2 = GroupConv(
            self.out_channels, kernel=1, strides=1, num_groups=self.num_groups
        )
        self.depth_conv = DepthwiseConvBn(kernel_size=3, stride=1)
        self.bns = [layers.BatchNormalization() for _ in range(3)]

    def call(self, inputs):
        x = self.group_conv1(inputs)
        x = self.bns[0](x)
        x = tf.nn.relu(x)
        x = channel_shuffle(x, self.num_groups)
        x = self.depth_conv(x)
        x = self.bns[1](x)
        x = self.group_conv2(x)
        x = self.bns[2](x)
        x = layers.add([inputs, x])
        x = tf.nn.relu(x)
        return x


class ShuffleNetUnitB(layers.Layer):
    def __init__(self, out_channels, num_groups):
        super(ShuffleNetUnitB, self).__init__()
        self.out_channels = out_channels
        self.num_groups = num_groups

    @tf_utils.shape_type_conversion
    def build(self, input_shape):
        self.in_channels = input_shape[-1]
        self.out_channels -= self.in_channels
        self.bottleneck_channels = self.out_channels // 4
        self.group_conv1 = GroupConv(
            self.bottleneck_channels, kernel=1, strides=1, num_groups=self.num_groups
        )
        self.group_conv2 = GroupConv(
            self.out_channels, kernel=1, strides=1, num_groups=self.num_groups
        )
        self.depth_conv = DepthwiseConvBn(kernel_size=3, stride=2)
        self.bns = [layers.BatchNormalization() for _ in range(2)]

    def call(self, inputs):
        x = self.group_conv1(inputs)
        x = self.bns[0](x)
        x = tf.nn.relu(x)
        x = channel_shuffle(x, self.num_groups)
        x = self.depth_conv(x)
        x = self.group_conv2(x)
        x = self.bns[1](x)
        y = layers.AveragePooling2D(pool_size=3, strides=2, padding="same")(inputs)
        x = tf.concat([y, x], axis=-1)
        x = tf.nn.relu(x)

        return x


class ShuffleNet(layers.Layer):
    def __init__(self, first_stage_channels=240, num_groups=3, **kwargs):
        super().__init__(**kwargs)
        self.conv = layers.Conv2D(filters=24, kernel_size=3, strides=2, padding="same")
        self.unit_as = [ShuffleNetUnitA(num_groups) for i in range(3)]
        self.unit_bs = [
            ShuffleNetUnitB(first_stage_channels * (2 ** i), num_groups)
            for i in range(3)
        ]
        self.dense = layers.Dense(1000)

        def fwd(inputs):
            x = self.conv(inputs)

            x = tf.keras.layers.AveragePooling2D(
                pool_size=3, strides=2, padding="same"
            )(x)

            x = self.stage(x, self.unit_as[0], self.unit_bs[0], 3)
            x = self.stage(x, self.unit_as[1], self.unit_bs[1], 7)
            x = self.stage(x, self.unit_as[2], self.unit_bs[2], 3)

            x = tf.keras.layers.GlobalAveragePooling2D()(x)
            x = self.dense(x)
            return x, None

        self.fwd = fwd

    def call(self, inputs):
        return self.fwd(inputs)

    def stage(self, x, ua, ub, n):
        x = ub(x)
        for _ in range(n):
            x = ua(x)
        return x


def stage(inputs, out_channels, num_groups, n):

    x = ShuffleNetUnitB(out_channels, num_groups)(inputs)

    for _ in range(n):
        x = ShuffleNetUnitA(num_groups)(x)
    return x


def shuffle_net(inputs, first_stage_channels=240, num_groups=3):
    x = tf.keras.layers.Conv2D(filters=24, kernel_size=3, strides=2, padding="same")(
        inputs
    )
    x = tf.keras.layers.AveragePooling2D(pool_size=3, strides=2, padding="same")(x)

    x = stage(x, first_stage_channels, num_groups, n=3)
    x = stage(x, first_stage_channels * 2, num_groups, n=7)
    x = stage(x, first_stage_channels * 4, num_groups, n=3)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(1000)(x)

    return x, None


def train_perf():
    model = ShuffleNet()

    def criterion(labels, logits):
        return tf.nn.softmax_cross_entropy_with_logits(labels, logits)

    # criterion = tf.keras.losses.CategoricalCrossentropy()

    optimizer = tf.optimizers.SGD(learning_rate=0.002)
    dtype = "float32"
    img = np.random.uniform(-1, 1, [batch, 3, 224, 224]).astype(dtype)
    img_tensor = tf.convert_to_tensor(img)
    label_tensor = tf.convert_to_tensor(
        torch.empty(batch, dtype=torch.long).random_(1000).numpy()
    )
    label_tensor = tf.one_hot(label_tensor, depth=1000)
    number = 10
    repeats = 10

    @tf.function(experimental_compile=USE_XLA)
    def model_loss(img_tensor):
        output, _ = model(img_tensor)
        # print(label_tensor, output)
        loss = criterion(label_tensor, output)

        return loss

    for i in range(number):
        time_record = []
        for j in range(repeats):
            with tf.GradientTape() as tape:
                loss = model_loss(img_tensor)

            start = time.time()
            gradients = tape.gradient(loss, model.trainable_variables)
            stop = time.time()
            total = (stop - start) * 1000.0

            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            time_record.append(total)
        print("Average training latency", np.mean(time_record))
        print("Median training latency", np.median(time_record))
    print("batch = ", batch)


def inference_perf():
    model = ShuffleNet()
    dtype = "float32"
    img = np.random.uniform(-1, 1, [batch, 3, 224, 224]).astype(dtype)
    img_tensor = tf.convert_to_tensor(img)
    number = 10
    repeats = 10

    @tf.function(experimental_compile=USE_XLA)
    def model_func(img_tensor):
        output = model(img_tensor)
        return output

    for i in range(number):
        time_record = []
        for j in range(repeats):

            start = time.time()
            output = model_func(img_tensor)
            stop = time.time()
            total = (stop - start) * 1000.0

            time_record.append(total)
        print("Average inference latency", np.mean(time_record))
        print("Median inference latency", np.median(time_record))
    print("batch = ", batch)


if __name__ == "__main__":
    device = 0
    for xla in [True, False]:
        for batch in [1, 16, 32, 64]:
            USE_XLA = xla
            with tf.device("GPU:" + str(device)):
                train_perf()
                inference_perf()
                print("use XLA:", xla)
                print()
