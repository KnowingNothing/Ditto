import tensorflow as tf
from tensorflow.keras.applications import resnet50 as R
from tensorflow.keras import layers
import numpy as np
import time


def test_train_perf(batch=1):
    model = R.ResNet50(classes=1000, weights=None)
    dtype = "float32"
    img = np.random.uniform(-1, 1, [batch, 3, 224, 224]).astype(dtype)
    img_tensor = tf.convert_to_tensor(img)
    # label_tensor = tf.convert_to_tensor(np.random.randint(1000))
    label_tensor = tf.one_hot(np.random.randint(1000), depth=1000)
    number = 10
    repeats = 10

    @tf.function(experimental_compile=USE_XLA)
    def model_loss(img_tensor):
        x = model(img_tensor)
        # x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
        # x = layers.Dense(1000, activation="softmax",
        #                  name='predictions')(x)
        loss = tf.nn.softmax_cross_entropy_with_logits(label_tensor, x)
        return loss

    model_loss(img_tensor)

    optimizer = tf.optimizers.SGD(learning_rate=0.002)

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


def test_infer_perf(batch=1):
    model = R.ResNet50(classes=1000, weights=None)
    dtype = "float32"
    img = np.random.uniform(-1, 1, [batch, 3, 224, 224]).astype(dtype)
    img_tensor = tf.convert_to_tensor(img)
    number = 10
    repeats = 10

    @tf.function(experimental_compile=USE_XLA)
    def model_func(img_tensor):
        x = model(img_tensor)
        # x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
        # x = layers.Dense(1000, activation="softmax",
        #                  name='predictions')(x)
        return x

    model_func(img_tensor)

    for i in range(number):
        time_record = []
        for j in range(repeats):

            start = time.time()
            output = model_func(img_tensor)
            stop = time.time()
            total = (stop - start) * 1000.

            time_record.append(total)
        print("Average inference latency", np.mean(time_record))
        print("Median inference latency", np.median(time_record))
    print("batch = ", batch)


if __name__ == "__main__":
    tf.keras.backend.set_image_data_format("channels_first")
    device = 0
    for xla in [True, False]:
        for batch in [1, 16, 32, 64]:
            USE_XLA = xla
            with tf.device('GPU:'+str(device)):
                test_train_perf(batch)
                test_infer_perf(batch)
                print("use XLA:", xla)
                print()
