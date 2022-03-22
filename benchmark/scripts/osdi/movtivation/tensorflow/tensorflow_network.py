import tensorflow as tf
from tensorflow.keras.applications import mobilenet, nasnet, resnet50, xception
from tensorflow.keras.mixed_precision import experimental as mixed_precision
from tensorflow.keras import layers
from transformers.models.auto.modeling_tf_auto import TFAutoModel
from transformers.models.auto.configuration_auto import AutoConfig
import numpy as np
import time
import argparse

import milstm_tf
import shufflenet_tf


def get_mobilenet():
    tf.keras.backend.set_image_data_format("channels_first")
    return mobilenet.MobileNet()


def get_mobilenet_shape():
    return [3, 224, 224]


def get_nasnet():
    tf.keras.backend.set_image_data_format("channels_last")
    return nasnet.NASNetLarge()


def get_nasnet_shape():
    return [331, 331, 3]


def get_resnet50():
    tf.keras.backend.set_image_data_format("channels_first")
    return resnet50.ResNet50()


def get_resnet50_shape():
    return [3, 224, 224]


def get_xception():
    tf.keras.backend.set_image_data_format("channels_last")
    return xception.Xception()


def get_xception_shape():
    return [299, 299, 3]


def get_bert():
    config = AutoConfig.from_pretrained("bert-base-uncased")
    model = TFAutoModel.from_config(config)

    def _inner(inp):
        return model(inp, training=False)

    return _inner


def get_bert_shape():
    return [512]


def get_milstm():
    model = milstm_tf.MILSTM(8, 10)
    return model


def get_milstm_shape():
    return [8, 512]


def get_shufflenet():
    return shufflenet_tf.ShuffleNet()


def get_shufflenet_shape():
    return [3, 224, 224]


MODEL_DICT = {
    "mobile": get_mobilenet,
    "nas": get_nasnet,
    "res50": get_resnet50,
    "xception": get_xception,
    "bert": get_bert,
    "mi-lstm": get_milstm,
    "shuffle": get_shufflenet,
}

SHAPE_DICT = {
    "mobile": get_mobilenet_shape,
    "nas": get_nasnet_shape,
    "res50": get_resnet50_shape,
    "xception": get_xception_shape,
    "bert": get_bert_shape,
    "mi-lstm": get_milstm_shape,
    "shuffle": get_shufflenet_shape,
}


def test_infer_perf(
    fget_model,
    input_shape,
    batch=1,
    dtype="float32",
    mix_p=False,
    repeats=1,
    use_xla=True,
    input_is_index=False,
):
    if mix_p:
        policy = mixed_precision.Policy("mixed_float16")
        mixed_precision.set_policy(policy)
    model = fget_model()
    if input_is_index:
        dtype = "int32"
    else:
        dtype = dtype
    img = np.random.uniform(-1, 1, [batch, *input_shape]).astype(dtype)
    img_tensor = tf.convert_to_tensor(img)

    @tf.function(experimental_compile=use_xla)
    def model_func(img_tensor):
        x = model(img_tensor)
        return x

    model_func(img_tensor)

    time_record = []
    for i in range(repeats):
        start = time.time()
        output = model_func(img_tensor)
        stop = time.time()
        total = (stop - start) * 1000.0

        time_record.append(total)
    return time_record


if __name__ == "__main__":
    print("NOTE: please export XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/local/cuda-11.1")
    print("NOTE: please export CUDA_VISIBLE_DEVICES=0")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dnn",
        help="the dnn to test",
        default="res50",
        choices=["mobile", "nas", "res50", "xception", "bert", "mi-lstm", "shuffle"],
    )
    parser.add_argument("--batch", help="batch size", type=int, default=1)
    parser.add_argument(
        "--dtype", help="data type", default="fp32", choices=["fp32", "fp16", "mix"]
    )
    parser.add_argument(
        "--repeats", help="number of repeat exectution", type=int, default=1
    )
    parser.add_argument("--use_xla", help="enable xla jit", action="store_true")

    args = parser.parse_args()

    fget_model = MODEL_DICT[args.dnn]
    shape = SHAPE_DICT[args.dnn]()
    batch = args.batch
    if args.dtype == "fp32":
        dtype = "float32"
        mix_p = False
    elif args.dtype == "fp16":
        raise ValueError("No support for fp16 inference in TensorFlow.\n")
    elif args.dtype == "mix":
        dtype = "float16"
        mix_p = True
    repeats = args.repeats
    use_xla = args.use_xla
    input_is_index = args.dnn == "bert"
    device = 0

    print()
    print("Configurations:")
    print("--------------------------------")
    print("dnn =", args.dnn)
    print("input shape =", shape)
    print("batch =", batch)
    print("dtype =", dtype)
    print("mix precision =", mix_p)
    print("repeats =", repeats)
    print("use xla =", use_xla)
    print("input is index =", input_is_index)
    print()
    with tf.device("GPU:" + str(device)):
        time_record = test_infer_perf(
            fget_model,
            shape,
            batch=batch,
            dtype=dtype,
            mix_p=mix_p,
            repeats=repeats,
            use_xla=use_xla,
            input_is_index=input_is_index,
        )
        print("Average inference latency", np.mean(time_record))
