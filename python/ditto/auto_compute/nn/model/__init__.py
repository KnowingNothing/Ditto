from .bert import *
from .BertEncoder import *
from .capsule_lenet import *
from .lenet import *
from .LLTM import *
from .MI_LSTM import *
from .resnet import *
from .SCRNN import *
from .shufflenet import *
from .subLSTM import *
from .transformer import *


all_models = {
    "std-bert": BERT,
    "bert": BertBaseEncoder,
    "capsule": CapsNet,
    "lenet": LeNet5,
    "lltm": LLTM,
    "MI": MI_LSTM,
    "res18": resnet18,
    "res34": resnet34,
    "res50": resnet50,
    "res101": resnet101,
    "res152": resnet152,
    "sc": SCRNN,
    "shuffle": ShuffleNet,
    "sub": subLSTM,
    "transformer": TransformerBlock
}

model_inputs_shape = {
    "bert": [[1, 512, 768]],
    "capsule": [[1, 1, 28, 28]],
    "lenet": [[1, 1, 32, 32]],
    "lltm": [[1, 28 * 28], [1, 128], [1, 128]],
    "MI": [[1, 28 * 28], [1, 1024], [1, 1024]],
    "res18": [[1, 3, 224, 224]],
    "res34": [[1, 3, 224, 224]],
    "res50": [[1, 3, 224, 224]],
    "res101": [[1, 3, 224, 224]],
    "res152": [[1, 3, 224, 224]],
    "sc": [[1, 28 * 28], [1, 128], [1, 64]],
    "shuffle": [[1, 3, 224, 224]],
    "sub": [[1, 28 * 28], [1, 128], [1, 128]],
}


def get_dnn_model(name_key):
    return all_models[name_key]


def get_dnn_input_shape(name_key):
    return model_inputs_shape[name_key]
