import torch.nn as nn


class BaseSafetyEvaluator:
    def __init__(self):
        pass
    
    def evaluate(self, module: nn.Module, model: nn.Module, input_shape: list, **kwargs):
        return 0
