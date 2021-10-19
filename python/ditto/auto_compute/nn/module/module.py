import tvm


class Parameter(object):
    def __init__(self, shape, name="param", dtype="float32"):
        self.tensor = tvm.te.placeholder(shape, name=name, dtype=dtype)
        
    @property
    def shape(self):
        return self.tensor.shape
    
    @property
    def name(self):
        return self.tensor.name
    
    @property
    def dtype(self):
        return self.tensor.dtype


class Module(object):
    def __init__(self):
        self.training = True

    def forward(self, *args, **kwargs):
        raise NotImplementedError()

    def parameters(self):
        for k, v in self.__dict__.items():
            if isinstance(v, Parameter):
                yield v.tensor
            elif isinstance(v, Module):
                for ww in v.parameters():
                    yield ww
            elif isinstance(v, list):
                for elem in v:
                    if isinstance(elem, Module):
                        for ww in elem.parameters():
                            yield ww

    def eval(self):
        self.training = False
        for k, v in self.__dict__.items():
            if isinstance(v, Module):
                v.eval()
            elif isinstance(v, list):
                for elem in v:
                    if isinstance(elem, Module):
                        elem.eval()

    def train(self):
        self.training = True
        for k, v in self.__dict__.items():
            if isinstance(v, Module):
                v.eval()
            elif isinstance(v, list):
                for elem in v:
                    if isinstance(elem, Module):
                        elem.eval()

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
