import tvm


def softmax(x, dim=-1):
    assert dim >= -(len(x.shape)) and dim < len(x.shape)
    dim = dim + len(x.shape) if dim < 0 else dim
    r = tvm.te.reduce_axis([0, x.shape[dim]], "r")
    sum_shape = []
    for i, xx in enumerate(x.shape):
        if i != dim:
            sum_shape.append(xx)

    def _inner_sum(*args):
        real_args = []
        j = 0
        for i in range(len(x.shape)):
            if i == dim:
                real_args.append(r)
            else:
                real_args.append(args[j])
                j += 1
        return tvm.te.sum(tvm.te.exp(x(*real_args)), axis=[r])

    sum_val = tvm.te.compute(sum_shape, _inner_sum, "sum_val")

    def _inner(*args):
        real_args = []
        for i, xx in enumerate(args):
            if i != dim:
                real_args.append(xx)
        return tvm.te.exp(x(*args)) / sum_val(*real_args)

    outputs = tvm.te.compute(x.shape, _inner, "softmax")

    return outputs
