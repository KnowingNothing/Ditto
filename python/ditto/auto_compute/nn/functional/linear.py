import tvm


def linear(inputs, weight, bias=None, out_dtype="float32"):
    """Linear function

    Args:
    -----------------------------
    inputs: tvm.te.tensor.Tensor
        shape [batch, ..., in_feature]
    weight: tvm.te.tensor.Tensor
        shape [out_feature, in_feature]
    bias  : tvm.te.tensor.Tensor
        shape [out_feature]
    -----------------------------

    Returns:
    -----------------------------
    tvm.te.tensor.Tensor
        shape [batch, ..., out_feature]
    -----------------------------
    """
    assert (inputs.shape[-1].value == weight.shape[1].value)
    k = tvm.te.reduce_axis((0, inputs.shape[-1]))

    def _inner(*args):
        return tvm.te.sum(inputs[(*args[:-1], k)].astype(out_dtype) * weight[args[-1], k].astype(out_dtype), axis=k)

    output = tvm.te.compute((*inputs.shape[:-1], weight.shape[0]), _inner)
    if bias is not None:
        assert (bias.shape[0].value == weight.shape[0].value)

        def _add(*args):
            return output[args] + bias[args[-1]]
        output = tvm.te.compute(output.shape, _add)
    return output
