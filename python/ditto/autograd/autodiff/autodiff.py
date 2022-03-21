"""
Author: size zheng
"""

"""Automatic differentiation of tensor expressions."""
from .. import _ffi_api


def gradient(output, inputs, head=None):
    """Perform reverse-mode automatic differentiation.

    Parameters
    ----------
    output : Tensor
        The tensor to differentiate.

    inputs : List[Tensor]
        The list of input tensors to be differentiated wrt.

    head : Tensor
        The adjoint of the output, in other words, some tensor, by which the Jacobians
        will be multiplied. Its shape must be of the form `prefix + output.shape`.
        If `None` is passed, the identity tensor of shape `output.shape + output.shape`
        will be used.

    Returns
    -------
    tensors: List[Tensor]
        The result gradient, in the same order as the inputs
    """
    if not isinstance(inputs, list):
        inputs = [inputs]
    return _ffi_api.Gradient(output, inputs, head)


def expr_equal(a, b):
    """check expr equal"""
    return _ffi_api.expr_equal(a, b)


def grad_op(a, b, c):
    """grad op"""
    return _ffi_api.grad_op(a, b, c)
