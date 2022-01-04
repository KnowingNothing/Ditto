import tvm._ffi
import tvm
from . import _ffi_api
from tvm.runtime import Object


@tvm._ffi.register_object("ditto.hardware.ISA")
class ISA(Object):
    """Virtual ISA object"""

    def __str__(self) -> str:
        ret = f"ISA({self.name})"
        return ret

    def __repr__(self) -> str:
        return str(self)


@tvm._ffi.register_object("ditto.hardware.ScalarISA")
class ScalarISA(ISA):
    """Scalar ISA object"""

    def __str__(self) -> str:
        ret = f"ScalarISA({self.name})"
        return ret

    def __repr__(self) -> str:
        return str(self)


def scalar_isa(latency, func, name="scalar_isa"):
    """Get a scalar ISA

    Args:
        latency (float): the latency in cycle
        func (tvm.te.tensor.Operation): the functionality
        name (str, optional): name of the isa. Defaults to "scalar_isa".

    Returns:
        hardware.ScalarISA: the scalar isa
    """
    return _ffi_api.ScalarISA(name, latency, func)


@tvm._ffi.register_object("ditto.hardware.MatrixISA")
class MatrixISA(ISA):
    """Matrix ISA object"""

    def __str__(self) -> str:
        ret = f"MatrixISA({self.name})"
        return ret

    def __repr__(self) -> str:
        return str(self)


def matrix_isa(latency, func, name="matrix_isa"):
    """Get a matrix ISA

    Args:
        latency (float): the latency in cycle
        func (tvm.te.tensor.Operation): the functionality
        name (str, optional): name of the isa. Defaults to "matrix_isa".

    Returns:
        hardware.MatrixISA: the matrix isa
    """
    return _ffi_api.MatrixISA(name, latency, func)


def scalar_binary_add(latency, lhs_type, rhs_type, res_type, name="scalar_binary_add"):
    """Get a scalar ISA

    Args:
        latency (float): the latency in cycle
        lhs_type (str): lhs data type
        rhs_type (str): rhs data type
        res_type (str): result data type
        name (str, optional): name of the isa. Defaults to "scalar_binary_add".

    Returns:
        hardware.ScalarISA: the scalar isa
    """
    return _ffi_api.ScalarBinaryAdd(name, latency, lhs_type, rhs_type, res_type)


def scalar_binary_sub(latency, lhs_type, rhs_type, res_type, name="scalar_binary_sub"):
    """Get a scalar ISA

    Args:
        latency (float): the latency in cycle
        lhs_type (str): lhs data type
        rhs_type (str): rhs data type
        res_type (str): result data type
        name (str, optional): name of the isa. Defaults to "scalar_binary_sub".

    Returns:
        hardware.ScalarISA: the scalar isa
    """
    return _ffi_api.ScalarBinarySub(name, latency, lhs_type, rhs_type, res_type)


def scalar_binary_mul(latency, lhs_type, rhs_type, res_type, name="scalar_binary_mul"):
    """Get a scalar ISA

    Args:
        latency (float): the latency in cycle
        lhs_type (str): lhs data type
        rhs_type (str): rhs data type
        res_type (str): result data type
        name (str, optional): name of the isa. Defaults to "scalar_binary_mul".

    Returns:
        hardware.ScalarISA: the scalar isa
    """
    return _ffi_api.ScalarBinaryMul(name, latency, lhs_type, rhs_type, res_type)


def scalar_binary_div(latency, lhs_type, rhs_type, res_type, name="scalar_binary_div"):
    """Get a scalar ISA

    Args:
        latency (float): the latency in cycle
        lhs_type (str): lhs data type
        rhs_type (str): rhs data type
        res_type (str): result data type
        name (str, optional): name of the isa. Defaults to "scalar_binary_div".

    Returns:
        hardware.ScalarISA: the scalar isa
    """
    return _ffi_api.ScalarBinaryDiv(name, latency, lhs_type, rhs_type, res_type)


def scalar_binary_mod(latency, lhs_type, rhs_type, res_type, name="scalar_binary_mod"):
    """Get a scalar ISA

    Args:
        latency (float): the latency in cycle
        lhs_type (str): lhs data type
        rhs_type (str): rhs data type
        res_type (str): result data type
        name (str, optional): name of the isa. Defaults to "scalar_binary_mod".

    Returns:
        hardware.ScalarISA: the scalar isa
    """
    return _ffi_api.ScalarBinaryMod(name, latency, lhs_type, rhs_type, res_type)


def scalar_multiply_add(latency, lhs_type, rhs_type, res_type, name="scalar_multiply_add"):
    """Get a scalar ISA

    Args:
        latency (float): the latency in cycle
        lhs_type (str): lhs data type
        rhs_type (str): rhs data type
        res_type (str): result data type
        name (str, optional): name of the isa. Defaults to "scalar_multiply_add".

    Returns:
        hardware.ScalarISA: the scalar isa
    """
    return _ffi_api.ScalarMultiplyAdd(name, latency, lhs_type, rhs_type, res_type)


def direct():
    return _ffi_api.ISADirect()


def none():
    return _ffi_api.ISANone()
