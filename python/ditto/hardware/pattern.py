import tvm._ffi
import tvm
from . import _ffi_api
from tvm.runtime import Object


@tvm._ffi.register_object("ditto.hardware.Pattern")
class Pattern(Object):
    """Hardware memory pattern object"""

    def __str__(self) -> str:
        ret = f"Pattern({self.name})"
        return ret

    def __repr__(self) -> str:
        return str(self)


@tvm._ffi.register_object("ditto.hardware.ScalarPattern")
class ScalarPattern(Pattern):
    """Hardware memory scalar pattern object"""

    def __str__(self) -> str:
        ret = f"ScalarPattern({self.name})"
        return ret

    def __repr__(self) -> str:
        return str(self)


def scalar_pattern(dtype, qualifier, name="scalar_pattern"):
    """The scalar memory pattern

    Args:
        dtype (str): data type
        qualifier (str): scope of the memory
        name (str, optional): name of the pattern. Defaults to "scalar_pattern".

    Returns:
        hardware.ScalarPattern: the scalar pattern
    """
    return _ffi_api.ScalarPattern(name, dtype, qualifier)


@tvm._ffi.register_object("ditto.hardware.MatrixPattern")
class MatrixPattern(Pattern):
    """Hardware memory matrix pattern object"""

    def __str__(self) -> str:
        ret = f"MatrixPattern({self.name})"
        return ret

    def __repr__(self) -> str:
        return str(self)


def matrix_pattern(dtype, m, n, qualifier, name="scalar_pattern"):
    """The scalar memory pattern

    Args:
        dtype (str): data type
        m (int): height
        n (int): width
        qualifier (str): scope of the memory
        name (str, optional): name of the pattern. Defaults to "scalar_pattern".

    Returns:
        hardware.ScalarPattern: the scalar pattern
    """
    return _ffi_api.MatrixPattern(name, dtype, m, n, qualifier)
