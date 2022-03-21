"""Utils for hyper fusion"""
import logging
import tvm
from typing import Union


hf_logger = logging.getLogger("HyperFusion")
hf_logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter("[%(name)s] %(asctime)s: [%(levelname)s] %(message)s")
ch.setFormatter(formatter)
hf_logger.addHandler(ch)


def ceil(x: Union[int, tvm.ir.PrimExpr], y: Union[int, tvm.ir.PrimExpr]):
    return (x + y - 1) // y
