import tvm

from ditto.auto_tensorize.hyper_fusion import TensorizeHyperFusionState
from ditto.hardware.hw_compute import HardwareCompute
from . import _ffi_api
from tvm.runtime import Object
import ditto.auto_compute as ac
from typing import List
from ditto.hardware import hw_param


@tvm._ffi.register_object("ditto.auto_tensorize.SerialFusionState")
class SerialFusionState(Object):
    pass


def build_serial_fusion_state(
    layer: ac.Layer, tensorizeAxes=[], tensorWeight=[1.0, 1.0]
):
    return _ffi_api.build_serial_fusion_state(layer, tensorizeAxes, tensorWeight)


def single_op_schedule(
    op: tvm.te.operation,
    tensorizeAxes: List[tvm.tir.IterVar],
    hw_param: hw_param,
    searchType: str = "normal",
    mode: str = "best",
):
    return _ffi_api.SingleOpSchedule(op, tensorizeAxes, hw_param, searchType, mode)


def build_fusion_context(
    sfs: SerialFusionState,
    layer,
    state: TensorizeHyperFusionState,
    code: str,
    path: str,
    hw_param: hw_param,
    searchType: str = "normal",
    mode: str = "best",
    dtype="float32",
):
    """
    searchType = [normal|stochastic]
    mode = [survey|best]
    """
    return _ffi_api.buildFusionContext(
        sfs, layer, state, code, path, hw_param, searchType, mode, dtype
    )


@tvm._ffi.register_object("ditto.auto_tensorize.FusionContext")
class fusion_context(Object):
    def run(self, i, sch: tvm.te.Schedule, verbose=False):
        return _ffi_api.runFusion(self, i, sch, verbose)

    def getComputation(self, i):
        return _ffi_api.getComputation(self, i)

    def getOccupancy(self, i):
        return _ffi_api.getOccupancy(self, i)

    def getPredCost(self, i):
        return _ffi_api.getPredCost(self, i)

    def getPredCostList(self, i):
        return _ffi_api.getPredCostList(self, i)


@tvm._ffi.register_object("ditto.auto_tensorize.ScheduleContext")
class schedule_context(Object):
    def run(
        self,
        i,
        sch: tvm.te.schedule,
        op: tvm.te.operation,
        tensorizeAxes: List[tvm.tir.IterVar],
        intrin,
        code,
        path="",
    ):
        return _ffi_api.run(self, i, sch, op, tensorizeAxes, intrin, code, path)
