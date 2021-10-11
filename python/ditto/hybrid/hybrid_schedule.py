"""The computation hybrid api of TVM."""
import tvm._ffi
from tvm._ffi.base import string_types

from tvm.runtime import Object, convert
from tvm.ir import container as _container
from tvm.tir import IterVar, Buffer

from tvm.te import tensor as _tensor
from . import _ffi_api

def create_hybrid_schedule(ops):
    """Create a hybrid_schedule for list of ops

    Parameters
    ----------
    ops : list of Operations
        The source expression.

    Returns
    -------
    sch : hybrid.hybridSchedule
        The created hybrid_schedule.
    """
    if not isinstance(ops, (list, _container.Array)):
        ops = [ops]
    return _ffi_api.CreateHybridSchedule(ops)


@tvm._ffi.register_object
class HybridSchedule(Object):
    """HybridSchedule for all the stages."""

    def __getitem__(self, k):
        if isinstance(k, _tensor.Tensor):
            k = k.op
        if not isinstance(k, _tensor.Operation):
            raise ValueError("Expect schedule key to be Tensor or Operation")
        if k not in self.sch.stage_map:
            raise ValueError("Cannot find the operation %s in schedule" % (str(k)))
        return self.sch.stage_map[k]

    def normalize(self):
        """Build a normalized schedule from the current schedule.

        Insert necessary rebase to make certain iter var to start from 0.
        This is needed before bound inference and followup step.

        Returns
        -------
        sch : Schedule
            The normalized schedule.
        """
        return _ffi_api.ScheduleNormalize(self.sch)

    def create_group(self, outputs, inputs, include_inputs=False):
        """Create stage group by giving output and input boundary.

        The operators between outputs and inputs are placed as member of group.
        outputs are include in the group, while inputs are not included.

        Parameters
        ----------
        outputs : list of Tensors
            The outputs of the group.

        inputs : list of Tensors
            The inputs of the group.

        include_inputs : boolean, optional
            Whether include input operations in the group if they are used by outputs.

        Returns
        -------
        group : Stage
            A virtual stage represents the group, user can use compute_at to move
            the attachment point of the group.
        """
        if isinstance(outputs, _tensor.Tensor):
            outputs = [outputs]
        if isinstance(inputs, _tensor.Tensor):
            inputs = [inputs]
        return _ffi_api.ScheduleCreateGroup(self.sch, outputs, inputs, include_inputs)

    def cache_read(self, tensor, scope, readers):
        """Create a cache read of original tensor for readers.

        This will mutate the body of the readers.
        A new cache stage will be created for the tensor.
        Call this before doing any split/fuse schedule.

        Parameters
        ----------
        tensor : Tensor
            The tensor to be cached.
        scope : str
            The scope of cached
        readers : list of Tensor or Operation
            The readers to read the cache.

        Returns
        -------
        cache : Tensor
            The created cache tensor.
        """
        if isinstance(readers, (_tensor.Tensor, _tensor.Operation)):
            readers = [readers]
        readers = [t.op if isinstance(t, _tensor.Tensor) else t for t in readers]
        return _ffi_api.ScheduleCacheRead(self.sch, tensor, scope, readers)

    def cache_write(self, tensor, scope):
        """Create a cache write of original tensor, before storing into tensor.

        This will mutate the body of the tensor.
        A new cache stage will created before feed into the tensor.

        This function can be used to support data layout transformation.
        If there is a split/fuse/reorder on the data parallel axis of tensor
        before cache_write is called. The intermediate cache stores
        the data in the layout as the iteration order of leave axis.
        The data will be transformed back to the original layout in the original tensor.
        User can further call compute_inline to inline the original layout and keep
        the data stored in the transformed layout.

        Parameters
        ----------
        tensor : Tensor, list or tuple
            The tensors to be feed to. All the tensors must be produced by one computeOp
        scope : str
            The scope of cached

        Returns
        -------
        cache : Tensor
            The created cache tensor.
        """
        return _ffi_api.ScheduleCacheWrite(self.sch, tensor, scope)

    def rfactor(self, tensor, axis, factor_axis=0):
        """Factor a reduction axis in tensor's schedule to be an explicit axis.

        This will create a new stage that generated the new tensor with axis
        as the first dimension. The tensor's body will be rewritten as a reduction
        over the factored tensor.

        Parameters
        ----------
        tensor : Tensor
            The tensor to be factored.
        axis : IterVar
            The reduction axis in the schedule to be factored.
        factor_axis : int
            The position where the new axis is placed.

        Returns
        -------
        tfactor : Tensor or Array of Tensor
            The created factored tensor.
        """
        factored = _ffi_api.ScheduleRFactor(self.sch, tensor, axis, factor_axis)
        return factored[0] if len(factored) == 1 else factored
    
    def slice(self, tensor, axis, slice_point):
        """
        slice a stage into two stages by slicing a loop.
        """
        return _ffi_api.HybridScheduleSlice(self, tensor, axis, slice_point)

tvm._ffi._init_api("hybrid", __name__)
