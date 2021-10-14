from collections import namedtuple
import json


ActionRecord = namedtuple(
    "ActionRecord",
    ["op_id", "axis_ids", "transform_id", "parameter_json"]
)


class RecordApplier(object):
    def apply(self, layer_state, record):
        op_id = record.op_id
        axis_ids = record.axis_ids
        transform_id = record.transform_id
        parameter_json = record.parameter_json
        if transform_id == "fold":
            pass
        elif transform_id == "unfold":
            pass
        elif transform_id == "shuffle":
            pass
        elif transform_id == "eliminate":
            pass
        else:
            raise ValueError(f"Unknown transform: {transform_id}.\n")
        
    def apply_fold(self, layer_state, op_id, axis_ids, parameter_json):
        all_ops = layer_state.get_current_ops()
        op = all_ops[op_id]
        axis = layer_state[op].axis()
        reduce_axis = layer_state[op].reduce_axis()
        all_axis = [*axis, *reduce_axis]
        
        iv_lst = []
        for iv_id in axis_ids:
            iv = all_axis[iv_id]
            iv_lst.append(iv)
        
        obj = json.loads(parameter_json)
        factor = obj["factor"]
        explicit = obj["explicit"]
        if explicit:
            layer_state[op].explicit_fold(op, iv_lst[0], factor)
        else:
            layer_state[op].implicit_fold(op, iv_lst[0], factor)
        
    
    def apply_unfold(self, layer_state, op_id, axis_ids, parameter_json):
        all_ops = layer_state.get_current_ops()
        op = all_ops[op_id]
        axis = layer_state[op].axis()
        reduce_axis = layer_state[op].reduce_axis()
        all_axis = [*axis, *reduce_axis]
        
        iv_lst = []
        for iv_id in axis_ids:
            iv = all_axis[iv_id]
            iv_lst.append(iv)
        
        obj = json.loads(parameter_json)
        explicit = obj["explicit"]
        if explicit:
            layer_state[op].explicit_unfold(op, *iv_lst)
        else:
            layer_state[op].implicit_unfold(op, *iv_lst)
    
    def apply_shuffle(self, layer_state, op_id, axis_ids, parameter_json):
        all_ops = layer_state.get_current_ops()
        op = all_ops[op_id]
        axis = layer_state[op].axis()
        reduce_axis = layer_state[op].reduce_axis()
        all_axis = [*axis, *reduce_axis]
        
        # the order reflects in the order of axis_ids
        iv_lst = []
        for iv_id in axis_ids:
            iv = all_axis[iv_id]
            iv_lst.append(iv)
        
        obj = json.loads(parameter_json)
        explicit = obj["explicit"]
        if explicit:
            layer_state[op].explicit_shuffle(op, *iv_lst)
        else:
            layer_state[op].implicit_shuffle(op, *iv_lst)
    
    def apply_eliminate(self, layer_state, op_id, axis_ids, parameter_json):
        all_ops = layer_state.get_current_ops()
        op = all_ops[op_id]
        axis = layer_state[op].axis()
        reduce_axis = layer_state[op].reduce_axis()
        all_axis = [*axis, *reduce_axis]
        
        iv_lst = []
        for iv_id in axis_ids:
            iv = all_axis[iv_id]
            iv_lst.append(iv)
        
        obj = json.loads(parameter_json)
        explicit = obj["explicit"]
        if explicit:
            layer_state[op].explicit_eliminate(op, iv_lst[0])
        else:
            layer_state[op].implicit_eliminate(op, iv_lst[0])