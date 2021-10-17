from collections import namedtuple
import json


ActionRecord = namedtuple(
    "ActionRecord",
    ["op_id", "action_id", "pattern_json"]
)


TransformRecord = namedtuple(
    "TransformRecord",
    ["op_id", "axis_ids", "transform_id", "parameter_json"]
)


class TransformApplier(object):
    def apply(self, layer_state, record):
        op_id = record.op_id
        axis_ids = record.axis_ids
        transform_id = record.transform_id
        parameter_json = record.parameter_json
        if transform_id == "fold":
            self.apply_fold(layer_state, op_id, axis_ids, parameter_json)
        elif transform_id == "unfold":
            self.apply_unfold(layer_state, op_id, axis_ids, parameter_json)
        elif transform_id == "shuffle":
            self.apply_shuffle(layer_state, op_id, axis_ids, parameter_json)
        elif transform_id == "eliminate":
            self.apply_eliminate(layer_state, op_id, axis_ids, parameter_json)
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
            layer_state.explicit_fold(op, iv_lst[0], factor)
        else:
            layer_state.implicit_fold(op, iv_lst[0], factor)
        
    
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
            layer_state.explicit_unfold(op, *iv_lst)
        else:
            layer_state.implicit_unfold(op, *iv_lst)
    
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
            layer_state.explicit_shuffle(op, *iv_lst)
        else:
            layer_state.implicit_shuffle(op, *iv_lst)
    
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
            layer_state.explicit_eliminate(op, iv_lst[0])
        else:
            layer_state.implicit_eliminate(op, iv_lst[0])