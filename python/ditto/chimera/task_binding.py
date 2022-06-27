from ditto import auto_compute as ac
from ditto import auto_schedule
from collections import OrderedDict


def task_binding(tasks):
    """
    Assign schedulers for each task
    ---
    tasks: OrderedDict[scheduler(str), OrderedDict[key(str), List[Layer]]]]
    """
    ret = OrderedDict()
    ret["chimera"] = OrderedDict()
    ret["ansor"] = OrderedDict()
    for k, lst in tasks.items():
        if ac.nn.module.is_self_attention_layer(lst[0]):
            ret["chimera"][k] = lst
        else:
            ret["ansor"][k] = lst
    return ret
