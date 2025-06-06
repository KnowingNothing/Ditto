from collections import OrderedDict
import tvm
from ditto import auto_compute as ac
from .schedulers import (
    auto_schedule_dispatch,
    auto_schedule_model_dispatch,
    auto_schedule_tasks_dispatch,
    get_tasks_scheduler_builder,
    ScheduleOption,
)


def auto_schedule(compute, schedule_option):
    """
    Automatic schedule function

    Args:
    ---
    compute: callable
        The compute
    schedule_option: ditto.auto_schedule.schedule.ScheduleOption
        The options for schedule

    Returns:
    ---
    [tvm.te.schedule.Schedule, List[tvm.te.Tensor]]
    """
    scheduler = auto_schedule_dispatch(schedule_option)
    return scheduler(compute, schedule_option)


def auto_schedule_layer(fget_layer, schedule_option):
    def compute():
        layer = fget_layer()
        all_tensors = layer.schedule_tensors
        return all_tensors

    return auto_schedule(compute, schedule_option)


def auto_schedule_model(inputs, fget_model, schedule_option):
    """
    Automatic schedule function for the whole model

    Args:
    ---
    inputs: List[ditto.auto_compute.LayerTensor]
        The inputs for model
    fget_model: callable
        The function to get model
    schedule_option: ditto.auto_schedule.schedule.ScheduleOption
        The options for schedule

    Returns:
    ---
    Dict{workload_key(str):tvm.te.Schedue}
    """
    scheduler, builder = auto_schedule_model_dispatch(schedule_option)
    scheduler(inputs, fget_model, schedule_option)
    return builder(inputs, fget_model, schedule_option)


def auto_schedule_tasks(tasks, schedule_option):
    """
    Automatic schedule function for the whole model

    Args:
    ---
    tasks: OrderedDict[key, List[ditto.auto_compute.Layer]]
    schedule_option: ditto.auto_schedule.schedule.ScheduleOption
        The options for schedule

    Returns:
    ---
    Dict{workload_key(str):tvm.te.Schedue}
    """
    scheduler, builder = auto_schedule_tasks_dispatch(schedule_option)
    scheduler(tasks, schedule_option)
    return builder(tasks, schedule_option)


def auto_schedule_bound_tasks(bound_tasks, schedule_option):
    """
    Automatic schedule function for tasks that are already bound to schedule_options

    Args:
    ---
    tasks: OrderedDict[scheduler, OrderedDict[key, List[ditto.auto_compute.Layer]]]
    schedule_option: ditto.auto_schedule.schedule.ScheduleOption
        The options for schedule

    Returns:
    ---
    Dict{workload_key(str):tvm.te.Schedue}
    """
    ret = {}
    for scheduler_name, tasks in bound_tasks.items():
        scheduler, builder = get_tasks_scheduler_builder(scheduler_name)
        scheduler(tasks, schedule_option)
        tmp = builder(tasks, schedule_option)
        for k, v in tmp.items():
            assert k not in ret, "The same workload assigned to different schedulers!"
            ret[k] = v
    return ret
