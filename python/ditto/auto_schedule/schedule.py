import tvm
from ditto import auto_compute as ac
from .schedulers import (
    auto_schedule_dispatch,
    auto_schedule_model_dispatch,
    ScheduleOption
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


def auto_schedule_model(inputs, fget_model, schedule_option):
    """
    Automatic schedule function for the whole model

    Args:
    ---
    inputs: List[tvm.te.Tensor]
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
