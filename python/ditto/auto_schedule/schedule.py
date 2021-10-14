import tvm
from ditto import auto_compute as ac
from .schedulers import auto_schedule_dispatch, ScheduleOption


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
