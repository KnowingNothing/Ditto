from ditto import hardware as hw

from .ansor_integrate import auto_schedule as auto_schedule_ansor
from .ansor_integrate import auto_schedule_model as auto_schedule_model_ansor
from .ansor_integrate import (
    auto_schedule_build_graph as auto_schedule_build_graph_ansor,
)
from .ansor_integrate import retrieve_schedule as retrieve_schedule_ansor
from .ansor_integrate import auto_schedule_tasks as auto_schedule_tasks_ansor
from .ansor_integrate import (
    auto_schedule_build_tasks as auto_schedule_build_tasks_ansor,
)
from .chimera_integrate import auto_schedule_tasks as auto_schedule_tasks_chimera
from .chimera_integrate import auto_schedule_build_tasks as auto_schedule_build_tasks_chimera


class ScheduleOption(object):
    """
    Schedule options

    Args:
    ---
    target: str
        "cuda", "llvm", etc.
    target_host: str
        "llvm", etc.
    trials: int
        number of trials for auto-scheduler
    task_name: str
        name of the task
    log_file: str
        file for schedule logging
    builder: tvm.auto_scheduler.measure.ProgramBuilder
        build the function
    runner: tvm.auto_scheduler.measure.ProgramRunner
        run the function
    scheduler: None or str
        the scheduler to use [ansor, etc.]
    verbose: int
        verbose level 0, 1, 2
    """

    def __init__(
        self,
        target,
        target_host="llvm",
        trials=100,
        task_name="",
        log_file="tmp.log",
        builder="local",
        runner="local",
        scheduler=None,
        verbose=2,
        device_name=None,
        dtype="float32",
        mma_in_dtype="float16",
        mma_acc_dtype="float32",
        mma_MI=16,
        mma_NI=16,
        mma_KI=16,
    ):
        self.target = target
        self.target_host = target_host
        self.trials = trials
        self.task_name = task_name
        self.log_file = log_file
        self.builder = builder
        self.runner = runner
        self.scheduler = scheduler
        self.verbose = verbose
        if device_name is not None:
            self.device = hw.query_hw_param(device_name)
        else:
            self.device = None
        self.dtype = dtype
        self.mma_in_dtype = mma_in_dtype
        self.mma_acc_dtype = mma_acc_dtype
        self.mma_MI = mma_MI
        self.mma_NI = mma_NI
        self.mma_KI = mma_KI


def auto_schedule_dispatch(schedule_option):
    if schedule_option.scheduler is None:
        return auto_schedule_ansor
    elif schedule_option.scheduler == "ansor":
        return auto_schedule_ansor
    else:
        raise ValueError(f"Scheduler not known: {schedule_option.scheduler}.\n")


def auto_schedule_model_dispatch(schedule_option):
    if schedule_option.scheduler is None:
        return (auto_schedule_model_ansor, auto_schedule_build_graph_ansor)
    elif schedule_option.scheduler == "ansor":
        return (auto_schedule_model_ansor, auto_schedule_build_graph_ansor)
    else:
        raise ValueError(f"Scheduler not known: {schedule_option.scheduler}.\n")


def auto_schedule_tasks_dispatch(schedule_option):
    if schedule_option.scheduler is None:
        return (auto_schedule_tasks_ansor, auto_schedule_build_tasks_ansor)
    elif schedule_option.scheduler == "ansor":
        return (auto_schedule_tasks_ansor, auto_schedule_build_tasks_ansor)
    else:
        raise ValueError(f"Scheduler not known: {schedule_option.scheduler}.\n")

def get_tasks_scheduler_builder(scheduler_name):
    """
    scheduler_name: str
    """
    if scheduler_name is None:
        return (auto_schedule_tasks_ansor, auto_schedule_build_tasks_ansor)
    elif scheduler_name == "ansor":
        return (auto_schedule_tasks_ansor, auto_schedule_build_tasks_ansor)
    elif scheduler_name == "chimera":
        return (auto_schedule_tasks_chimera, auto_schedule_build_tasks_chimera)
    else:
        raise ValueError(f"Scheduler not known: {scheduler_name}.\n")

def retrieve_schedule(compute, schedule_option):
    if schedule_option.scheduler is None:
        return retrieve_schedule_ansor(compute, schedule_option)
    elif schedule_option.scheduler == "ansor":
        return retrieve_schedule_ansor(compute, schedule_option)
    else:
        raise ValueError(f"Scheduler not known: {schedule_option.scheduler}.\n")


def retrieve_schedule_model(inputs, fget_model, schedule_option):
    if schedule_option.scheduler is None:
        return auto_schedule_build_graph_ansor(inputs, fget_model, schedule_option)
    elif schedule_option.scheduler == "ansor":
        return auto_schedule_build_graph_ansor(inputs, fget_model, schedule_option)
    else:
        raise ValueError(f"Scheduler not known: {schedule_option.scheduler}.\n")


def retrieve_schedule_tasks(tasks, schedule_option):
    if schedule_option.scheduler is None:
        return auto_schedule_build_tasks_ansor(tasks, schedule_option)
    elif schedule_option.scheduler == "ansor":
        return auto_schedule_build_tasks_ansor(tasks, schedule_option)
    else:
        raise ValueError(f"Scheduler not known: {schedule_option.scheduler}.\n")

def retrieve_schedule_bound_tasks(bound_tasks, schedule_option):
    schedules = {}
    for scheduler_name, tasks in bound_tasks.items():
        if scheduler_name is None:
            tmp = auto_schedule_build_tasks_ansor(tasks, schedule_option)
            schedules.update(tmp)
        elif scheduler_name == "ansor":
            tmp = auto_schedule_build_tasks_ansor(tasks, schedule_option)
            schedules.update(tmp)
        elif scheduler_name == "chimera":
            tmp = auto_schedule_build_tasks_chimera(tasks, schedule_option)
            schedules.update(tmp)
        else:
            raise ValueError(f"Scheduler not known: {schedule_option.scheduler}.\n")
    return schedules
