from .ansor_integrate import auto_schedule as auto_schdule_ansor


class ScheduleOption(object):
    """
    Schedule options
    
    Args:
    ---
    target: str
        "cuda", "llvm", etc.
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
    def __init__(self, target, trials, task_name, log_file, builder, runner, scheduler=None, verbose=2):
        self.target = target
        self.trials = trials
        self.task_name = task_name
        self.log_file = log_file
        self.builder = builder
        self.runner = runner
        self.scheduler = scheduler
        self.verbose = verbose


def auto_schedule_dispatch(schedule_option):
    if schedule_option.scheduler is None:
        return auto_schdule_ansor
    elif schedule_option.scheduler == "ansor":
        return auto_schdule_ansor
    else:
        raise ValueError(f"Scheduler not known: {schedule_option.scheduler}.\n")
        