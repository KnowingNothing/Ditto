import tvm
from ditto import auto_compute as ac
from tvm import auto_scheduler


def auto_schedule(compute, schedule_option):
    """
    Auto-schedule function using Ansor
    
    Args:
    ---
    compute: callable
        The compute
    schedule_option: ditto.auto_schedule.schedulers.dispatch.ScheduleOption
        Schedule options
    
    Returns:
    ---
    [tvm.te.schedule.Schedule, List[tvm.te.Tensor]]
    """
    auto_scheduler.register_workload(schedule_option.task_name, f=compute)

    target = tvm.target.Target(schedule_option.target)
    task = auto_scheduler.SearchTask(
        schedule_option.task_name, args=(), target=target
    )
    
    # print(task.compute_dag)
    
    log_file = schedule_option.log_file
    tune_option = auto_scheduler.TuningOptions(
        num_measure_trials=schedule_option.trials,  # change this to 1000 to achieve the best performance
        builder=schedule_option.builder,
        runner=schedule_option.runner,
        measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
        verbose=schedule_option.verbose,
    )

    ######################################################################
    # Run the search
    # ^^^^^^^^^^^^^^
    # Now we get all inputs ready. Pretty simple, isn't it?
    # We can kick off the search and let the auto-scheduler do its magic.
    # After some measurement trials, we can load the best schedule from the log
    # file and apply it.

    # Run auto-tuning (search)
    task.tune(tune_option)
    # Apply the best schedule
    sch, args = task.apply_best(log_file)
    return sch, args
