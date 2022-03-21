import tvm
import numpy as np
import json
import pathlib
from ditto import auto_compute as ac
from tvm import auto_scheduler
from tvm.target import Target
from .common import extract_tasks_from_model


class ApplyHistoryBest(auto_scheduler.DispatchContext):
    """
    Apply the history best config

    This is a simplified version that only
    uses workload_key as unique id for task

    Parameters
    ----------
    records : str or iterator of (auto_scheduler.measure.MeasureInput,\
                                  auto_scheduler.measure.MeasureResult)
        Collection of tuning records.
        If is str, then it should be the filename of a records log file.
        Each row of this file is an encoded record pair. Otherwise, it is an iterator.
    n_lines: Optional[int]
        if it is not None, only load the first `n_lines` lines of log.
    include_compatible: bool
        When set to True, compatible records will also be considered.
    """

    def __init__(self, records, n_lines=None, include_compatible=False):
        super(ApplyHistoryBest, self).__init__()
        self.include_compatible = include_compatible
        self.records = {}

        self.load(records, n_lines)

    def load(self, records, n_lines=None):
        """Load records to this dispatch context

        Parameters
        ----------
        records : str or iterator of (auto_scheduler.measure.MeasureInput,\
                                      auto_scheduler.measure.MeasureResult)
            Collection of tuning records.
            If is str, then it should be the filename of a records log file.
            Each row of this file is an encoded record pair. Otherwise, it is an iterator.
        n_lines: Optional[int]
            if it is not None, only load the first `n_lines` lines of log
        """
        if isinstance(records, pathlib.Path):
            records = str(records)

        if isinstance(records, str):
            records = auto_scheduler.load_records(records)

        if not records:
            return

        counter = 0
        for inp, res in records:
            if n_lines is not None and counter >= n_lines:
                break
            counter += 1
            if res.error_no != 0:
                continue

            costs = [x.value for x in res.costs if isinstance(x, tvm.tir.FloatImm)]
            cost = np.mean(costs)
            key = json.loads(inp.task.workload_key)[0]
            if key in self.records:
                if cost < self.records[key][1]:
                    self.records[key] = (inp.state, cost)
            else:
                self.records[key] = (inp.state, cost)

    def _query_inside(self, target, workload_key, func_name):
        if workload_key in self.records:
            return self.records[workload_key][0]
        else:
            raise ValueError(f"No record for {workload_key}.\n")


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
    task = auto_scheduler.SearchTask(schedule_option.task_name, args=(), target=target)

    log_file = schedule_option.log_file
    tune_option = auto_scheduler.TuningOptions(
        num_measure_trials=schedule_option.trials,
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


def retrieve_schedule(compute, schedule_option):
    auto_scheduler.register_workload(schedule_option.task_name, f=compute)

    target = tvm.target.Target(schedule_option.target)
    task = auto_scheduler.SearchTask(schedule_option.task_name, args=(), target=target)

    log_file = schedule_option.log_file

    # Apply the best schedule
    sch, args = task.apply_best(log_file)
    return sch, args


def auto_schedule_model(inputs, fget_model, schedule_option):
    """
    Auto-schedule function for the whole model using Ansor

    Args:
    ---
    inputs: List[ditto.auto_compute.LayerTensor]
        The inputs for model
    fget_model: callable
        The function to get model
    schedule_option: ditto.auto_schedule.schedulers.dispatch.ScheduleOption
        Schedule options

    Returns:
    ---
    None
    """
    tasks = extract_tasks_from_model(inputs, fget_model)
    auto_schedule_tasks(tasks, schedule_option)


def auto_schedule_build_graph(inputs, fget_model, schedule_option):
    """
    Build function for the whole model using Ansor

    Args:
    ---
    inputs: List[ditto.auto_compute.LayerTensor]
        The inputs for model
    fget_model: callable
        The function to get model
    schedule_option: ditto.auto_schedule.schedulers.dispatch.ScheduleOption
        Schedule options

    Returns:
    ---
    Dict{workload_key(str):tvm.te.Schedue}
    """
    tasks = extract_tasks_from_model(inputs, fget_model)
    return auto_schedule_build_tasks(tasks, schedule_option)


def auto_schedule_tasks(tasks, schedule_option):
    target = schedule_option.target
    log_file = schedule_option.log_file

    tune_tasks = []
    task_weights = []

    def fget_tasks(wkl_key):
        def _inner():
            assert wkl_key in tasks and len(tasks[wkl_key])
            layer = tasks[wkl_key][0]
            all_tensors = layer.schedule_tensors
            return all_tensors

        return _inner

    for i, (wkl_key, layers) in enumerate(tasks.items()):
        auto_scheduler.register_workload(wkl_key, f=fget_tasks(wkl_key))

        target = tvm.target.Target(target)
        tune_task = auto_scheduler.SearchTask(wkl_key, target=target)
        tune_tasks.append(tune_task)
        task_weights.append(len(layers))

    tuner = auto_scheduler.TaskScheduler(tune_tasks, task_weights)
    tune_option = auto_scheduler.TuningOptions(
        num_measure_trials=schedule_option.trials,
        builder=schedule_option.builder,
        runner=schedule_option.runner,
        measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
        verbose=schedule_option.verbose,
    )

    tuner.tune(tune_option)


def auto_schedule_build_tasks(tasks, schedule_option):
    target = schedule_option.target
    target_host = schedule_option.target_host
    log_file = schedule_option.log_file
    target, target_host = Target.check_and_update_host_consist(target, target_host)

    schedules = {}
    with ApplyHistoryBest(log_file):
        for wkl_key, layers in tasks.items():
            dispatcher = auto_scheduler.DispatchContext.current
            assert len(layers) > 0
            layer = layers[0]
            outputs = layer.ops
            output_tensors = [op.output(0) for op in outputs]
            (
                relay_io_tensors,
                has_layout_free,
                has_complex_op,
            ) = auto_scheduler.relay_integration.traverse_to_get_io_tensors(
                output_tensors
            )
            io_tensors = layer.schedule_tensors
            dag = auto_scheduler.ComputeDAG(io_tensors)
            state = dispatcher.query(target, wkl_key, has_complex_op, dag, None)

            if state is not None:
                schedule, tensors = dag.apply_steps_from_state(state)
                schedules[wkl_key] = (schedule, tensors)
            else:
                raise ValueError(f"No schedule for {wkl_key}")

    return schedules
