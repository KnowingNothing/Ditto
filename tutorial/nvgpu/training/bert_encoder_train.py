import tvm
import argparse
import numpy as np
from ditto import auto_compute as ac
from ditto import autograd as ag
from ditto import auto_schedule
from ditto import runtime
from ditto.auto_compute.nn.model import BertBaseEncoder


def train_fp32(test=False):
    A = ac.layer_tensor([1, 512, 768], dtype="float32", name="A")
    model = BertBaseEncoder()
    outputs = model(A)
    

    graph = ac.graph([A], [outputs])
    grad_graph = ag.grad_graph(graph, reserve_forward=True)
    print(grad_graph)

    tasks = auto_schedule.extract_tasks_from_graph(grad_graph)
    
    # new_tasks = {}
    # for i, (k, v) in enumerate(tasks.items()):
    #     if i == 7:
    #         new_tasks[k] = v
    #         print(v[0], flush=True)
    # tasks = new_tasks

    target = "cuda"
    target_host = "llvm"
    trials = 10000
    task_name = "BertBaseEncoder_train_fp32"
    log_file = "BertBaseEncoder_train_fp32.log"
    builder = "local"
    runner = "local"

    schedule_option = auto_schedule.ScheduleOption(
        target, target_host=target_host,
        trials=trials, task_name=task_name,
        log_file=log_file, builder=builder, runner=runner
    )
    
    if not test:
        schedules = auto_schedule.auto_schedule_tasks(tasks, schedule_option)
    else:
        schedules = auto_schedule.retrieve_schedule_tasks(tasks, schedule_option)
    
    built_mods = {}
    dev = tvm.device(target)
    for (key, (sch, args)) in schedules.items():
        mod = tvm.build(sch, args, target, target_host)
        built_mods[key] = mod

    ge = runtime.create_graph_engine(grad_graph, built_mods, tvm.nd.device(target))
    fcompile = ge.get_function("compile")
    frun = ge.get_function("run")
    ftimeit = ge.get_function("timeit")
    fset_inputs = ge.get_function("set_inputs")
    fset_weight = ge.get_function("set_weight")
    fget_outputs = ge.get_function("get_outputs")
    fcompile()
    print("Compile done", flush=True)
    frun()
    print("Run done", flush=True)
    outputs = fget_outputs()
    print("Results:", flush=True)
    print(outputs[0].asnumpy())
    cost = ftimeit(10).value
    print("Time cost", cost, "ms")
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", help="tune or test", choices=["tune", "test"])
    parser.add_argument("--dtype", help="precision", choices=["fp32"])
    
    args = parser.parse_args()
    if args.dtype == "fp32":
        train_fp32(test=(args.mode == "test"))
    else:
        raise ValueError(f"Unknown dtype {args.dtype}.\n")
    