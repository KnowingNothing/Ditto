import tvm
from tvm import auto_scheduler as ansor
import argparse
import numpy as np
from ditto import auto_compute as ac
from ditto import autograd as ag
from ditto import auto_schedule
from ditto import runtime
from ditto.auto_compute.nn.module import CapsuleConv2d


def main(batch=1, trials=10000, timeout=15,
               use_rpc=False, key=None, host=None, port=None,
               test=False):
    inputs = [ac.layer_tensor([batch, 64, 56, 56], dtype="float32", name="input")]
    task_name = "capsule_conv_fp32"
    log_file = "capsule_conv_fp32.log"
    model = CapsuleConv2d(
        64, 256, 3, padding=1, num_caps=8
    )
    outputs = model(*inputs)
    

    graph = ac.graph(inputs, outputs)
    print(graph)

    tasks = auto_schedule.extract_tasks_from_graph(graph)

    target = "cuda"
    target_host = "llvm"
    if use_rpc:
        builder = ansor.LocalBuilder()
        runner = ansor.RPCRunner(key, host=host, port=port, timeout=timeout)
    else:
        builder = ansor.LocalBuilder()
        runner = ansor.LocalRunner(timeout=timeout)

    schedule_option = auto_schedule.ScheduleOption(
        target, target_host=target_host,
        trials=trials, task_name=task_name,
        log_file=log_file, builder=builder, runner=runner
    )
    
    if not test:
        schedules = auto_schedule.auto_schedule_tasks(tasks, schedule_option)
    else:
        schedules = auto_schedule.retrieve_schedule_tasks(tasks, schedule_option)
    
    if use_rpc:
        from tvm.auto_scheduler.utils import request_remote
        from tvm.contrib import utils, tar
        remote = request_remote(key, host, port)
        dev = remote.device(target)
        built_mods = {}
        for i, (key, (sch, args)) in enumerate(schedules.items()):
            mod = tvm.build(sch, args, target, target_host)
            temp = utils.tempdir()
            file_name = f"kernel_{i}.tar"
            path_lib = temp.relpath(file_name)
            mod.export_library(path_lib, tar.tar)
            remote.upload(path_lib)
            mod = remote.load_module(file_name)
            built_mods[key] = mod
        ge = runtime.create_graph_engine(graph, built_mods, dev)
    else:
        dev = tvm.nd.device(target)
        built_mods = {}
        for (key, (sch, args)) in schedules.items():
            mod = tvm.build(sch, args, target, target_host)
            built_mods[key] = mod
        ge = runtime.create_graph_engine(graph, built_mods, dev)

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
    cost = ftimeit(100).value
    print("Time cost", cost, "ms")
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", help="tune or test", choices=["tune", "test"])
    parser.add_argument("--dtype", help="precision", choices=["fp32"])
    parser.add_argument("--batch", help="batch size for network", default=1, type=int)
    parser.add_argument("--use_rpc", help="use rpc for evaluation, give the rpc information: <key>:<host>:<port>", default="", type=str)
    parser.add_argument("--timeout", help="evaluation timeout (s)", default=15, type=int)
    parser.add_argument("--trials", help="trials for the whole model", default=10000, type=int)
    args = parser.parse_args()
    
    batch = args.batch
    use_rpc = args.use_rpc != ""
    if use_rpc:
        key, host, port = args.use_rpc.split(":")
        port = int(port)
    else:
        key = None
        host = None
        port = None
    timeout = args.timeout
    trials = args.trials
    if args.dtype == "fp32":
        main(
            batch=batch,
            trials=trials,
            timeout=timeout,
            use_rpc=use_rpc,
            key=key,
            host=host,
            port=port,
            test=(args.mode == "test"))
    else:
        raise ValueError(f"Unknown dtype {args.dtype}.\n")
    