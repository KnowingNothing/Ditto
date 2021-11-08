import os

import numpy as np

import torch
import torchvision

import tvm
from tvm import relay, autotvm
import tvm.relay.testing
from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner
import tvm.contrib.graph_executor as runtime


def tune_tasks(
    tasks,
    measure_option,
    tuner="xgb",
    n_trial=1000,
    early_stopping=None,
    log_filename="tuning.log",
    use_transfer_learning=True,
):
    # create tmp log file
    tmp_log_file = log_filename + ".tmp"
    if os.path.exists(tmp_log_file):
        os.remove(tmp_log_file)

    for i, tsk in enumerate(reversed(tasks)):
        prefix = "[Task %2d/%2d] " % (i + 1, len(tasks))

        # create tuner
        if tuner == "xgb" or tuner == "xgb-rank":
            tuner_obj = XGBTuner(tsk, loss_type="rank")
        elif tuner == "ga":
            tuner_obj = GATuner(tsk, pop_size=100)
        elif tuner == "random":
            tuner_obj = RandomTuner(tsk)
        elif tuner == "gridsearch":
            tuner_obj = GridSearchTuner(tsk)
        else:
            raise ValueError("Invalid tuner: " + tuner)

        if use_transfer_learning:
            if os.path.isfile(tmp_log_file):
                tuner_obj.load_history(autotvm.record.load_from_file(tmp_log_file))

        # do tuning
        tsk_trial = min(n_trial, len(tsk.config_space))
        tuner_obj.tune(
            n_trial=tsk_trial,
            early_stopping=early_stopping,
            measure_option=measure_option,
            callbacks=[
                autotvm.callback.progress_bar(tsk_trial, prefix=prefix),
                autotvm.callback.log_to_file(tmp_log_file),
            ],
        )

    # pick best records to a cache file
    autotvm.record.pick_best(tmp_log_file, log_filename)
    os.remove(tmp_log_file)


def tune_and_evaluate(tuning_opt, mod, params, input_shapes, target, dtype="float32"):
    # extract workloads from relay program
    print("Extract tasks...")
    tasks = autotvm.task.extract_from_program(
        mod["main"], target=target, params=params
    )
    print("tasks", tasks)

    # run tuning tasks
    print("Tuning...")
    tune_tasks(tasks, **tuning_opt)
    
    # compile kernels with history best records
    with autotvm.apply_history_best(tuning_opt["log_filename"]):
        print("Compile...")
        with tvm.transform.PassContext(opt_level=3):
            lib = relay.build_module.build(mod, target=target, params=params)

        # load parameters
        dev = tvm.device(str(target), 0)
        module = runtime.GraphModule(lib["default"](dev))

        if not isinstance(dtype, (tuple, list)):
            dtype = [dtype for _ in range(len(input_shapes))]

        for i, (shape, dt) in enumerate(zip(input_shapes, dtype)):
            data_tvm = tvm.nd.array((np.random.uniform(size=shape)).astype(dt))
            module.set_input(f"data_{i}", data_tvm)

        # evaluate
        print("Evaluate inference time cost...")
        print(module.benchmark(dev, number=1, repeat=600))


def bench_network(model, input_shapes, model_name="model", n_trial=2000):
    input_data = [torch.randn(s) for s in input_shapes]
    shape_list = [(f"data_{i}", shape) for i, shape in enumerate(input_shapes)]

    scripted_model = torch.jit.trace(model, input_data).eval()
    print("scripted_model", scripted_model)

    mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)
    print(mod["main"])

    target = tvm.target.cuda()

    tuning_opt = {
        "log_filename": "%s.log" % model_name,
        "tuner": "xgb",
        "n_trial": n_trial,  # 2000
        "early_stopping": 600,
        "measure_option": autotvm.measure_option(
            builder=autotvm.LocalBuilder(timeout=10),
            runner=autotvm.LocalRunner(number=20, repeat=3, timeout=4, min_repeat_ms=150),
        ),
    }
    tune_and_evaluate(tuning_opt, mod, params, input_shapes, target=target)


if __name__ == "__main__":
    model_name = "resnet18"
    model = getattr(torchvision.models, model_name)(pretrained=False)
    input_shapes = [[1, 3, 224, 224]]
    bench_network(model, input_shapes, model_name, n_trial=100)
