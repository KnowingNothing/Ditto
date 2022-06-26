from heapq import nlargest
import tvm
import argparse
import numpy as np
from ditto import auto_compute as ac
from ditto import auto_tensorize as at
from ditto import autograd as ag
from ditto import auto_schedule
from ditto import runtime
from ditto import chimera
from ditto import hardware as hw
from ditto.auto_compute.nn.model import BERT


def inference(
    dtype="float16",
    test=False,
    use_chimera=True,
    batch=1,
    seq_len=512,
    hidden=768,
    n_layers=12,
    attn_heads=12,
    trials=1000,
):
    mma_in_dtype = "float16"
    mma_acc_dtype = "float32"
    mma_MI = mma_NI = mma_KI = 16
    hw_name = "gpu.cuda.A100"

    A = ac.layer_tensor([batch, seq_len, hidden], dtype=dtype, name="A")
    model = BERT(
        hidden=hidden,
        n_layers=n_layers,
        attn_heads=attn_heads,
        dtype=dtype,
        mma_in_dtype=mma_in_dtype,
        mma_acc_dtype=mma_acc_dtype,
        mma_MI=mma_MI,
        mma_NI=mma_NI,
        mma_KI=mma_KI,
    )
    outputs = model(A)

    graph = ac.graph([A], [outputs])
    # print("Before fusion:")
    # print(graph)
    if use_chimera:
        graph = chimera.graph_fusion(graph)
    # print()
    # print("After fusion:")
    # print(graph)

    target = "cuda"
    target_host = "llvm"
    trials = trials
    if use_chimera:
        task_name = f"bert_batch{batch}_seq{seq_len}_hidden{hidden}_heads{attn_heads}_layers{n_layers}_inference_fp16_mix_mma"
        log_file = f"bert_batch{batch}_seq{seq_len}_hidden{hidden}_heads{attn_heads}_layers{n_layers}_inference_fp16_mix_mma.log"
    else:
        task_name = (
            log_file
        ) = f"bert_batch{batch}_seq{seq_len}_hidden{hidden}_heads{attn_heads}_layers{n_layers}_inference_fp16_cuda"
        log_file = f"bert_batch{batch}_seq{seq_len}_hidden{hidden}_heads{attn_heads}_layers{n_layers}_inference_fp16_cuda.log"
    builder = "local"
    runner = "local"

    schedule_option = auto_schedule.ScheduleOption(
        target,
        target_host=target_host,
        trials=trials,
        task_name=task_name,
        log_file=log_file,
        builder=builder,
        runner=runner,
        dtype=dtype,
        mma_in_dtype=mma_in_dtype,
        mma_acc_dtype=mma_acc_dtype,
        mma_MI=mma_MI,
        mma_NI=mma_NI,
        mma_KI=mma_KI,
        device_name=hw_name,
    )

    tasks = auto_schedule.extract_tasks_from_graph(graph)
    if use_chimera:
        bound_tasks = chimera.task_binding(tasks)

        # tmp_bound_tasks = {}
        # for k, v in bound_tasks.items():
        #     if k == "chimera":
        #         tmp_bound_tasks[k] = v

        if not test:
            schedules = auto_schedule.auto_schedule_bound_tasks(
                bound_tasks, schedule_option
            )
        else:
            schedules = auto_schedule.retrieve_schedule_bound_tasks(
                bound_tasks, schedule_option
            )
    else:
        if not test:
            schedules = auto_schedule.auto_schedule_tasks(tasks, schedule_option)
        else:
            schedules = auto_schedule.retrieve_schedule_tasks(tasks, schedule_option)

    built_mods = {}
    dev = tvm.device(target)
    for (key, (sch, args)) in schedules.items():
        mod = tvm.build(sch, args, target, target_host)
        built_mods[key] = mod

    ge = runtime.create_graph_engine(graph, built_mods, tvm.nd.device(target))
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

"""
python bert_inference_chimera_cuda.py --mode tune --dtype float16 --chimera --trials 10000 --batch 1 --seq_len 512 --hidden 768 --n_heads 12 --n_layers 12 |& tee trace.log
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", help="tune or test", choices=["tune", "test"])
    parser.add_argument("--dtype", help="precision", choices=["float16", "float32"])
    parser.add_argument("--chimera", help="use chimera", action="store_true")
    parser.add_argument(
        "--trials", help="number of trails for ansor", type=int, default=10000
    )
    parser.add_argument("--batch", help="bert batch size", type=int, default=1)
    parser.add_argument("--seq_len", help="bert sequence length", type=int, default=512)
    parser.add_argument("--hidden", help="bert hidden size", type=int, default=768)
    parser.add_argument("--n_heads", help="bert attention heads", type=int, default=12)
    parser.add_argument(
        "--n_layers", help="bert number of layers", type=int, default=12
    )

    args = parser.parse_args()
    inference(
        args.dtype,
        test=(args.mode == "test"),
        use_chimera=args.chimera,
        batch=args.batch,
        seq_len=args.seq_len,
        hidden=args.hidden,
        attn_heads=args.n_heads,
        n_layers=args.n_layers,
        trials=args.trials
    )
