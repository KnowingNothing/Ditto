import tvm
from ditto import auto_compute as ac
from ditto import auto_tensorize as at

from pebble import concurrent
from concurrent.futures import TimeoutError
from pebble import ProcessPool, ProcessExpired

import numpy as np
import heapq
import json
from functools import reduce


EVALUTE_SCHEDULE_INPUTS = None


def evaluate_schedule_worker_cuda(dummy):
    global EVALUTE_SCHEDULE_INPUTS
    sch, args, ins, outs = EVALUTE_SCHEDULE_INPUTS
    func = tvm.build(sch, args, "cuda")
    inputs_np = [
        np.random.uniform(-1, 1, [int(x) for x in y.shape]).astype(y.dtype) for y in ins
    ]

    outputs_np = [
        np.random.uniform(-1, 1, [int(x) for x in y.shape]).astype(y.dtype)
        for y in outs
    ]
    ctx = tvm.cuda()
    inputs_tvm = [tvm.nd.array(x, ctx) for x in inputs_np]
    outputs_tvm = [tvm.nd.array(x, ctx) for x in outputs_np]

    evaluator = func.time_evaluator(func.entry_name, ctx, min_repeat_ms=600)
    cost = evaluator(*inputs_tvm, *outputs_tvm).mean * 1e3
    # print(f"Our code uses {cost} ms")
    return cost


def evaluate_schedule_cuda(sch, args, ins, outs):
    global EVALUTE_SCHEDULE_INPUTS
    EVALUTE_SCHEDULE_INPUTS = (sch, args, ins, outs)
    with ProcessPool(1) as pool:
        future = pool.map(evaluate_schedule_worker_cuda, [0], timeout=100)
        iterator = future.result()

        while True:
            try:
                results = next(iterator)
                print(".Y", end="", flush=True)
            except StopIteration:
                break
            except TimeoutError as error:
                print(".T", end="", flush=True)
                results = 1e10
            except Exception as error:
                # print(error)
                print(".E", end="", flush=True)
                results = 1e10

        return results


def check_bmm_chain_applicable(layer):
    return ac.nn.module.is_self_attention_layer(layer)


def make_fusion_choice(fop, sop, fmatch, smatch):
    axis = [x for x in sop.axis]
    unmatch_axis = list(filter(lambda x: x not in smatch.axis, axis))
    raxis = [x for x in sop.reduce_axis]
    unmatch_raxis = list(filter(lambda x: x not in smatch.axis, raxis))
    fuse_choice = at.fusion_choice(
        fop, sop, [*unmatch_axis, *unmatch_raxis, *smatch.axis], 3
    )
    return fuse_choice


class ParamSubSpace(object):
    def __init__(self, choices):
        self.choices = choices
        assert len(self.choices) > 0

    def rand(self):
        return int(np.random.choice(self.choices))


class ParamSpace(object):
    def __init__(self):
        self.subspaces = {}

    def add_subspace(self, name, subspace):
        self.subspaces[name] = subspace

    def get(self, key):
        return self.subspaces[key]


def make_tensor_core_tensorize_param_space():
    ty_sizes = ParamSubSpace([1, 2, 4, 8])
    tz_size = ParamSubSpace([1, 2, 4, 8])
    input_vector_len = ParamSubSpace([1, 2, 4, 8])
    serial_y = ParamSubSpace([1, 2, 4])
    serial_z = ParamSubSpace([1, 2, 4])
    block_rx = ParamSubSpace([1, 2, 4, 8])
    warp_rx = ParamSubSpace([1, 2, 4, 8])
    block_ry = ParamSubSpace([1, 2, 4, 8])
    warp_ry = ParamSubSpace([1, 2, 4, 8, 32, 64])
    unroll_steps = ParamSubSpace([1, 32, 64, 128, 256, 1024])

    space = ParamSpace()
    space.add_subspace("ty_size", ty_sizes)
    space.add_subspace("tz_size", tz_size)
    space.add_subspace("input_vector_len", input_vector_len)
    space.add_subspace("serial_y", serial_y)
    space.add_subspace("serial_z", serial_z)
    space.add_subspace("block_rx", block_rx)
    space.add_subspace("warp_rx", warp_rx)
    space.add_subspace("block_ry", block_ry)
    space.add_subspace("warp_ry", warp_ry)
    space.add_subspace("unroll_steps", unroll_steps)

    return space


def auto_schedule_bmm_chain_cuda_with_params(
    layer,
    hw_param,
    tensorize_param,
    dtype="float32",
    mma_in_dtype="float16",
    mma_acc_dtype="float32",
    mma_MI=16,
    mma_NI=16,
    mma_KI=16,
):
    state = at.build_serial_fusion_state(layer)

    fop, sop = state.first_op, state.second_op

    first_packed = at.cuda_wmma(
        M=mma_MI,
        N=mma_NI,
        K=mma_KI,
        in_dtype=mma_in_dtype,
        out_dtype=mma_acc_dtype,
        scope="shared",
    )

    first_match_info_choices = at.intrinsic_match(
        fop.output(0), first_packed, ["InnerMost", "SameRange"]
    )

    choice = first_match_info_choices[0]

    first_match_info = at.match_info(choice, first_packed)

    second_packed = at.cuda_wmma(
        M=mma_MI,
        N=mma_NI,
        K=mma_KI,
        in_dtype=mma_in_dtype,
        out_dtype=mma_acc_dtype,
        scope="global",
    )

    second_match_info_choices = at.intrinsic_match(
        sop.output(0), second_packed, ["InnerMost", "SameRange"]
    )

    choice = second_match_info_choices[0]

    second_match_info = at.match_info(choice, second_packed)

    fuse_choice = make_fusion_choice(fop, sop, first_match_info, second_match_info)

    tensorize_state = at.tensorize_hyper_fusion_state(
        layer, fuse_choice, {fop: first_match_info, sop: second_match_info}
    )

    sch = at.tensorize_cuda(layer, tensorize_state, hw_param, tensorize_param)
    return sch


class Entry(object):
    def __init__(self, record, value):
        self.record = record
        self.value = value

    def __lt__(self, other):
        # to make a min-heap
        return self.value < other.value

    def to_json(self):
        return {"record": self.record, "value": self.value}


def evaluate_param_cuda(
    param,
    layer,
    hw_param,
    dtype="float32",
    mma_in_dtype="float16",
    mma_acc_dtype="float32",
    mma_MI=16,
    mma_NI=16,
    mma_KI=16,
    profile=True,
):
    if profile:
        sch = auto_schedule_bmm_chain_cuda_with_params(
            layer,
            hw_param,
            param,
            dtype=dtype,
            mma_in_dtype=mma_in_dtype,
            mma_acc_dtype=mma_acc_dtype,
            mma_MI=mma_MI,
            mma_NI=mma_NI,
            mma_KI=mma_KI,
        )

        cost = evaluate_schedule_cuda(
            sch,
            layer.schedule_tensors,
            [*layer.inputs, *layer.weights],
            [x.output(0) for x in layer.ops],
        )
    else:
        fop_input1_smem_elems = (
            param.tz_size * param.serial_z * mma_MI * param.warp_ry * mma_KI
        )
        fop_input2_smem_elems = (
            param.ty_size * param.serial_y * mma_NI * param.warp_ry * mma_KI
        )
        fop_output_smem_elems = (
            param.tz_size
            * param.serial_z
            * mma_MI
            * param.ty_size
            * param.serial_y
            * mma_NI
        )
        sop_input2_smem_elems = (
            param.ty_size * param.serial_y * mma_NI * param.warp_rx * mma_KI
        )
        fop_input1_frag_elems = param.serial_z * mma_MI * param.warp_ry * mma_KI
        fop_input2_frag_elems = param.serial_y * mma_NI * param.warp_ry * mma_KI
        fop_output_frag_elems = param.serial_z * mma_MI * param.serial_y * mma_NI
        sop_input1_frag_elems = param.serial_z * mma_MI * param.warp_rx * mma_KI
        sop_input2_frag_elems = param.serial_y * mma_NI * param.warp_rx * mma_KI
        sop_output_frag_elems = param.serial_z * mma_MI * param.serial_y * mma_NI

        assert dtype in ["float16", "float32"]
        assert mma_acc_dtype in ["float16", "float32"]
        in_bytes = 2 if dtype == "float16" else 4
        acc_byptes = 2 if mma_acc_dtype == "float16" else 4

        smem_kb = (
            (fop_input1_smem_elems + fop_input2_smem_elems + sop_input2_smem_elems)
            * in_bytes
            + fop_output_smem_elems * acc_byptes
        ) / 1e3
        reg_kb = (
            (
                fop_input1_frag_elems
                + fop_input2_frag_elems
                + sop_input1_frag_elems
                + sop_input2_frag_elems
            )
            * in_bytes
            + (fop_output_frag_elems + sop_output_frag_elems) * acc_byptes
        ) / 1e3
        if smem_kb > hw_param.shared_memory_per_group_kb:
            return 1e10
        if reg_kb > hw_param.register_per_processor_kb:
            return 1e10

        ops = [x for x in layer.ops]
        total_spatial = [int(iv.dom.extent) for iv in ops[-1].axis]
        total_spatial = reduce(lambda x, y: x * y, total_spatial, 1)
        waves = (
            total_spatial
            / (
                param.tz_size
                * param.ty_size
                * param.serial_y
                * param.serial_z
                * mma_MI
                * mma_NI
            )
            / hw_param.num_groups
        )
        cost = waves * (smem_kb / hw_param.shared_memory_bandwidth_gbs)
    return cost


def select_tensor_core_tensorize_param(
    space,
    wkl_key,
    log_file,
    layer,
    hw_param,
    dtype="float32",
    mma_in_dtype="float16",
    mma_acc_dtype="float32",
    mma_MI=16,
    mma_NI=16,
    mma_KI=16,
    init_pop=10,
    descent_steps=1000,
):
    keys = [
        "ty_size",
        "tz_size",
        "input_vector_len",
        "serial_y",
        "serial_z",
        "block_rx",
        "warp_rx",
        "block_ry",
        "warp_ry",
        "unroll_steps",
    ]
    # log file
    flog = open(log_file, "a")

    def log(entry):
        print(
            json.dumps({"wkl_key": wkl_key, "entry": entry.to_json()}),
            file=flog,
            flush=True,
        )

    # the default param
    default_tensorize_param = at.cuda_tensorize_param(
        warp_size=32,
        ty_size=2,
        tz_size=2,
        input_vector_len=4,
        serial_y=2,
        serial_z=1,
        block_rx=8,
        warp_rx=8,
        block_ry=4,
        warp_ry=4,
        unroll_steps=64,
    )
    default_kwargs = default_tensorize_param.to_json()
    # working heap
    param_heap = []
    retire_heap = []
    evaluate_heap = []
    print("Building tensorize paramters for Tenosr Core...", flush=True)
    # get init random values
    for i in range(init_pop):
        kwargs = {"warp_size": 32}
        for key in keys:
            subspace = space.get(key)
            value = subspace.rand()
            kwargs[key] = value
        param = at.cuda_tensorize_param(**kwargs)
        cost = evaluate_param_cuda(
            param,
            layer,
            hw_param,
            dtype=dtype,
            mma_in_dtype=mma_in_dtype,
            mma_acc_dtype=mma_acc_dtype,
            mma_MI=mma_MI,
            mma_NI=mma_NI,
            mma_KI=mma_KI,
            profile=False,
        )
        entry = Entry(kwargs, cost)
        heapq.heappush(param_heap, entry)

    # coordinate descent
    for i in range(descent_steps):
        entry = heapq.heappop(param_heap)
        kwargs = entry.record
        heapq.heappush(retire_heap, entry)
        for key in keys:
            subspace = space.get(key)
            for choice in subspace.choices:
                if kwargs[key] == choice:
                    continue
                tmp_kwargs = kwargs.copy()
                tmp_kwargs[key] = choice
                tmp_param = at.cuda_tensorize_param(**tmp_kwargs)
                cost = evaluate_param_cuda(
                    tmp_param,
                    layer,
                    hw_param,
                    dtype=dtype,
                    mma_in_dtype=mma_in_dtype,
                    mma_acc_dtype=mma_acc_dtype,
                    mma_MI=mma_MI,
                    mma_NI=mma_NI,
                    mma_KI=mma_KI,
                    profile=False,
                )
                heapq.heappush(param_heap, Entry(tmp_kwargs, cost))
    for p in param_heap:
        heapq.heappush(retire_heap, p)

    print("Evaluate Candidates...", flush=True)
    # add default into consideration
    cost = evaluate_param_cuda(
        default_tensorize_param,
        layer,
        hw_param,
        dtype=dtype,
        mma_in_dtype=mma_in_dtype,
        mma_acc_dtype=mma_acc_dtype,
        mma_MI=mma_MI,
        mma_NI=mma_NI,
        mma_KI=mma_KI,
        profile=True,
    )
    entry = Entry(default_kwargs, cost)
    heapq.heappush(evaluate_heap, entry)
    log(entry)

    for i in range(init_pop):
        entry = heapq.heappop(retire_heap)
        tmp_param = at.cuda_tensorize_param(**entry.record)
        cost = evaluate_param_cuda(
            tmp_param,
            layer,
            hw_param,
            dtype=dtype,
            mma_in_dtype=mma_in_dtype,
            mma_acc_dtype=mma_acc_dtype,
            mma_MI=mma_MI,
            mma_NI=mma_NI,
            mma_KI=mma_KI,
            profile=True,
        )
        entry = Entry(entry.record, cost)
        heapq.heappush(evaluate_heap, entry)
        log(entry)

    best_entry = heapq.heappop(evaluate_heap)
    print(
        f"\nDone! The best expected performance is {best_entry.value} ms.", flush=True
    )
    flog.close()
    return best_entry


def auto_schedule_bmm_chain_cuda(
    key,
    log_file,
    layer,
    hw_param,
    dtype="float32",
    mma_in_dtype="float16",
    mma_acc_dtype="float32",
    mma_MI=16,
    mma_NI=16,
    mma_KI=16,
):
    space = make_tensor_core_tensorize_param_space()
    tensorize_param = select_tensor_core_tensorize_param(
        space,
        key,
        log_file,
        layer,
        hw_param,
        dtype=dtype,
        mma_in_dtype=mma_in_dtype,
        mma_acc_dtype=mma_acc_dtype,
        mma_MI=mma_MI,
        mma_NI=mma_NI,
        mma_KI=mma_KI,
    )
    return tensorize_param
    # return auto_schedule_bmm_chain_cuda_with_params(
    #     layer,
    #     hw_param,
    #     tensorize_param,
    #     dtype=dtype,
    #     mma_in_dtype=mma_in_dtype,
    #     mma_acc_dtype=mma_acc_dtype,
    #     mma_MI=mma_MI,
    #     mma_NI=mma_NI,
    #     mma_KI=mma_KI,
    # )


def auto_schedule(key, layer, schedule_option):
    if schedule_option.target == "cuda":
        assert schedule_option.device is not None, "Should provide device_name"
        if check_bmm_chain_applicable(layer):
            # this is a layer can be transformed as bmm_chain
            return auto_schedule_bmm_chain_cuda(
                key,
                "chimera_" + schedule_option.log_file,
                layer,
                schedule_option.device,
                dtype=schedule_option.dtype,
                mma_in_dtype=schedule_option.mma_in_dtype,
                mma_acc_dtype=schedule_option.mma_acc_dtype,
                mma_MI=schedule_option.mma_MI,
                mma_NI=schedule_option.mma_NI,
                mma_KI=schedule_option.mma_KI,
            )
        else:
            raise RuntimeError("The layer should not be dispatched to Chimera.")
    else:
        raise RuntimeError(f"Target {schedule_option.target} is not supported yet.")


def auto_schedule_tasks(tasks, schedule_option):
    params = []
    for wkl_key, layers in tasks.items():
        entry = auto_schedule(wkl_key, layers[0], schedule_option)
        params.append({"wkl_key": wkl_key, "entry": entry.to_json()})

    # log_file = schedule_option.log_file
    # with open(log_file, "a") as fout:
    #     for p in params:
    #         fout.write(json.dumps(p) + "\n")


def auto_schedule_build_tasks(tasks, schedule_option):
    params = {}
    schedules = {}
    with open("chimera_" + schedule_option.log_file, "r") as fin:
        for line in fin:
            obj = json.loads(line)
            if obj["wkl_key"] not in params:
                params[obj["wkl_key"]] = obj["entry"]
            else:
                if obj["entry"]["value"] < params[obj["wkl_key"]]["value"]:
                    params[obj["wkl_key"]] = obj["entry"]

    for k, v in params.items():
        assert k in tasks, f"{k}"
        layer = tasks[k][0]
        tensorize_param = at.cuda_tensorize_param(**v["record"])
        sch = auto_schedule_bmm_chain_cuda_with_params(
            layer,
            schedule_option.device,
            tensorize_param,
            dtype=schedule_option.dtype,
            mma_in_dtype=schedule_option.mma_in_dtype,
            mma_acc_dtype=schedule_option.mma_acc_dtype,
            mma_MI=schedule_option.mma_MI,
            mma_NI=schedule_option.mma_NI,
            mma_KI=schedule_option.mma_KI,
        )
        schedules[k] = (sch, layer.schedule_tensors)

    return schedules
