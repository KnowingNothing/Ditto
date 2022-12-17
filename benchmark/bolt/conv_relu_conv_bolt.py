import tvm
from tvm import relay
import numpy as np
import argparse
import time
from tvm.relay.op.contrib.cutlass import partition_for_cutlass
from tvm.contrib.cutlass import (
    tune_cutlass_kernels,
    build_cutlass_kernels,
    build_cutlass_kernels_vm,
)


def relay_conv_conv(
    batch,
    C,
    H,
    W,
    K1,
    K2,
    stride1,
    stride2,
    k1,
    k2,
    in_dtype="float16",
    acc_dtype="float16",
    target="cuda",
):
    Img = relay.var("Img", shape=[batch, C, H, W], dtype=in_dtype)
    W1 = relay.var("W1", shape=[K1, C, k1, k1], dtype=in_dtype)
    W2 = relay.var("W2", shape=[K2, K1, k2, k2], dtype=in_dtype)
    Conv1 = relay.nn.conv2d(
        Img,
        W1,
        strides=(stride1, stride1),
        padding=(k1 // 2, k1 // 2),
        out_dtype=acc_dtype,
    )
    cast = relay.cast(Conv1, dtype=in_dtype)
    cast = relay.nn.relu(cast)
    Conv2 = relay.nn.conv2d(
        cast,
        W2,
        strides=(stride2, stride2),
        padding=(k2 // 2, k2 // 2),
        out_dtype=acc_dtype,
    )
    cast = relay.cast(Conv2, dtype=in_dtype)
    args = relay.analysis.free_vars(cast)
    func = relay.Function(args, cast)

    return func


def has_cutlass():
    return tvm.get_global_func("relay.ext.cutlass", True) != None


def convert_conv2d_layout(mod, desired_layouts):
    with tvm.transform.PassContext(opt_level=3):
        seq = tvm.transform.Sequential([relay.transform.ConvertLayout(desired_layouts)])
        return seq(mod)


def profile_and_build(
    mod, params, sm, tmp_dir="./tmp", lib_path="compile.so", use_fast_math=False
):
    mod = partition_for_cutlass(mod)
    mod, num_cutlass_partition = tune_cutlass_kernels(
        mod, sm, profile_all=False, use_multiprocessing=False, tmp_dir=tmp_dir
    )
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(mod, target="cuda", params=params)
    lib = build_cutlass_kernels(lib, sm, tmp_dir, lib_path, use_fast_math=use_fast_math)
    dev = tvm.device("cuda", 0)
    rt_mod = tvm.contrib.graph_executor.GraphModule(lib["default"](dev))
    return rt_mod, dev, num_cutlass_partition


def get_ref_rt_mod(mod, params, target="cuda"):
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(mod, target=target, params=params)
    dev = tvm.device(target, 0)
    rt_mod = tvm.contrib.graph_executor.GraphModule(lib["default"](dev))
    return rt_mod, dev


def get_output(rt_mod, names, inputs):
    for name, inp in zip(names, inputs):
        rt_mod.set_input(name, inp)
    rt_mod.run()
    return rt_mod.get_output(0).asnumpy()


def verify_conv2d_chain(func, ref_target="cuda", sm=80, atol=1e-5, rtol=1e-5):
    if not has_cutlass():
        return
    mod = tvm.IRModule.from_expr(func)
    typ = relay.transform.InferType()(mod)["main"].body.checked_type
    use_vm = any(isinstance(s, tvm.tir.Any) for s in typ.shape)
    # x_np = np.random.uniform(-1, 1, (batch, M, K)).astype("float16")
    # y_np = np.random.uniform(-1, 1, (batch, K, L)).astype("float16")
    # z_np = np.random.uniform(-1, 1, (batch, L, N)).astype("float16")
    
    mod = convert_conv2d_layout(mod, {"nn.conv2d": ["NHWC", "OHWI"]})

    if use_vm:
        rt_mod, dev, num_partition = profile_and_build_vm(mod, {}, sm)
        rt_mod_ref, dev = get_ref_vm(mod, {}, target=ref_target)
        assert num_partition > 0
        x = tvm.nd.array(x_np, device=dev)
        y = tvm.nd.array(y_np, device=dev)
        out = get_output_vm(rt_mod, ["x", "y"], [x, y])
        ref_out = get_output_vm(rt_mod_ref, ["x", "y"], [x, y])
    else:
        rt_mod, dev, num_partition = profile_and_build(mod, {}, sm)
        # rt_mod_ref, dev = get_ref_rt_mod(mod, {})
        assert num_partition > 0

        # x = tvm.nd.array(x_np, device=dev)
        # y = tvm.nd.array(y_np, device=dev)
        # z = tvm.nd.array(z_np, device=dev)
        # out = get_output(rt_mod, ["x", "y"], [x, y])
        # ref_out = get_output(rt_mod_ref, ["x", "y"], [x, y])

    # np.testing.assert_allclose(out, ref_out, atol=atol, rtol=rtol)

    # print("CUTLASS:", rt_mod.benchmark(dev, number=1, repeat=600))
    # print("TVM Tensorcore (no tuning):", rt_mod_ref.benchmark(dev, number=1, repeat=600))

    cost = rt_mod.benchmark(dev, number=1, repeat=600).mean * 1e3
    return cost


def main(
    batch, C, H, W, K1, K2, stride1, stride2, k1, k2, in_dtype, acc_dtype, only_once
):
    return verify_conv2d_chain(
        relay_conv_conv(
            batch,
            C,
            H,
            W,
            K1,
            K2,
            stride1,
            stride2,
            k1,
            k2,
        )
    )


example_text = """
 example:
    python conv_conv_relay.py --in_dtype float16 --acc_dtype float16 --begin 0 --num 1
"""


def ceil(x, y):
    return (x + y - 1) // y


def uround(x, y):
    return int(ceil(x, y) * y)


shapes = [
    # C, H, W, K1, K2, st1, st2, k1, k2
    (64, 112, 112, 192, 128, 2, 1, 3, 1),  # Yolo
    (32, 147, 147, 64, 80, 2, 1, 3, 1),  # Inception-V3
    (64, 56, 56, 128, 64, 1, 1, 3, 1),  # Darknet-19
    (128, 28, 28, 256, 128, 1, 1, 3, 1),  # Darknet-19
    (16, 227, 227, 64, 16, 4, 1, 3, 1),  # Squeezenet-V1.1
    (64, 56, 56, 64, 64, 1, 1, 1, 3),  # ResNet-50
    (64, 56, 56, 64, 64, 1, 1, 1, 1),  # modified ResNet-50
    (256, 56, 56, 256, 64, 1, 1, 1, 1),  # modified ResNet-50
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="base_maker",
        description="template maker",
        epilog=example_text,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--only_once", action="store_true")
    parser.add_argument(
        "--in_dtype",
        type=str,
        choices=["float16", "int8"],
        default="float16",
    )
    parser.add_argument(
        "--acc_dtype",
        type=str,
        choices=["float16", "float32", "int32"],
        default="float16",
    )
    parser.add_argument(
        "--begin", type=int, choices=list(range(len(shapes))), default=0
    )
    parser.add_argument(
        "--num", type=int, choices=list(range(1, len(shapes) + 1)), default=len(shapes)
    )
    parser.add_argument("--batch", type=int, default=1)

    args = parser.parse_args()

    costs = []
    for ss in shapes[args.begin : args.begin + args.num]:
        C, H, W, K1, K2, stride1, stride2, k1, k2 = ss
        cost = main(
            args.batch,
            C,
            H,
            W,
            K1,
            K2,
            stride1,
            stride2,
            k1,
            k2,
            args.in_dtype,
            args.acc_dtype,
            args.only_once,
        )
        costs.append((ss, cost))
    print("batch,C,H,W,K1,K2,stride1,stride2,k1,k2,in_dtype,acc_dtype,cost")
    for cc in costs:
        print(
            f"{args.batch},"
            + ",".join(map(str, cc[0]))
            + f",{args.in_dtype},{args.acc_dtype},{cc[1]}"
        )
