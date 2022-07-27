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


def relay_bmm_bmm(
    batch, M, N, K, L, in_dtype="float16", acc_dtype="float16", target="cuda"
):
    A = relay.var("A", shape=[batch, M, K], dtype=in_dtype)
    B = relay.var("B", shape=[batch, K, L], dtype=in_dtype)
    C = relay.var("C", shape=[batch, L, N], dtype=in_dtype)
    B = relay.transpose(B, axes=(0, 2, 1))
    D = relay.nn.batch_matmul(A, B, acc_dtype)
    # E = relay.cast(D, dtype=in_dtype)
    C = relay.transpose(C, axes=(0, 2, 1))
    F = relay.nn.batch_matmul(D, C, acc_dtype)
    # G = relay.cast(F, dtype=in_dtype)
    args = relay.analysis.free_vars(F)
    func = relay.Function(args, F)
    
    return func


def has_cutlass():
    return tvm.get_global_func("relay.ext.cutlass", True) != None


def profile_and_build(mod, params, sm, tmp_dir="./tmp", lib_path="compile.so", use_fast_math=False):
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


def verify_batch_matmul_chain(
    func, batch, M, N, K, L, ref_target="cuda", sm=80, atol=1e-5, rtol=1e-5
):
    if not has_cutlass():
        return
    mod = tvm.IRModule.from_expr(func)
    typ = relay.transform.InferType()(mod)["main"].body.checked_type
    use_vm = any(isinstance(s, tvm.tir.Any) for s in typ.shape)
    x_np = np.random.uniform(-1, 1, (batch, M, K)).astype("float16")
    y_np = np.random.uniform(-1, 1, (batch, K, L)).astype("float16")
    z_np = np.random.uniform(-1, 1, (batch, L, N)).astype("float16")

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
        rt_mod_ref, dev = get_ref_rt_mod(mod, {})
        assert num_partition > 0

        x = tvm.nd.array(x_np, device=dev)
        y = tvm.nd.array(y_np, device=dev)
        z = tvm.nd.array(z_np, device=dev)
        # out = get_output(rt_mod, ["x", "y"], [x, y])
        # ref_out = get_output(rt_mod_ref, ["x", "y"], [x, y])

    # np.testing.assert_allclose(out, ref_out, atol=atol, rtol=rtol)

    # print("CUTLASS:", rt_mod.benchmark(dev, number=1, repeat=600))
    # print("TVM Tensorcore (no tuning):", rt_mod_ref.benchmark(dev, number=1, repeat=600))
    
    cost = rt_mod.benchmark(dev, number=1, repeat=600).mean * 1e3
    return cost



def main(B, M, N, K, L, in_dtype, acc_dtype, only_once):
    return verify_batch_matmul_chain(relay_bmm_bmm(B, M, N, K, L), B, M, N, K, L)


example_text = """
 example:
    python bmm_bmm_bolt.py --in_dtype float16 --acc_dtype float16 --begin 0 --num 1
"""

def ceil(x, y):
    return (x + y - 1) // y


def uround(x, y):
    return int(ceil(x, y) * y)

shapes = [
    # (batch, M, N, K, L)
    (8, 512, 512 // 8, 512 // 8, 512),  # Bert-Small
    (12, 512, 768 // 12, 768 // 12, 512),  # Bert-Base
    (16, 512, 1024 // 16, 1024 // 16, 512),  # Bert-Large
    (12, 256, 768 // 12, 768 // 12, 256),  # ViT-Base/14
    (16, 256, 1024 // 16, 1024 // 16, 256),  # ViT-Large/14
    (16, 256, 1280 // 16, 1280 // 16, 256),  # ViT-Huge/14
    (12, uround(196, 16), 768 // 12, 768 // 12, uround(196, 16)),  # ViT-Base/16
    (16, uround(196, 16), 1024 // 16, 1024 // 16, uround(196, 16)),  # ViT-Large/16
    (16, uround(196, 16), 1280 // 16, 1280 // 16, uround(196, 16)),  # ViT-Huge/16
    (1, 512, uround(49, 16), uround(49, 16), 256),  # Mixer-Small/32-S
    (1, 768, uround(49, 16), uround(49, 16), 384),  # Mixer-Base/32-S
    (1, 1024, uround(49, 16), uround(49, 16), 512),  # Mixer-Large/32-S
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

    args = parser.parse_args()
    costs = []
    for ss in shapes[args.begin : args.begin + args.num]:
        B, M, N, K, L = ss
        cost = main(B, M, N, K, L, args.in_dtype, args.acc_dtype, args.only_once)
        costs.append((ss, cost))
    print("B,M,N,K,L,in_dtype,acc_dtype,cost")
    for cc in costs:
        print(f"{cc[0][0]},{cc[0][1]},{cc[0][2]},{cc[0][3]},{cc[0][4]},{args.in_dtype},{args.acc_dtype},{cc[1]}")
