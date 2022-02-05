import tvm
from tvm import relay
import numpy as np
import argparse
import time


def relay_bmm_softmax_bmm(
    batch, M, N, K, L, in_dtype="float16", acc_dtype="float32", target="cuda"
):
    A = relay.var("A", shape=[batch, M, K], dtype=in_dtype)
    B = relay.var("B", shape=[batch, K, L], dtype=in_dtype)
    C = relay.var("C", shape=[batch, L, N], dtype=in_dtype)
    B = relay.transpose(B, axes=(0, 2, 1))
    D = relay.nn.batch_matmul(A, B, in_dtype)
    E = relay.nn.softmax(data=D)
    C = relay.transpose(C, axes=(0, 2, 1))
    F = relay.nn.batch_matmul(E, C, in_dtype)
    args = relay.analysis.free_vars(F)
    func = relay.Function(args, F)

    mod = tvm.IRModule.from_expr(func)
    mod = relay.transform.InferType()(mod)
    params = {}
    import tvm.contrib.graph_executor as runtime

    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build_module.build(mod, target=target, params=params)

        # load parameters
        dev = tvm.device(str(target), 0)
        module = runtime.GraphModule(lib["default"](dev))
    return [[batch, M, K], [batch, K, L], [batch, L, N]], [[batch, M, N]], module


def main(profile):
    in_dtype = "float16"
    acc_dtype = "float32"
    target = "cuda -libs=cublas,cudnn"
    ins, outs, module = relay_bmm_softmax_bmm(
        12, 512, 64, 64, 512, in_dtype=in_dtype, acc_dtype=acc_dtype, target=target
    )

    inputs_np = [
        np.random.uniform(-1, 1, [int(x) for x in y]).astype(in_dtype) for y in ins
    ]

    outputs_np = [
        np.random.uniform(-1, 1, [int(x) for x in y]).astype(in_dtype) for y in outs
    ]
    ctx = tvm.cuda()
    dev = tvm.device(str(target), 0)
    inputs_tvm = [tvm.nd.array(x, ctx) for x in inputs_np]
    outputs_tvm = [tvm.nd.array(x, ctx) for x in outputs_np]
    if profile:
        module.set_input(key=0, value=inputs_tvm[0])
        module.set_input(key=1, value=inputs_tvm[1])
        module.set_input(key=2, value=inputs_tvm[2])
        # module.set_input(key=3, value=outputs_tvm[0])
        module.run()
    else:
        print(module.benchmark(dev, min_repeat_ms=600))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", action="store_true")

    args = parser.parse_args()
    main(args.profile)
