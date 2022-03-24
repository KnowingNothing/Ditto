import torch
import torch.nn as nn
import numpy as np
import tensorrt as trt
import argparse


class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
        self.softmax = nn.Softmax(-1)

    def forward(self, q, k, v):
        x = torch.bmm(q, k)
        x = self.softmax(x)
        x = torch.bmm(x, v)
        return x
    
def export_model(B, M, N, K, L, dtype):
    in_dtype = dtype

    inputs_np = [
        np.random.uniform(-1, 1, [int(x) for x in y]).astype(in_dtype)
        for y in [[B, M, K], [B, K, L], [B, L, N]]
    ]

    inputs_torch = [torch.tensor(x).cuda() for x in inputs_np]
    model = Attention()
    torch.onnx.export(model, args=tuple(inputs_torch), f=f"bmm_softmax_bmm-{B}-{M}-{N}-{K}-{L}-{dtype}.onnx", verbose=True)

def main(B, M, N, K, L, dtype, only_once=False):
    export_model(B, M, N, K, L, dtype)
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    success = parser.parse_from_file(f"bmm_softmax_bmm-{B}-{M}-{N}-{K}-{L}-{dtype}.onnx")
    for idx in range(parser.num_errors):
        print(parser.get_error(idx))

    assert success
    config = builder.create_builder_config()
    if dtype == "float16":
        config.set_flag(trt.BuilderFlag.FP16)
    else:
        raise RuntimeError(f"No support for dtype {dtype}.")
    # print(dir(config))
    # config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1<<20) # 1MB
    serialized_engine = builder.build_serialized_network(network, config)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(serialized_engine)

    context = engine.create_execution_context()

    inputs_np = [
        np.random.uniform(-1, 1, [int(x) for x in y]).astype(dtype)
        for y in [[B, M, K], [B, K, L], [B, L, N]]
    ]

    outputs_np = [np.random.uniform(-1, 1, [int(x) for x in [B, M, N]]).astype(dtype)]

    inputs_torch = [torch.tensor(x).cuda() for x in inputs_np]
    outputs_torch = [torch.tensor(x).cuda() for x in outputs_np]
    inputs_ptr = [x.data_ptr() for x in inputs_torch]
    outputs_ptr = [x.data_ptr() for x in outputs_torch]

    stream = torch.cuda.Stream()
    if only_once:
        context.execute_async_v2(inputs_ptr + outputs_ptr, stream.cuda_stream)
        cost = -1
    else:
        repeat = 5
        # warm up
        for i in range(repeat):
            context.execute_async_v2(inputs_ptr + outputs_ptr, stream.cuda_stream)

        costs = []
        for i in range(repeat):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            stream.synchronize()
            start.record(stream=stream)
            context.execute_async_v2(inputs_ptr + outputs_ptr, stream.cuda_stream)
            end.record(stream=stream)
            stream.synchronize()
            total = start.elapsed_time(end)
            costs.append(total)
        cost = np.mean(costs)
    return cost


example_text = """
 example:
    python bmm_softmax_bmm.py --dtype float16 --begin 0 --num 1
"""

shapes = [
    # (batch, M, N, K, L)
    (8, 512, 512 // 8, 512 // 8, 512),  # Bert-Small
    (12, 512, 768 // 12, 768 // 12, 512),  # Bert-Base
    (16, 512, 1024 // 16, 1024 // 16, 512),  # Bert-Large
    (12, 256, 768 // 12, 768 // 12, 256),  # ViT-Base/14
    (16, 256, 1024 // 16, 1024 // 16, 256),  # ViT-Large/14
    (16, 256, 1280 // 16, 1280 // 16, 256),  # ViT-Huge/14
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
        "--dtype",
        type=str,
        choices=["float16", "int8"],
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
        cost = main(B, M, N, K, L, args.dtype, args.only_once)
        costs.append((ss, cost))
    print("B,M,N,K,L,dtype,cost")
    for cc in costs:
        print(f"{cc[0][0]},{cc[0][1]},{cc[0][2]},{cc[0][3]},{cc[0][4]},{args.dtype},{cc[1]}")
