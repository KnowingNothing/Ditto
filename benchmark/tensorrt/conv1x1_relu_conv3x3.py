import torch
import torch.nn as nn
import numpy as np
import tensorrt as trt
import argparse


class ConvChain(nn.Module):
    def __init__(self, C, K1, K2, stride1, stride2):
        super(ConvChain, self).__init__()
        self.conv1 = torch.nn.Conv2d(C, K1, 1, stride=stride1, padding=0, bias=False)
        self.conv2 = torch.nn.Conv2d(K1, K2, 3, stride=stride2, padding=1, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        return x
    
def export_model(batch, C, H, W, K1, K2, stride1, stride2, dtype):
    in_dtype = dtype

    inputs_np = [
        np.random.uniform(-1, 1, [int(x) for x in y]).astype(in_dtype)
        for y in [[batch, C, H, W]]
    ]

    inputs_torch = [torch.tensor(x).cuda() for x in inputs_np]
    model = ConvChain(C, K1, K2, stride1, stride2).half().cuda()
    torch.onnx.export(model, args=tuple(inputs_torch), f=f"conv1x1_relu_conv3x3-{batch}-{C}-{H}-{W}-{K1}-{K2}-{stride1}-{stride2}-{dtype}.onnx", verbose=True)

def main(batch, C, H, W, K1, K2, stride1, stride2, dtype, only_once=False):
    export_model(batch, C, H, W, K1, K2, stride1, stride2, dtype)
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    success = parser.parse_from_file(f"conv1x1_relu_conv3x3-{batch}-{C}-{H}-{W}-{K1}-{K2}-{stride1}-{stride2}-{dtype}.onnx")
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
        for y in [[batch, C, H, W]]
    ]

    outputs_np = [np.random.uniform(-1, 1, [int(x) for x in [batch, K2, H, W]]).astype(dtype)]

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
    python conv1x1_relu_conv3x3.py --dtype float16 --begin 0 --num 1
"""

shapes = [
    (64, 56, 56, 64, 64, 1, 1),  # ResNet-50
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
    parser.add_argument(
        "--batch", type=int, default=1
    )

    args = parser.parse_args()

    costs = []
    for ss in shapes[args.begin : args.begin + args.num]:
        C, H, W, K1, K2, stride1, stride2 = ss
        cost = main(args.batch, C, H, W, K1, K2, stride1, stride2, args.dtype, args.only_once)
        costs.append((ss, cost))
    print("batch,C,H,W,K1,K2,stride1,stride2,dtype,cost")
    for cc in costs:
        print(
            f"{args.batch}," + ",".join(map(str, cc[0])) + f",{args.dtype},{cc[1]}"
        )