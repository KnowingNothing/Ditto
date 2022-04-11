import time
import torch 
import numpy as np
import argparse

REPEAT = 1000
SERVER = None

ServerConfig = {
    'sc':{
        'name': 'sc',
        'parallelism': 20,
        'cacheSizes': [32 * 16, 32 * 1024, 256 * 1024, 25344 * 1024],
        'corePerLevel': [1.0,1.0,1.0,10.0],
        'bandwidth': [293.72, 81.72, 38.54, 13.14],
        'isa': 'avx2',
        'peakgflops': 704
    },
    'sccc':{
        'name':'sccc',
        'parallelism': 32,
        'cacheSizes': [32 * 16, 32 * 1024, 1024 * 1024, 25344 * 1024],
        'corePerLevel': [1.0,1.0,1.0,16.0],
        'bandwidth': [293.72, 81.72, 38.54, 13.14],
        'isa': 'avx512',
        'peakgflops': 2150.5
    },
    'scccc':{
        'name': 'scccc',
        'parallelism': 36,
        'cacheSizes': [32 * 16, 32 * 1024, 1024 * 1024, 25344 * 1024],
        'corePerLevel': [1.0,1.0,1.0,18.0],
        'bandwidth': [293.72, 81.72, 38.54, 13.14],
        'isa': 'avx512',
        'peakgflops': 2995.20
    },
}


def test_torch(shape, dtype = "float32"):
    N, C0, P0, Q0, C1, R1, S1, C2, R2, S2, padding1, padding2, stride1, stride2 = shape
    P0_pad = P0 + 2 * padding1
    Q0_pad = Q0 + 2 * padding1
    P1 = (P0_pad - R1) // stride1 + 1
    Q1 = (Q0_pad - S1) // stride1 + 1
    P1_pad = P1 + 2 * padding2
    Q1_pad = Q1 + 2 * padding2
    P2 = (P1_pad - R2) // stride2 + 1
    Q2 = (Q1_pad - S2) // stride2 + 1
    if dtype == "float32":
        dataType = np.float32
    elif dtype == "int8":
        dataType = np.int8
    num = REPEAT
    per = 100
    num = ((num-1) // per + 1) * per
    cost = 0
    for outest in range(num // per + 1):
        Img_np = []
        Weight1_np = []
        Weight2_np = []

        Img_tensor = []
        Weight1_tensor = []
        Weight2_tensor = []
        O2_tensor = []

        for trial in range(per):
            Img_np.append(np.random.uniform(size = (N, C0, P0, Q0)).astype(dtype))
            Weight1_np.append(np.random.uniform(size = (C1, C0, R1, S1)).astype(dtype))
            Weight2_np.append(np.random.uniform(size = (C2, C1, R1, S1)).astype(dtype))

            Img_tensor.append(torch.from_numpy(Img_np[trial]))
            Weight1_tensor.append(torch.from_numpy(Weight1_np[trial]))
            Weight2_tensor.append(torch.from_numpy(Weight2_np[trial]))



        t = 0
        for trial in range(per):
            start = time.time()
            O1 = torch.nn.functional.conv2d(Img_tensor[trial], Weight1_tensor[trial], padding=padding1,stride=stride1)
            torch.nn.functional.relu(O1, inplace=True)
            O2 = torch.nn.functional.conv2d(O1, Weight2_tensor[trial], padding = padding2, stride = stride2)
            end = time.time()
            t += end - start
            O2_tensor.append(O2)
        t /= per 
        if outest > 0:
            cost += t
#        print("%d: %g" % (outest, cost_tmp))

    cost /= num // per
    print(shape,cost)
    return {'torch_time': cost}

def setGlobals(shape):
    global MI, NI1, KI1, NI2, KI2, WORKLOAD, mk1, mk2
    N, C0, H, W, C1, R1, S1, C2, R2, S2, padding1, padding2, stride1, stride2 = shape
    WORKLOAD = N * (C0 * H * W * C1 * R1 * S1 + C1 * H * W * C2 * R2 * S2)

def main(shape, dtype, server):
    global SERVER
    SERVER = ServerConfig[server]
    setGlobals(shape)
    print("shape,dtype,WORKLOAD,SERVER")
    print(shape, dtype, WORKLOAD, SERVER)
    time = test_torch(shape, dtype)
    ret = {}
    for k in time:
        ret[k] = {}
        ret[k]['time(s)'] = time[k]
        ret[k]['%peak'] = (WORKLOAD / time[k] / 1e9) / SERVER['peakgflops']
        ret[k]['gflops'] = (WORKLOAD / time[k] / 1e9)
    print(shape, dtype, ":", ret)
    return ret


example_text = "python ./conv_relu_conv.py --server sc"


shapes = [
    [1, 64, 114, 112, 192, 3, 3, 128, 1, 1, 1, 0, 1, 1],
    [1, 32, 147, 148, 64, 3, 3, 96, 1, 1, 1, 0, 1, 1], # modify
    [1, 64, 57, 56, 128, 3, 3, 64, 1, 1, 1, 0, 1, 1],
    [1, 128, 27, 28, 256, 3, 3, 128, 1, 1, 1, 0, 1, 1],
    [1, 16, 228, 227, 64, 3, 3, 32, 1, 1, 1, 0, 1, 1], # modify
    [1, 64, 57, 56, 64, 1, 1, 64, 3, 3, 0, 1, 1, 1],
    [1, 64, 57, 56, 64, 1, 1, 64, 1, 1, 0, 0, 1, 1],
    [1, 256, 57, 56, 256, 1, 1, 64, 1, 1, 0, 0, 1, 1]
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
        choices=["float32", 'float64'],
        default="float32",
    )
    parser.add_argument(
        "--begin", type=int, choices=list(range(len(shapes))), default=0
    )
    parser.add_argument(
        "--num", type=int, choices=list(range(1, len(shapes) + 1)), default=len(shapes)
    )
    parser.add_argument(
        "--server", type=str, choices=['sc', 'sccc', 'scccc']
    )

    args = parser.parse_args()

    costs = []
    for ss in shapes[args.begin: args.begin + args.num]:
        cost = main(
            ss,
            dtype=args.dtype,
            server=args.server
        )
        costs.append((ss, cost))

    print("shape,dtype,args,sm,cost")
    for cc in costs:
        print(
            f"{cc[0]},{args.dtype},{args.server},{cc[1]}"
        )
    for cc in costs:
        print(cc[1])
    with open("conv_relu_conv_torch.pkl", "wb") as f:
        pkl.dump(costs, f)

