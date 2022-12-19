import time
import torch 
import numpy as np
import argparse
import streamlit.runtime.legacy_caching as caching 
import pickle as pkl
import os

REPEAT = 2000
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
    'Xeon-Gold-6348': {
        'parallelism': 112,
        'cacheSizes': [32 * 32, 2.6 * 1024 * 1024, 70 * 1024 * 1024, 84 * 1024 * 1024],
        'corePerLevel': [1.0, 1.0, 1.0, 28.0],
        'bandwidth': [293.72, 100.72, 50.54, 13.14],
        'isa': 'avx512',
        'peakgflops': 4659.2
    }
}


Weight1_tensor = None 
Weight2_tensor = None 
Img_tensor = None 

def dataGen(shape, dtype):
    N, C0, P0, Q0, C1, R1, S1, C2, R2, S2, padding1, padding2, stride1, stride2 = shape
    P0_pad = P0 + 2 * padding1
    Q0_pad = Q0 + 2 * padding1
    P1 = (P0_pad - R1) // stride1 + 1
    Q1 = (Q0_pad - S1) // stride1 + 1
    P1_pad = P1 + 2 * padding2
    Q1_pad = Q1 + 2 * padding2
    P2 = (P1_pad - R2) // stride2 + 1
    Q2 = (Q1_pad - S2) // stride2 + 1
    global Img_tensor, Weight1_tensor, Weight2_tensor
    Img_tensor = torch.tensor(np.random.uniform(size = (N, C0, P0, Q0)).astype(dtype))
    Weight1_tensor = torch.tensor(np.random.uniform(size = (C1, C0, R1, S1)).astype(dtype))
    Weight2_tensor = torch.tensor(np.random.uniform(size = (C2, C1, R2, S2)).astype(dtype))



def test_torch(shape, dtype = "float32"):
    # warm up
    N, C0, P0, Q0, C1, R1, S1, C2, R2, S2, padding1, padding2, stride1, stride2 = shape
    P0_pad = P0 + 2 * padding1
    Q0_pad = Q0 + 2 * padding1
    P1 = (P0_pad - R1) // stride1 + 1
    Q1 = (Q0_pad - S1) // stride1 + 1
    P1_pad = P1 + 2 * padding2
    Q1_pad = Q1 + 2 * padding2
    P2 = (P1_pad - R2) // stride2 + 1
    Q2 = (Q1_pad - S2) // stride2 + 1
    O1 = torch.nn.functional.conv2d(Img_tensor, Weight1_tensor, padding=padding1,stride=stride1)
    torch.nn.functional.relu(O1, inplace=True)
    O2 = torch.nn.functional.conv2d(O1, Weight2_tensor, padding = padding2, stride = stride2)
    cost = 0
    for _ in range(REPEAT):
        caching.clear_cache()
        start = time.time()
        O1 = torch.nn.functional.conv2d(Img_tensor, Weight1_tensor, padding=padding1,stride=stride1)
        torch.nn.functional.relu(O1, inplace=True)
        O2 = torch.nn.functional.conv2d(O1, Weight2_tensor, padding = padding2, stride = stride2)
        end = time.time()
        cost += end - start
    cost = cost / REPEAT
    print(shape,cost)
    return cost

def setGlobals(shape):
    global MI, NI1, KI1, NI2, KI2, WORKLOAD
    N, C0, H, W, C1, R1, S1, C2, R2, S2, padding1, padding2, stride1, stride2 = shape
    WORKLOAD = N * (C0 * H * W * C1 * R1 * S1 + C1 * H * W * C2 * R2 * S2)

def main(shape, dtype, server):
    global SERVER
    SERVER = ServerConfig[server]
    setGlobals(shape)
    print("shape,dtype,WORKLOAD,SERVER")
    print(shape, dtype, WORKLOAD, SERVER)
    time = test_torch(shape, dtype)
    toPeak = WORKLOAD / time / 1e9 / SERVER['peakgflops']
    print(shape, dtype, ":", time, toPeak)
    return time, toPeak


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
        "--server", type=str, choices=ServerConfig.keys()
    )
    
    parser.add_argument(
        "--output", type=str, default = "result"
    )

    args = parser.parse_args()

    costs = []
    for ss in shapes[args.begin: args.begin + args.num]:
        dataGen(ss, args.dtype)
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

    os.system(f'mkdir -p {args.output}')
    with open(f"{args.output}/conv_relu_conv-torch.pkl", "wb") as f:
        pkl.dump(costs, f)

