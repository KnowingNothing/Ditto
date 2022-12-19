import time
import torch 
import numpy as np
import argparse
import pickle as pkl 
import os
import streamlit as st

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


Qtensor = []
Ktensor = []
Vtensor = []

num = 2000
per = 1
def dataGen(batch, M, N, K, L, dtype):
    global Qtensor, Ktensor, Vtensor
    for _ in range(per):
        Qtensor.append(torch.from_numpy(np.random.uniform(-1, 1, (batch, M, K)).astype(dtype)))
        Ktensor.append(torch.from_numpy(np.random.uniform(-1, 1, (batch, K, N)).astype(dtype)))
        Vtensor.append(torch.from_numpy(np.random.uniform(-1, 1, (batch, N, L)).astype(dtype)))

@st.experimental_singleton
def test_func():
    for i in range(per):
        QK = torch.bmm(Qtensor[i], Ktensor[i])
        QKV = torch.bmm(QK, Vtensor[i])



def test_torch(batch, M, N, K, L, dtype = "float32"):
    # warm up
    test_func()
    
    cost = 0
    for i in range(num):
        st.experimental_singleton.clear()
        start = time.time()
        test_func()
        # for j in range(per):
        #     QK = torch.bmm(Qtensor[j], Ktensor[j])
        #     QKV = torch.bmm(QK, Vtensor[j])
        end = time.time()
        cost += (end - start)
    tmp = torch.get_num_interop_threads()
    cost = cost / (num * per)
    wl = batch * ( M * K * L + M * L * N)
    ratioToPeak = (wl / cost / 1e9) / SERVER['peakgflops']
    return cost, ratioToPeak

example_text = "python bmm_bmm.py --begin 0 --num 1"
def ceil(x, y):
    return (x + y - 1) // y


def uround(x, y):
    return int(ceil(x, y) * y)

shapes = [
    # (batch, M, N, K, L)
    (8, 512, 512 // 8, 512 // 8, 512),      # Bert-Small
    (12, 512, 768 // 12, 768 // 12, 512),   # Bert-Base
    (16, 512, 1024 // 16, 1024 // 16, 512), # Bert-Large
    (12, 256, 768 // 12, 768 // 12, 256),   # ViT-Base/14
    (16, 256, 1024 // 16, 1024 // 16, 256), # ViT-Large/14
    (16, 256, 1280 // 16, 1280 // 16, 256), # ViT-Huge/14
    (12, uround(196, 16), 768 // 12, 768 // 12, uround(196, 16)),   # ViT-Base/16
    (16, uround(196, 16), 1024 // 16, 1024 // 16, uround(196, 16)), # ViT-Large/16
    (16, uround(196, 16), 1280 // 16, 1280 // 16, uround(196, 16)), # ViT-Huge/16
    (1, 512, uround(49, 16), uround(49, 16), 256),  # Mixer-Small/32-S
    (1, 768, uround(49, 16), uround(49, 16), 384),  # Mixer-Base/32-S
    (1, 1024, uround(49, 16), uround(49, 16), 512), # Mixer-Large/32-S
]

def setServer(server):
    global SERVER
    SERVER = ServerConfig[server]
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="base_maker",
        description="template maker",
        epilog=example_text,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--only_once", action="store_true")
    parser.add_argument("--enable_cudnn", action="store_true")
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["float32", "int8"],
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
        "--output", type = str, default = "result"
    )

    args = parser.parse_args()
    
    os.system(f'mkdir -p {args.output}')

    setServer(args.server)
    print ("the Server:", SERVER)
    # torch.set_num_interop_threads(1)
    costs = []
    for ss in shapes[args.begin : args.begin + args.num]:
        B, M, N, K, L = ss
        dataGen(*ss, args.dtype)
        cost = test_torch(B, M, N, K, L, args.dtype)
        costs.append((ss, cost))
    print("B,M,N,K,L,dtype,cost,SERVER")
    print("torch parallel info", torch.__config__.parallel_info())
    for cc in costs:
        print(f"{cc[0][0]},{cc[0][1]},{cc[0][2]},{cc[0][3]},{cc[0][4]},{args.dtype},{cc[1]},",SERVER)
    with open(f"{args.output}/bmm_bmm-torch.pkl", 'wb') as f:
        pkl.dump(costs, f)
    