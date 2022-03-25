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


def test_torch(batch, M, N, K, L, dtype = "float32"):
    if dtype == "float32":
        dataType = np.float32
    elif dtype == "int8":
        dataType = np.int8
    num = REPEAT
    per = 100
    num = ((num-1) // per + 1) * per
    cost = 0
    for outest in range(num // per + 1):
        Q_np = []
        K_np = []
        V_np = []
        Qtensor = []
        Ktensor = []
        Vtensor = []
        QKV_np = []
        for trial in range(per):
            Q_np.append(np.random.uniform(
                size=(batch, M, K)).astype(dataType))
            K_np.append(np.random.uniform(
                size=(batch, K, L)).astype(dataType))
            V_np.append(np.random.uniform(
                size=(batch, L, N)).astype(dataType))
            Qtensor.append(torch.from_numpy(Q_np[trial]))
            Ktensor.append(torch.from_numpy(K_np[trial]))
            Vtensor.append(torch.from_numpy(V_np[trial]))
            QKV_np.append(np.random.uniform(
                size=(batch, M, N)).astype(dataType))
            for i in range(batch):
                QKV_np[trial][i] = Q_np[trial][i].dot(
                    K_np[trial][i]).dot(V_np[trial][i])

        QKV = []

        start = time.time()
        for trial in range(per):
            QK = torch.bmm(Qtensor[trial], Ktensor[trial])
            # QK_relu = torch.nn.ReLU()(QK)
            QKV.append(torch.bmm(QK, Vtensor[trial]))

        end = time.time()
        cost_tmp = (end - start) / per
        if outest > 0:
            cost += cost_tmp
#        print("%d: %g" % (outest, cost_tmp))

        for trial in range(per):
            np.testing.assert_allclose(
                QKV_np[trial], QKV[trial].numpy(), rtol=1e-3)

    cost /= num // per
    wl = batch * ( M * K * L + M * L * N)
    ratioToPeak = (wl / cost / 1e9) / SERVER['peakgflops']
    print(batch,M,N,K,L,cost,ratioToPeak)
    return cost, ratioToPeak
example_text = "python bmm_bmm.py --begin 0 --num 1"
shapes = [
    # (batch, M, N, K, L)
    (8, 512, 512 // 8, 512 // 8, 512),  # Bert-Small
    (12, 512, 768 // 12, 768 // 12, 512),  # Bert-Base
    (16, 512, 1024 // 16, 1024 // 16, 512),  # Bert-Large
    (12, 256, 768 // 12, 768 // 12, 256),  # ViT-Base/14
    (16, 256, 1024 // 16, 1024 // 16, 256),  # ViT-Large/14
    (16, 256, 1280 // 16, 1280 // 16, 256),  # ViT-Huge/14
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
        "--server", type=str, choices=['sc', 'sccc', 'scccc']
    )
    
    args = parser.parse_args()

    setServer(args.server)
    print ("the Server:", SERVER)
    costs = []
    for ss in shapes[args.begin : args.begin + args.num]:
        B, M, N, K, L = ss
        cost = test_torch(B, M, N, K, L, args.dtype)
        costs.append((ss, cost))
    print("B,M,N,K,L,dtype,cost,SERVER")
    for cc in costs:
        print(f"{cc[0][0]},{cc[0][1]},{cc[0][2]},{cc[0][3]},{cc[0][4]},{args.dtype},{cc[1]},",SERVER)
