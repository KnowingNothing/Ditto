import time
import torch 
import numpy as np
import argparse
import pickle as pkl 
import sys 

per = 100
num = 10000

Qtensor = []
Ktensor = []
def dataGen(batch, M, N, K):
    for trial in range(per):
        Qtensor.append(torch.from_numpy(np.random.uniform(-1, 1, (batch, M, K)).astype("float32")))
        Ktensor.append(torch.from_numpy(np.random.uniform(-1, 1, (batch, K, N)).astype("float32")))

def test_torch(batch, M, N, K, dtype = "float32"):
    global per, num
    num = ((num-1) // per + 1) * per
    cost = 0
    for outest in range(num // per + 1):
        for trial in range(per):
            QK = torch.bmm(Qtensor[trial], Ktensor[trial])

    return


if __name__ == "__main__":
    assert len(sys.argv) >= 5
    B = int(sys.argv[1])
    M = int(sys.argv[2])
    N = int(sys.argv[3])
    K = int(sys.argv[4])
    dataGen(B, M, N, K)
    test_torch(B, M, N, K)