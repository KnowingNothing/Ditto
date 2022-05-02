import time
import torch 
import numpy as np
import argparse
import pickle as pkl 
import streamlit as st
import sys


Qtensor = [] 
Ktensor = [] 
Vtensor = []

num = 1000
per = 10
def dataGen(batch, M, N, K, L, dtype):
    global Qtensor, Ktensor, Vtensor
    for _ in range(per):
        Qtensor.append(torch.from_numpy(np.random.uniform(-1, 1, (batch, M, K)).astype(dtype)))
        Ktensor.append(torch.from_numpy(np.random.uniform(-1, 1, (batch, K, N)).astype(dtype)))
        Vtensor.append(torch.from_numpy(np.random.uniform(-1, 1, (batch, N, L)).astype(dtype)))

# def test_func():
#     for i in range(per):
#         QK = torch.bmm(Qtensor[i], Ktensor[i])
#         QKV = torch.bmm(QK, Vtensor[i])

def test_torch(batch, M, N, K, L):
    
    cost = 0
    for i in range(num):
        start = time.time()
        for j in range(per):
            QK = torch.bmm(Qtensor[j], Ktensor[j])
            QKV = torch.bmm(QK, Vtensor[j])
        end = time.time()
        cost += (end - start)
    tmp = torch.get_num_interop_threads()
    cost = cost / (num * per)
    wl = batch * ( M * K * L + M * L * N)
    toPeak = wl/cost/1e9/2995.2
    print("time:", cost)
    print("toPeak:", toPeak)
    return cost, toPeak

if __name__ == "__main__":
    assert len(sys.argv) >= 6
    B = int(sys.argv[1])
    M = int(sys.argv[2])
    N = int(sys.argv[3])
    K = int(sys.argv[4])
    L = int(sys.argv[5])
    dataGenStart = time.time()
    dataGen(B, M, N, K, L, "float32")
    dataGenEnd = time.time()
    print("dataGenTime: ", dataGenEnd - dataGenStart)
    start = time.time()
    test_torch(B, M, N, K, L)
    end = time.time()
    print("time elapsed", end - start)