import sys
import tvm 
import numpy as np
import time
import pickle as pkl
import torch
import streamlit as st


data = []
Ktensor = []
Vtensor = []

number = 1000
per = 10

def dataGen(tensor_shapes):
    global data
    for trial in range(per):
        tmp = []
        for tensor_shape in tensor_shapes:
            tmp.append(tvm.nd.array(torch.from_numpy(np.random.uniform(-1, 1, tensor_shape).astype("float32")), tvm.cpu()))
        data.append(tmp)

def test_func(_func):
    for i in range(per):
        _func(*data[i])

def main(B, M, N, K, L, libname):
    func = tvm.runtime.load_module(libname)
    # eval = func.time_evaluator(func.entry_name, dev = tvm.cpu(), repeat = number, number=1, f_preproc="cache_flush_cpu_non_first_arg")
    # cost = eval(*data[0])
    # print(cost)
    cost = 0
    for i in range(number):
        start = time.time()
        test_func(func)
        end = time.time()
        cost += end - start
    cost /= (number*per)
    wl = B * M * L * (K + N)
    toPeak = wl/cost/1e9/2995.2
    print("time:", cost)
    print("toPeak:", toPeak)

def ceil(x, y):
    return (x + y - 1) // y

def uround(x, y):
    return int(ceil(x, y) * y)

def setGlobals(B, M, N, K, L, dtype):
    global MI, NI1, KI1, NI2, KI2
    MI = 6
    NI1 = 64 if N % 64 == 0 else 32 
    KI1 = 64 if K % 64 == 0 else K
    KI2 = NI1
    NI2 = 64 if L % 64 == 0 else 32
    M = uround(M, MI)
    N = uround(N, NI2)
    L = uround(L, NI1)
    return (B, M, N, K, L)

if __name__ == "__main__":
    assert len(sys.argv) >= 7
    libname = sys.argv[6]
    B = int(sys.argv[1])
    M = int(sys.argv[2])
    N = int(sys.argv[3])
    K = int(sys.argv[4])
    L = int(sys.argv[5])
    B, M, N, K, L = setGlobals(B, M, N, K, L, "float32")
    tensor_shapes = [[B, M // MI, MI, K // KI1, KI1], [B, K // KI1, KI1, L // NI1, NI1], [B, L // KI2, KI2, N // NI2, NI2], [B, M, N]]
    # eliminate the overhead of data preparation in torch
    dataGen(tensor_shapes)
    start = time.time()
    main(B, M, N, K, L, libname)
    end = time.time()
    print("time elapsed", end - start)
