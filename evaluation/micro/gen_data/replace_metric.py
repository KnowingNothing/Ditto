import pytest
import tvm
from ditto import auto_compute as ac
from ditto import auto_tensorize as at
from ditto import hardware as hw
import math
import numpy as np

from pebble import concurrent
from concurrent.futures import TimeoutError
from pebble import ProcessPool, ProcessExpired

MI = 16
NI = 16
KI = 16
WARP_SIZE = 32
IN_VEC = 4
OUT_VEC = 4


class Metric(object):
    def __init__(self, locality, parallelism, recomputation):
        self.locality = locality
        self.parallelism = parallelism
        self.recomputation = recomputation


def cal_F1(
    B,
    M,
    N,
    K,
    L,
    TM,
    TN,
    TK,
    TL,
    bytePerInputElem,
    bytePerAccElem,
    PeakBlock,
    CoreNum,
    MemByteLimitPerCore,
):
    ReadA = B * (M // TM) * (N // TN) * (L // TL) * TM * K
    ReadB = B * (M // TM) * (N // TN) * (L // TL) * K * TL
    ReadD = B * (M // TM) * (N // TN) * (L // TL) * TL * TN
    WriteE = B * (M // TM) * (N // TN) * (L // TL) * TM * TN
    DV = ReadA + ReadB + ReadD + WriteE

    MemUseA = TM * TK
    MemUseB = TK * TL
    MemUseC = TM * TL
    MemUseD = TL * TN
    MemUseAll = (
        max(MemUseA + MemUseB, MemUseD)
    ) * bytePerInputElem + MemUseC * bytePerAccElem

    locality = 1 / DV if MemUseAll < MemByteLimitPerCore else -float("inf")

    parallelism = min(
        B * (M // TM) * (N // TN),
        min(PeakBlock, CoreNum * (MemByteLimitPerCore // MemUseAll)),
    )

    recomputation = N // TN

    return Metric(locality, parallelism, recomputation)


def cal_F2(
    B,
    M,
    N,
    K,
    L,
    TM,
    TN,
    TK,
    TL,
    bytePerInputElem,
    bytePerAccElem,
    PeakBlock,
    CoreNum,
    MemByteLimitPerCore,
):
    ReadA = B * (M // TM) * (N // TN) * TM * K
    ReadB = B * (M // TM) * (N // TN) * K * L
    ReadD = B * (M // TM) * (N // TN) * L * TN
    WriteE = B * (M // TM) * (N // TN) * TM * TN
    DV = ReadA + ReadB + ReadD + WriteE

    MemUseA = TM * TK
    MemUseB = TK * TL
    MemUseC = TM * L
    MemUseD = TL * TN
    MemUseAll = (
        max(MemUseA + MemUseB, MemUseD)
    ) * bytePerInputElem + MemUseC * bytePerAccElem

    locality = 1 / DV if MemUseAll < MemByteLimitPerCore else -float("inf")

    parallelism = min(
        B * (M // TM) * (N // TN),
        min(PeakBlock, CoreNum * (MemByteLimitPerCore // MemUseAll)),
    )

    recomputation = N // TN

    return Metric(locality, parallelism, recomputation)


def cal_F3(
    B,
    M,
    N,
    K,
    L,
    TM,
    TN,
    TK,
    TL,
    bytePerInputElem,
    bytePerAccElem,
    PeakBlock,
    CoreNum,
    MemByteLimitPerCore,
):
    ReadA = B * (M // TM) * (L // TL) * TM * K
    ReadB = B * (M // TM) * (L // TL) * K * TL
    ReadD = B * (M // TM) * (L // TL) * TL * N
    WriteE = B * (M // TM) * (L // TL) * TM * N
    DV = ReadA + ReadB + ReadD + WriteE

    MemUseA = TM * TK
    MemUseB = TK * TL
    MemUseC = TM * TL
    MemUseD = TL * TN
    MemUseAll = (
        max(MemUseA + MemUseB, MemUseD)
    ) * bytePerInputElem + MemUseC * bytePerAccElem

    locality = 1 / DV if MemUseAll < MemByteLimitPerCore else -float("inf")

    parallelism = min(
        B * (M // TM),
        min(PeakBlock, CoreNum * (MemByteLimitPerCore // MemUseAll)),
    )

    recomputation = 0

    return Metric(locality, parallelism, recomputation)


def cal_F4(
    B,
    M,
    N,
    K,
    L,
    TM,
    TN,
    TK,
    TL,
    bytePerInputElem,
    bytePerAccElem,
    PeakBlock,
    CoreNum,
    MemByteLimitPerCore,
):
    ReadA = B * (N // TN) * (L // TL) * M * K
    ReadB = B * (N // TN) * (L // TL) * K * TL
    ReadD = B * (N // TN) * (L // TL) * TL * TN
    WriteE = B * (N // TN) * (L // TL) * M * TN
    DV = ReadA + ReadB + ReadD + WriteE

    MemUseA = TM * TK
    MemUseB = TK * TL
    MemUseC = M * TL
    MemUseD = TL * TN
    MemUseAll = (
        max(MemUseA + MemUseB, MemUseD)
    ) * bytePerInputElem + MemUseC * bytePerAccElem

    locality = 1 / DV if MemUseAll < MemByteLimitPerCore else -float("inf")

    parallelism = min(
        B * (N // TN),
        min(PeakBlock, CoreNum * (MemByteLimitPerCore // MemUseAll)),
    )

    recomputation = N // TN

    return Metric(locality, parallelism, recomputation)


def cal_F5(
    B,
    M,
    N,
    K,
    L,
    TM,
    TN,
    TK,
    TL,
    bytePerInputElem,
    bytePerAccElem,
    PeakBlock,
    CoreNum,
    MemByteLimitPerCore,
):
    ReadA = B * (M // TM) * TM * K
    ReadB = B * (M // TM) * K * L
    ReadD = B * (M // TM) * L * N
    WriteE = B * (M // TM) * TM * N
    DV = ReadA + ReadB + ReadD + WriteE

    MemUseA = TM * TK
    MemUseB = TK * TL
    MemUseC = TM * L
    MemUseD = TL * TN
    MemUseAll = (
        max(MemUseA + MemUseB, MemUseD)
    ) * bytePerInputElem + MemUseC * bytePerAccElem

    locality = 1 / DV if MemUseAll < MemByteLimitPerCore else -float("inf")

    parallelism = min(
        B * (M // TM),
        min(PeakBlock, CoreNum * (MemByteLimitPerCore // MemUseAll)),
    )

    recomputation = 0

    return Metric(locality, parallelism, recomputation)


def cal_F6(
    B,
    M,
    N,
    K,
    L,
    TM,
    TN,
    TK,
    TL,
    bytePerInputElem,
    bytePerAccElem,
    PeakBlock,
    CoreNum,
    MemByteLimitPerCore,
):
    ReadA = B * (N // TN) * M * K
    ReadB = B * (N // TN) * K * L
    ReadD = B * (N // TN) * L * TN
    WriteE = B * (N // TN) * M * TN
    DV = ReadA + ReadB + ReadD + WriteE

    MemUseA = TM * TK
    MemUseB = TK * TL
    MemUseC = M * L
    MemUseD = TL * TN
    MemUseAll = (
        max(MemUseA + MemUseB, MemUseD)
    ) * bytePerInputElem + MemUseC * bytePerAccElem

    locality = 1 / DV if MemUseAll < MemByteLimitPerCore else -float("inf")

    parallelism = min(
        B * (N // TN),
        min(PeakBlock, CoreNum * (MemByteLimitPerCore // MemUseAll)),
    )

    recomputation = N // TN

    return Metric(locality, parallelism, recomputation)


def cal_F7(
    B,
    M,
    N,
    K,
    L,
    TM,
    TN,
    TK,
    TL,
    bytePerInputElem,
    bytePerAccElem,
    PeakBlock,
    CoreNum,
    MemByteLimitPerCore,
):
    ReadA = B * (L // TL) * M * K
    ReadB = B * (L // TL) * K * TL
    ReadD = B * (L // TL) * TL * N
    WriteE = B * (L // TL) * M * N
    DV = ReadA + ReadB + ReadD + WriteE

    MemUseA = TM * TK
    MemUseB = TK * TL
    MemUseC = M * TL
    MemUseD = TL * TN
    MemUseAll = (
        max(MemUseA + MemUseB, MemUseD)
    ) * bytePerInputElem + MemUseC * bytePerAccElem

    locality = 1 / DV if MemUseAll < MemByteLimitPerCore else -float("inf")

    parallelism = min(
        B * 1,
        min(PeakBlock, CoreNum * (MemByteLimitPerCore // MemUseAll)),
    )

    recomputation = 0

    return Metric(locality, parallelism, recomputation)


def replace(batch, M, N, K, L, f_cal, name):
    data = []
    with open(f"dataset_attention_batch_gemm_chain_{name}.csv", "r") as fin:
        fin.readline()
        for line in fin:
            (
                fid,
                valid,
                Tm,
                Tn,
                Tk,
                Tl,
                locality,
                parallelism,
                recomputation,
                cost,
            ) = line.strip().split(",")
            fid = int(fid)
            valid = bool(valid)
            Tm = int(Tm)
            Tn = int(Tn)
            Tk = int(Tk)
            Tl = int(Tl)
            locality = float(locality)
            parallelism = float(parallelism)
            recomputation = float(recomputation)
            cost = float(cost)
            data.append(
                (fid, valid, Tm, Tn, Tk, Tl, locality, parallelism, recomputation, cost)
            )

    fout = open(f"replaced_dataset_attention_batch_gemm_chain_{name}.csv", "w")
    print("id,valid,Tm,Tn,Tk,Tl,locality,parallelism,recomputation,cost", flush=True)
    print(
        "id,valid,Tm,Tn,Tk,Tl,locality,parallelism,recomputation,cost",
        file=fout,
        flush=True,
    )
    for (
        fid,
        valid,
        tm,
        tn,
        tk,
        tl,
        locality,
        parallelism,
        recomputation,
        cost,
    ) in data:
        metric = f_cal(
            batch,
            M,
            N,
            K,
            L,
            tm,
            tn,
            tk,
            tl,
            2,
            4,
            108 * 32,
            108,
            128 * 1024,
        )
        print(
            f"{fid},{cost<1e10},{tm},{tn},{tk},{tl},{metric.locality},{metric.parallelism},{metric.recomputation},{cost}",
            flush=True,
        )
        print(
            f"{fid},{cost<1e10},{tm},{tn},{tk},{tl},{metric.locality},{metric.parallelism},{metric.recomputation},{cost}",
            file=fout,
            flush=True,
        )
    fout.close()


if __name__ == "__main__":
    batch = 12
    M = 512
    N = 64
    K = 64
    L = 512

    funcs = [cal_F1, cal_F2, cal_F3, cal_F4, cal_F5, cal_F6, cal_F7]
    names = ["F1", "F2", "F3", "F4", "F5", "F6", "F7"]

    for i in range(0, 7):
        replace(batch, M, N, K, L, funcs[i], names[i])
