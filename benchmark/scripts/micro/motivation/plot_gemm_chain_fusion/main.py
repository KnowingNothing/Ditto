from matplotlib import pyplot as plt
import matplotlib
import numpy as np
from matplotlib import rcParams
import argparse

params = {
    "font.family": "serif",
    "font.serif": "Arial",
    # 'font.style':'italic',
    "font.weight": "normal",  # or 'blod'
    "font.size": 12,  # or large,small
}
rcParams.update(params)


class Metric(object):
    def __init__(self, locality, parallelism, recomputation):
        self.locality = locality
        self.parallelism = parallelism
        self.recomputation = recomputation


def cal_F1(
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
    ReadA = (M // TM) * (N // TN) * (L // TL) * TM * K
    ReadB = (M // TM) * (N // TN) * (L // TL) * K * TL
    ReadD = (M // TM) * (N // TN) * (L // TL) * TL * TN
    WriteE = (M // TM) * (N // TN) * (L // TL) * TM * TN
    DV = ReadA + ReadB + ReadD + WriteE

    MemUseA = TM * TK
    MemUseB = TK * TL
    MemUseC = TM * TL
    MemUseD = TL * TN
    MemUseAll = (
        MemUseA + MemUseB + MemUseD
    ) * bytePerInputElem + MemUseC * bytePerAccElem

    locality = 1 / DV if MemUseAll < MemByteLimitPerCore else -float("inf")

    parallelism = min(
        (M // TM) * (N // TN),
        min(PeakBlock, CoreNum * (MemByteLimitPerCore // MemUseAll)),
    )

    recomputation = N // TN

    return Metric(locality, parallelism, recomputation)


def cal_F2(
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
    ReadA = (M // TM) * (N // TN) * TM * K
    ReadB = (M // TM) * (N // TN) * K * L
    ReadD = (M // TM) * (N // TN) * L * TN
    WriteE = (M // TM) * (N // TN) * TM * TN
    DV = ReadA + ReadB + ReadD + WriteE

    MemUseA = TM * TK
    MemUseB = TK * TL
    MemUseC = TM * L
    MemUseD = TL * TN
    MemUseAll = (
        MemUseA + MemUseB + MemUseD
    ) * bytePerInputElem + MemUseC * bytePerAccElem

    locality = 1 / DV if MemUseAll < MemByteLimitPerCore else -float("inf")

    parallelism = min(
        (M // TM) * (N // TN),
        min(PeakBlock, CoreNum * (MemByteLimitPerCore // MemUseAll)),
    )

    recomputation = N // TN

    return Metric(locality, parallelism, recomputation)


def cal_F3(
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
    ReadA = (M // TM) * (L // TL) * TM * K
    ReadB = (M // TM) * (L // TL) * K * TL
    ReadD = (M // TM) * (L // TL) * TL * N
    WriteE = (M // TM) * (L // TL) * TM * N
    DV = ReadA + ReadB + ReadD + WriteE

    MemUseA = TM * TK
    MemUseB = TK * TL
    MemUseC = TM * TL
    MemUseD = TL * TN
    MemUseAll = (
        MemUseA + MemUseB + MemUseD
    ) * bytePerInputElem + MemUseC * bytePerAccElem

    locality = 1 / DV if MemUseAll < MemByteLimitPerCore else -float("inf")

    parallelism = min(
        (M // TM),
        min(PeakBlock, CoreNum * (MemByteLimitPerCore // MemUseAll)),
    )

    recomputation = 0

    return Metric(locality, parallelism, recomputation)


def cal_F4(
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
    ReadA = (N // TN) * (L // TL) * M * K
    ReadB = (N // TN) * (L // TL) * K * TL
    ReadD = (N // TN) * (L // TL) * TL * TN
    WriteE = (N // TN) * (L // TL) * M * TN
    DV = ReadA + ReadB + ReadD + WriteE

    MemUseA = TM * TK
    MemUseB = TK * TL
    MemUseC = M * TL
    MemUseD = TL * TN
    MemUseAll = (
        MemUseA + MemUseB + MemUseD
    ) * bytePerInputElem + MemUseC * bytePerAccElem

    locality = 1 / DV if MemUseAll < MemByteLimitPerCore else -float("inf")

    parallelism = min(
        (N // TN),
        min(PeakBlock, CoreNum * (MemByteLimitPerCore // MemUseAll)),
    )

    recomputation = N // TN

    return Metric(locality, parallelism, recomputation)


def cal_F5(
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
    ReadA = (M // TM) * TM * K
    ReadB = (M // TM) * K * L
    ReadD = (M // TM) * L * N
    WriteE = (M // TM) * TM * N
    DV = ReadA + ReadB + ReadD + WriteE

    MemUseA = TM * TK
    MemUseB = TK * TL
    MemUseC = TM * L
    MemUseD = TL * TN
    MemUseAll = (
        MemUseA + MemUseB + MemUseD
    ) * bytePerInputElem + MemUseC * bytePerAccElem

    locality = 1 / DV if MemUseAll < MemByteLimitPerCore else -float("inf")

    parallelism = min(
        (M // TM),
        min(PeakBlock, CoreNum * (MemByteLimitPerCore // MemUseAll)),
    )

    recomputation = 0

    return Metric(locality, parallelism, recomputation)


def cal_F6(
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
    ReadA = (N // TN) * M * K
    ReadB = (N // TN) * K * L
    ReadD = (N // TN) * L * TN
    WriteE = (N // TN) * M * TN
    DV = ReadA + ReadB + ReadD + WriteE

    MemUseA = TM * TK
    MemUseB = TK * TL
    MemUseC = M * L
    MemUseD = TL * TN
    MemUseAll = (
        MemUseA + MemUseB + MemUseD
    ) * bytePerInputElem + MemUseC * bytePerAccElem

    locality = 1 / DV if MemUseAll < MemByteLimitPerCore else -float("inf")

    parallelism = min(
        (N // TN),
        min(PeakBlock, CoreNum * (MemByteLimitPerCore // MemUseAll)),
    )

    recomputation = N // TN

    return Metric(locality, parallelism, recomputation)


def cal_F7(
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
    ReadA = (L // TL) * M * K
    ReadB = (L // TL) * K * TL
    ReadD = (L // TL) * TL * N
    WriteE = (L // TL) * M * N
    DV = ReadA + ReadB + ReadD + WriteE

    MemUseA = TM * TK
    MemUseB = TK * TL
    MemUseC = M * TL
    MemUseD = TL * TN
    MemUseAll = (
        MemUseA + MemUseB + MemUseD
    ) * bytePerInputElem + MemUseC * bytePerAccElem

    locality = 1 / DV if MemUseAll < MemByteLimitPerCore else -float("inf")

    parallelism = min(
        1,
        min(PeakBlock, CoreNum * (MemByteLimitPerCore // MemUseAll)),
    )

    recomputation = 0

    return Metric(locality, parallelism, recomputation)


def main():
    M = 512
    N = 64
    K = 64
    L = 512
    bytePerInputElem = 2
    bytePerAccElem = 4
    PeakBlock = 108 * 32
    CoreNum = 108
    MemByteLimitPerCore = 128 * 1024
    count_x = 0
    locality = [[] for i in range(7)]
    parallelism = [[] for i in range(7)]
    recomputation = [[] for i in range(7)]
    funcs = [cal_F1, cal_F2, cal_F3, cal_F4, cal_F5, cal_F6, cal_F7]
    for TM in [2 ** x for x in range(10)]:
        for TN in [2 ** x for x in range(7)]:
            for TK in [2 ** x for x in range(7)]:
                for TL in [2 ** x for x in range(10)]:
                    count_x += 1
                    for i in range(7):
                        f = funcs[i](
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
                        )
                        locality[i].append(f.locality)
                        parallelism[i].append(f.parallelism)
                        recomputation[i].append(f.recomputation)

    return count_x, locality, parallelism, recomputation


def plot_figure(count_x, values, name, show=False, step=1, pdf=False):
    # plot the figure
    xs = np.arange(1, count_x + 1, 1)
    # f1 = plt.plot(xs, values[0], lw=1)
    # f2 = plt.plot(xs, values[1], lw=1)
    # f3 = plt.plot(xs, values[2], lw=1)
    # f4 = plt.plot(xs, values[3], lw=1)
    # f5 = plt.plot(xs, values[4], lw=1)
    # f6 = plt.plot(xs, values[5], lw=1)
    # f7 = plt.plot(xs, values[6], lw=1)

    sizes = [20 for i in range(7)]
    f7 = plt.scatter(
        xs[::step],
        values[6][::step],
        marker="x",
        c="lawngreen",
        s=sizes[6],
        edgecolors="none",
    )

    f3 = plt.scatter(
        xs[::step], values[2][::step], marker="1", s=sizes[2], edgecolors="none"
    )
    f5 = plt.scatter(
        xs[::step], values[4][::step], marker="+", s=sizes[4], edgecolors="none"
    )
    f2 = plt.scatter(
        xs[::step], values[1][::step], marker="o", s=sizes[1], edgecolors="none"
    )
    f1 = plt.scatter(
        xs[::step], values[0][::step], marker="^", s=sizes[0], edgecolors="none"
    )
    f4 = plt.scatter(
        xs[::step], values[3][::step], marker="s", s=sizes[3], edgecolors="none"
    )
    f6 = plt.scatter(
        xs[::step], values[5][::step], marker="*", s=sizes[5], edgecolors="none"
    )

    plt.legend(
        handles=[f1, f2, f3, f4, f5, f6, f7],
        labels=["F1", "F2", "F3", "F4", "F5", "F6", "F7"],
    )

    plt.xlabel("Tiling factor choices", fontsize=14)
    plt.ylabel(f"{name}", fontsize=16)

    if pdf:
        suffix = "pdf"
    else:
        suffix = "png"
    if show:
        plt.show()
    else:
        plt.savefig(f"{name}.{suffix}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--metric", choices=["loc", "par", "rec"])
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--pdf", action="store_true")
    parser.add_argument("--step", type=int, default=1)

    args = parser.parse_args()
    count_x, locality, parallelism, recomputation = main()
    print(count_x)
    if args.metric == "loc":
        plot_figure(
            count_x, locality, "locality", show=args.show, step=args.step, pdf=args.pdf
        )
    elif args.metric == "par":
        plot_figure(
            count_x,
            parallelism,
            "parallelism",
            show=args.show,
            step=args.step,
            pdf=args.pdf,
        )
    elif args.metric == "rec":
        plot_figure(
            count_x,
            recomputation,
            "recomputation",
            show=args.show,
            step=args.step,
            pdf=args.pdf,
        )
