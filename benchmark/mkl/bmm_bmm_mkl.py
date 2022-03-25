import subprocess
import argparse
import regex as re
REPEAT = 10000
peakflops = {'sc': 704, 'sccc': 2150.4, 'scccc': 2995.2}
def main(batch, M, N, K, L, server):
    args = [batch, M, N, K, L, REPEAT, peakflops[server]]
    args = [str(_) for _ in args]
    args = ' '.join(args)
    cmd = "./MKL2MM " + args
    s = subprocess.check_output(cmd.split()).decode('utf-8')
    ratioToPeak = re.findall('ratioToPeak: ([\d\.]*)', s)
    time = re.findall('time: ([\d\.]*)', s)
    return float(time[0]), float(ratioToPeak[0])

shapes = [
    # (batch, M, N, K, L)
    (8, 512, 512 // 8, 512 // 8, 512),  # Bert-Small
    (12, 512, 768 // 12, 768 // 12, 512),  # Bert-Base
    (16, 512, 1024 // 16, 1024 // 16, 512),  # Bert-Large
    (12, 256, 768 // 12, 768 // 12, 256),  # ViT-Base/14
    (16, 256, 1024 // 16, 1024 // 16, 256),  # ViT-Large/14
    (16, 256, 1280 // 16, 1280 // 16, 256),  # ViT-Huge/14
]
example_text = "python ./bmm_bmm_cpu.py --dtype float32 --begin 0 --num 1 --server sc"
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
        "--server", type=str, choices=['sc', 'sccc', 'scccc'], default='sccc'
    )

    args = parser.parse_args()

    subprocess.Popen(["make"]).wait()

    costs = []
    for ss in shapes[args.begin: args.begin + args.num]:
        B, M, N, K, L = ss
        cost = main(
            batch=B,
            M=M,
            N=N,
            K=K,
            L=L,
            server = args.server
        )
        costs.append((ss, cost))

    print("B,M,N,K,L,server,cost")
    for cc in costs:
        print(
            f"{cc[0][0]},{cc[0][1]},{cc[0][2]},{cc[0][3]},{cc[0][4]},{args.server},{cc[1]}"
        )
    for cc in costs:
        print(cc[1])