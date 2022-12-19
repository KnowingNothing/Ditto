import subprocess
import argparse
import regex as re
import pickle as pkl
import os

peakflops = {'sc': 704, 'sccc': 2150, 'scccc': 2995, 'Xeon-Gold-6348': 4659.2}
def main(shape, server):
    args = ["cpu"] + shape + [peakflops[server]]
    args = [str(_) for _ in args]
    args = ' '.join(args)
    cmd = "/home/gulang2022/workspace/Ditto/benchmark/oneDNN/bmm_bmm_f32 " + args
    print(cmd)
    s = subprocess.check_output(cmd.split()).decode('utf-8')
    s = s.replace("\n", ' ')
    print(s)
    ratioToPeak = re.findall('ratioToPeak: ([\d\.]*)', s)
    time = re.findall('time\(s\): ([\d\.]*)', s)
    return float(time[0]), float(ratioToPeak[0])

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

example_text = "python ./bmm_bmm.py --dtype float32 --begin 0 --num 1 --server sc"
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
        choices=["float32"],
        default="float32",
    )
    parser.add_argument(
        "--begin", type=int, choices=list(range(len(shapes))), default=0
    )
    parser.add_argument(
        "--num", type=int, choices=list(range(1, len(shapes) + 1)), default=len(shapes)
    )
    parser.add_argument(
        "--server", type=str, choices=peakflops.keys()
    )

    parser.add_argument(
        "--output", type = str, default = "result"
    )

    args = parser.parse_args()
    
    os.system(f'mkdir -p {args.output}')

    costs = []
    for ss in shapes[args.begin: args.begin + args.num]:
        ss = list(ss)
        cost = main(
            ss,
            server = args.server
        )
        costs.append((ss, cost))

    print("shape,server,cost")
    for cc in costs:
        print(
            f"{cc[0]},{args.server},{cc[1]}"
        )
    with open (f"{args.output}/bmm_bmm-onednn.pkl", 'wb') as f:
        pkl.dump(costs, f)