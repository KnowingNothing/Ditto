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
    cmd = "/home/gulang2022/workspace/Ditto/benchmark/oneDNN/conv_conv_f32 " + args
    print(cmd)
    s = subprocess.check_output(cmd.split()).decode('utf-8')
    s = s.replace("\n", ' ')
    print(s)
    ratioToPeak = re.findall('ratioToPeak: ([\d\.]*)', s)
    time = re.findall('time\(s\): ([\d\.]*)', s)
    return float(time[0]), float(ratioToPeak[0])

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

example_text = "python ./conv_conv.py --dtype float32 --begin 0 --num 1 --server sc"
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
    with open (f"{args.output}/conv_conv_oneDNN.pkl", 'wb') as f:
        pkl.dump(costs, f)