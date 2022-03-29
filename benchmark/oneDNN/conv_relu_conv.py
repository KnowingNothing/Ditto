import subprocess
import argparse
import regex as re
peakflops = {'sc': 704, 'sccc': 2150, 'scccc': 2995}
def main(shape, server):
    args = ["cpu"] + shape + [peakflops[server]]
    args = [str(_) for _ in args]
    args = ' '.join(args)
    cmd = "./conv_relu_conv_f32 " + args
    print(cmd)
    s = subprocess.check_output(cmd.split()).decode('utf-8')
    s = s.replace("\n", ' ')
    print(s)
    ratioToPeak = re.findall('ratioToPeak: ([\d\.]*)', s)
    time = re.findall('time\(s\): ([\d\.]*)', s)
    return float(time[0]), float(ratioToPeak[0])

shapes = [
    # (batch, C0, H1, W1, C1, R1, S1, C2, R2, S2)
    [1, 128, 390, 400, 64, 3, 3, 64, 3, 3],  # resnet
    [1, 64, 285, 288, 128, 3, 3, 128, 3, 3],# u_net
    [1, 128, 141, 144, 256, 3, 3, 256, 3, 3],
    [1, 256, 69, 80, 512, 3, 3, 512, 3, 3],
    [1, 512, 33, 32, 1024, 3, 3, 1024, 3, 3],
    [1, 1024, 57, 64, 512, 3, 3, 512, 3, 3],
    [1, 256, 201, 208, 128, 3, 3, 128, 3, 3],
    [1, 128, 393, 400, 64, 3, 3, 64, 3, 3]
]

example_text = "python ./conv_relu_conv.py --dtype float32 --begin 0 --num 1 --server sc"
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
        "--server", type=str, choices=['sc', 'sccc', 'scccc'], default='sccc'
    )

    args = parser.parse_args()

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