import taso as ts
import argparse
import onnx


def main(B, M, N, K, L, only_once=False):
    graph = ts.new_graph()
    q = graph.new_input(dims=(B, M, K))
    k = graph.new_input(dims=(B, K, L))
    v = graph.new_input(dims=(B, L, N))
    logits = graph.matmul(q, k)
    # TASO doesn't allow us to insert elem-wise op
    # between bmm and bmm
    # exp = graph.relu(input=logits)
    # sumv = graph.reduce_sum(input=exp, axes=(2,), keepdims=True)
    # div = graph.div(x=exp, y=sumv)
    output = graph.matmul(logits, v)

    new_graph = ts.optimize(graph, alpha=1.0, budget=100)
    cost = new_graph.run_time()
    return cost
    # onnx_model = ts.export_onnx(new_graph)
    # onnx.save(onnx_model, f"bmm_exp_bmm_{B}-{M}-{N}-{K}-{L}.onnx")


example_text = """
 example:
    python bmm_bmm.py --begin 0 --num 1
"""

shapes = [
    # (batch, M, N, K, L)
    (8, 512, 512 // 8, 512 // 8, 512),  # Bert-Small
    (12, 512, 768 // 12, 768 // 12, 512),  # Bert-Base
    (16, 512, 1024 // 16, 1024 // 16, 512),  # Bert-Large
    (12, 256, 768 // 12, 768 // 12, 256),  # ViT-Base/14
    (16, 256, 1024 // 16, 1024 // 16, 256),  # ViT-Large/14
    (16, 256, 1280 // 16, 1280 // 16, 256),  # ViT-Huge/14
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="base_maker",
        description="template maker",
        epilog=example_text,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--only_once", action="store_true")
    parser.add_argument(
        "--begin", type=int, choices=list(range(len(shapes))), default=0
    )
    parser.add_argument(
        "--num", type=int, choices=list(range(1, len(shapes) + 1)), default=len(shapes)
    )

    args = parser.parse_args()
    costs = []
    for ss in shapes[args.begin : args.begin + args.num]:
        B, M, N, K, L = ss
        cost = main(B, M, N, K, L, args.only_once)
        costs.append((ss, cost))
    print("B,M,N,K,L,cost")
    for cc in costs:
        print(f"{cc[0][0]},{cc[0][1]},{cc[0][2]},{cc[0][3]},{cc[0][4]},{cc[1]}")
