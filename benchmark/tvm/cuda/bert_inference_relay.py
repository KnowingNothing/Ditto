from tvm import relay
import tvm
import numpy as np
import argparse
import torch


bmm_chain = True


def multihead_attention(input, prefix, batch, seq_len, hidden, n_heads, dtype):
    """
    input: batch, seqlen, dim
    output: batch, seqlen, dim
    """
    assert hidden % n_heads == 0
    head_dim = hidden // n_heads

    packed = relay.nn.dense(
        input,
        relay.var(prefix + "WQ", dtype=dtype),
        units=hidden * 3,
        out_dtype=dtype,
    )
    packed = relay.reshape(packed, [seq_len, n_heads, head_dim * 3])
    packed = relay.transpose(packed, [1, 0, 2])
    # packed = relay.reshape(packed, [n_heads, seq_len, head_dim * 3])

    # batch n_head seqlen, dim / n_head
    Q, K, V = relay.split(packed, indices_or_sections=3, axis=-1)
    if not bmm_chain:
        QKV = Q + K + V
    else:
        # # batch * n_head seqlen, seqlen
        QK = relay.nn.batch_matmul(Q, K, transpose_b=True)
        # # batch * n_head seqlen, seqlen
        # QK_softmax = relay.nn.softmax(QK, axis=-1)
        QK_softmax = QK
        # # batch * n_head seqlen, dim / n_head
        QKV = relay.nn.batch_matmul(QK_softmax, V, transpose_b=False)
    QKV = relay.reshape(QKV, [n_heads, seq_len, head_dim])
    QKV = relay.transpose(QKV, [1, 0, 2])
    QKV = relay.reshape(QKV, [seq_len, hidden])
    QKV = QKV

    return QKV


def feedforward(input, prefix, batch, seq_len, hidden, n_heads):
    mlp_dim = min(hidden * 4, 3072)
    output = relay.nn.dense(input, relay.var(prefix + "fc1_w"), units=mlp_dim)
    output = relay.nn.bias_add(output, relay.var(prefix + "fc1_bias"), axis=-1)
    output = relay.nn.relu(output)

    output = relay.nn.dense(output, relay.var(prefix + "fc2_w"), units=hidden)
    output = relay.nn.bias_add(output, relay.var(prefix + "fc2_bias"), axis=-1)
    output = relay.nn.relu(output)

    return output


def create_workload(net, seed=0):
    """Helper function to create benchmark image classification workload.

    Parameters
    ----------
    net : tvm.relay.Function
        The selected function of the network.

    seed : int
        The seed used in initialization.

    Returns
    -------
    mod : tvm.IRModule
        The created relay module.

    params : dict of str to NDArray
        The parameters.
    """
    mod = tvm.IRModule.from_expr(net)
    mod = relay.transform.InferType()(mod)
    shape_dict = {v.name_hint: v.checked_type for v in mod["main"].params}
    # print("shape_dict", shape_dict)
    np.random.seed(seed)
    params = {}
    for k, v in shape_dict.items():
        if k == "data":
            continue
        init_value = np.random.uniform(size=v.concrete_shape).astype(v.dtype)
        params[k] = tvm.nd.array(init_value, device=tvm.cuda())
    return mod, params


def transformer(batch, seq_len, hidden, n_heads, n_layer, dtype):
    x = relay.var("input", shape=[seq_len, hidden], dtype=dtype)
    for i in range(n_layer):
        x = multihead_attention(x, f"attn{i}", batch, seq_len, hidden, n_heads, dtype)
        x = feedforward(x, f"ff{i}", batch, seq_len, hidden, n_heads)
    args = relay.analysis.free_vars(x)
    func = relay.Function(args, x)
    return func


def test_relay(batch, seq_len, hidden, n_heads, n_layer, dtype):
    net = transformer(batch, seq_len, hidden, n_heads, n_layer, dtype)
    mod, params = create_workload(net)

    target = "cuda -libs=cublas,cudnn"
    # target = "cuda"
    import tvm.contrib.graph_executor as runtime

    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build_module.build(mod, target=target, params={})
        dev = tvm.device(str(target), 0)
        module = runtime.GraphModule(lib["default"](dev))
    ret = module.benchmark(dev, min_repeat_ms=600)
    print(ret)
    return ret.mean * 1e3


example_text = """
 example:
    python bert_inference_relay.py --dtype float32 --begin 0 --num 1
"""


shapes = [
    # (batch, seq_len, hidden, n_heads, n_layers)
    (1, 1024, 512, 8, 4),  # Transformer-Small
    (1, 1024, 768, 12, 12),  # Transformer-Base
    (1, 1024, 1024, 16, 24),  # Transformer-Large
    (1, 512, 512, 8, 4),  # Bert-Small
    (1, 512, 768, 12, 12),  # Bert-Base
    (1, 512, 1024, 16, 24),  # Bert-Large
    # (1, 256, 768, 12, 12),  # ViT-Base/14
    # (1, 256, 1024, 16, 24),  # ViT-Large/14
    # (1, 256, 1280, 16, 24),  # ViT-Huge/14
    (1, 208, 768, 12, 12),  # ViT-Base/16
    (1, 208, 1024, 16, 24),  # ViT-Large/16
    (1, 208, 1280, 16, 24),  # ViT-Huge/16
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="base_maker",
        description="template maker",
        epilog=example_text,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--only_once", action="store_true")
    parser.add_argument("--enable_cudnn", action="store_true")
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["float32", "float16", "int8"],
        default="float16",
    )
    parser.add_argument(
        "--begin", type=int, choices=list(range(len(shapes))), default=0
    )
    parser.add_argument(
        "--num", type=int, choices=list(range(1, len(shapes) + 1)), default=len(shapes)
    )

    args = parser.parse_args()

    if args.enable_cudnn:
        assert torch.backends.cudnn.is_available()
        torch.backends.cudnn.enabled = True
    else:
        torch.backends.cudnn.enabled = False
    costs = []
    for ss in shapes[args.begin : args.begin + args.num]:
        batch, seq_len, hidden, n_heads, n_layers = ss
        cost = test_relay(batch, seq_len, hidden, n_heads, n_layers, args.dtype)
        costs.append((ss, cost))
    print("batch,seq_len,hidden,n_heads,n_layers,dtype,cost")
    for cc in costs:
        print(
            f"{cc[0][0]},{cc[0][1]},{cc[0][2]},{cc[0][3]},{cc[0][4]},{args.dtype},{cc[1]}"
        )
