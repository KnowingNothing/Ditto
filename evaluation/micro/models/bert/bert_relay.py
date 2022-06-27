from tvm import relay
import tvm
import numpy as np

seq_len = 512
dim = 512
depth = 4
heads = 8
mlp_dim = 512

# seq_len = 512
# dim = 768
# depth = 12
# heads = 12
# mlp_dim = 3072

# seq_len = 512
# dim = 1024
# depth = 24
# heads = 16
# mlp_dim = 3072

seqlen = 512
dim = 768
n_head = 12
mlp_dim = 3072
depth = 4
no_attn = True

in_dtype = "float32"
out_dtype = "float32"

def multihead_attention(input, prefix):
    '''
    input: batch, seqlen, dim
    output: batch, seqlen, dim
    '''
    assert(dim % n_head == 0)
    head_dim = dim // n_head 

    packed = relay.nn.dense(input, relay.var(prefix+"WQ", dtype=in_dtype), units = dim*3, out_dtype = out_dtype)
    packed = relay.reshape(packed, [seqlen, n_head, head_dim * 3])
    packed = relay.transpose(packed, [1,0,2])
    packed = relay.reshape(packed, [n_head, seqlen, head_dim * 3])

    # batch n_head seqlen, dim / n_head
    Q, K, V = relay.split(packed, indices_or_sections = 3, axis = -1)
    # # batch * n_head seqlen, seqlen
    QK = relay.nn.batch_matmul(Q, K, transpose_b= True)
    # # batch * n_head seqlen, seqlen
    QK_softmax = relay.nn.softmax(QK, axis = -1)
    # # batch * n_head seqlen, dim / n_head
    QKV = relay.nn.batch_matmul(QK_softmax, V, transpose_b = False)
    QKV = relay.reshape(QKV, [n_head, seqlen, head_dim])
    QKV = relay.transpose(QKV, [1,0,2])
    QKV = relay.reshape(QKV, [seqlen, dim])
    QKV = QKV + input
    
    return QKV


def feedforward(input, prefix):
    output = relay.nn.dense(input, relay.var(prefix+"fc1_w"), units = mlp_dim)
    output = relay.nn.bias_add(output, relay.var(prefix+"fc1_bias"), axis = -1)
    output = relay.nn.relu(output)
    
    output = relay.nn.dense(output, relay.var(prefix+"fc2_w"), units = dim)
    output = relay.nn.bias_add(output, relay.var(prefix+"fc2_bias"), axis = -1)
    output = relay.nn.relu(output)
    
    return output + input

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
    print("shape_dict", shape_dict)
    np.random.seed(seed)
    params = {}
    for k, v in shape_dict.items():
        if k == "data":
            continue
        init_value = np.random.uniform(size = v.concrete_shape).astype(v.dtype)
        params[k] = tvm.nd.array(init_value, device=tvm.cuda())
    return mod, params


def transformer():
    x = relay.var('input', shape=[seqlen, dim], dtype=in_dtype)
    for i in range(depth):
        if not no_attn:
            x = multihead_attention(x, f"attn{i}")
        x = feedforward(x, f'ff{i}')
    args = relay.analysis.free_vars(x)
    func = relay.Function(args, x)
    return func

def test_relay():
    net = transformer()
    mod, params = create_workload(net)

    target = "cuda -libs=cublas,cudnn"
    import tvm.contrib.graph_executor as runtime
    with tvm.transform.PassContext(opt_level = 3):
        lib = relay.build_module.build(mod, target = target, params = {})
        dev = tvm.device(str(target), 0)
        module = runtime.GraphModule(lib['default'](dev))
    ret = module.benchmark(dev, min_repeat_ms = 600)
    print(ret)
    return ret.mean


if __name__ == "__main__":
    time = test_relay()
    print("relay cost: %f s" % time)
