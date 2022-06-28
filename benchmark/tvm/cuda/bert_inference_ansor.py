from tvm import relay
import tvm
import numpy as np
from tvm import auto_scheduler
from tvm.contrib import graph_executor 

# seq_len = 512
# dim = 512
# depth = 4
# heads = 8
# mlp_dim = 512

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
depth = 12
bmm_chain = True

in_dtype = "float32"
out_dtype = "float32"

logfile = f"ansor_bert_seqlen{seqlen}_dim_{dim}_nhead{n_head}_mlpdim{mlp_dim}_depth{depth}_indtype{in_dtype}_outdtype{out_dtype}.log"
test = True


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
    if bmm_chain:
        # # batch * n_head seqlen, seqlen
        QK = relay.nn.batch_matmul(Q, K, transpose_b= True)
        # # batch * n_head seqlen, seqlen
        QK_softmax = relay.nn.softmax(QK, axis = -1)
        # # batch * n_head seqlen, dim / n_head
        QKV = relay.nn.batch_matmul(QK_softmax, V, transpose_b = False)
    else:
        QKV = Q + K + V
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
    print("mod['main'].ret_type", mod['main'].ret_type.shape)
    np.random.seed(seed)
    params = {}
    for k, v in shape_dict.items():
        if k == "data":
            continue
        init_value = np.random.uniform(size = v.concrete_shape).astype(v.dtype)
        params[k] = tvm.nd.array(init_value, device=tvm.cuda())
    output = tvm.nd.array(np.zeros(shape = [int(_) for _ in mod['main'].ret_type.shape]).astype(in_dtype), device = tvm.cuda())
    return mod, params, output


def transformer():
    x = relay.var('input', shape=[seqlen, dim], dtype=in_dtype)
    for i in range(depth):
        x = multihead_attention(x, f"attn{i}")
        x = feedforward(x, f'ff{i}')
    args = relay.analysis.free_vars(x)
    func = relay.Function(args, x)
    return func

def ansor_tune():
    net = transformer()
    mod, params, output = create_workload(net)
    target = "cuda"
    tasks, task_weights = auto_scheduler.extract_tasks(
        mod,
        {},
        target
    )
    print ("tasks", tasks)

    for idx, task in enumerate(tasks):
        print("============== Task %d (workload key %s) =======" % (idx, task.workload_key)) 
        print(task.compute_dag)
    
    if not test:
        print("begin tuning ...")

        measure_ctx = auto_scheduler.LocalRPCMeasureContext(repeat = 1, min_repeat_ms = 300, timeout = 10)
        tuner = auto_scheduler.TaskScheduler(tasks, task_weights)
        tune_option = auto_scheduler.TuningOptions(
            num_measure_trials = 5000,
            runner = measure_ctx.runner,
            measure_callbacks = [auto_scheduler.RecordToFile(logfile)],
        )

        tuner.tune(tune_option)
    return mod, params, output

def test_ansor():
    print("Compile...")
    target = "cuda"
    mod, params, output = ansor_tune()
    with auto_scheduler.ApplyHistoryBest(logfile):
        with tvm.transform.PassContext(opt_level = 3, config = {"relay.backend.use_auto_scheduler": True}):
            lib = relay.build(mod, target = target, params = {})
    dev = tvm.device(str(target), 0)
    module = graph_executor.GraphModule(lib['default'](dev))

    print ("Evaluation..")
    ret = module.benchmark(dev, min_repeat_ms = 600)
    print(ret)
    return ret.mean

if __name__ == "__main__":
    time = test_ansor()
    print("ansor cost: ", time)