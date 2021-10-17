import tvm
from tvm import auto_scheduler as ansor
from ditto import auto_compute as ac
from ditto import auto_schedule
from ditto.auto_compute.nn.functional import conv2d, channel_scale_nchw, ReLU


from collections import OrderedDict


TEST_CASES = OrderedDict()


def register_test(func):
    name = func.__name__
    prefix = "test"
    assert name[:len(prefix)] == prefix
    try:
        number = int(name[len(prefix):])

        def _inner(*args, **kwargs):
            print(func.__doc__)
            func(*args, **kwargs)
        assert number not in TEST_CASES, "Repeated test case number %d" % number
        TEST_CASES[number] = _inner
    except ValueError as e:
        print(e)
        print("Can't convert to number", name[len(prefix):])


def user_input():
    # a layer
    # conv->bn->relu->conv->bn->relu->conv->relu
    N = 12
    C = 256
    H = 56
    W = 56
    K = 256
    R = 3
    S = 3
    padding = 1
    stride = 1
    dilation = 1

    A = tvm.te.placeholder([N, C, H, W], dtype="float32", name="A")
    B = tvm.te.placeholder([K, C, R, S], dtype="float32", name="B")
    C = tvm.te.placeholder([K, K, R, S], dtype="float32", name="C")
    D = tvm.te.placeholder([K, K, R, S], dtype="float32", name="D")

    alpha1 = tvm.te.placeholder([K], dtype="float32", name="alpha1")
    beta1 = tvm.te.placeholder([K], dtype="float32", name="beta1")

    alpha2 = tvm.te.placeholder([K], dtype="float32", name="alpha2")
    beta2 = tvm.te.placeholder([K], dtype="float32", name="beta2")

    conv1 = conv2d(A, B, stride, padding, dilation,
                   layout="NCHW", out_dtype="float32")
    scale1 = channel_scale_nchw(conv1, alpha1, beta1)
    relu1 = ReLU(scale1)

    conv2 = conv2d(relu1, C, stride, padding, dilation,
                   layout="NCHW", out_dtype="float32")
    scale2 = channel_scale_nchw(conv2, alpha2, beta2)
    relu2 = ReLU(scale2)

    conv3 = conv2d(relu2, D, stride, padding, dilation,
                   layout="NCHW", out_dtype="float32")
    relu3 = ReLU(conv3)

    return ac.layer(relu3.op, inputs=[A], weights=[B, C, D, alpha1, beta1, alpha2, beta2])


@register_test
def test1():
    layer = user_input()
    layer_state = ac.create_layer_state(layer)
    # print(layer_state.layer.inputs)
    # print(layer_state.layer.weights)
    # print(layer_state.layer.ops)
    #####################################
    # design a new compute
    #####################################
    action = ac.design(layer)
    return
    def compute():
        layer = user_input()
        layer_state = ac.create_layer_state(layer)        
        #####################################
        # get a new compute
        #####################################
        layer_state = ac.auto_compute(layer_state, action)
        
        inputs = []
        for inp in layer_state.layer.inputs:
            tensor = layer_state[inp].op.output(0)
            inputs.append(ac.layer_tensor(
                tensor.shape,
                name=tensor.name,
                dtype=tensor.dtype
            ))

        layer = layer_state.make_compute(inputs)
        outputs = layer.ops
        output_tensors = [op.output(0) for op in outputs]
        all_tensors = [
            *layer.inputs,
            *layer.weights,
            *layer.const_scalars,
            *layer.const_tensors,
            *output_tensors
        ]
        return all_tensors
    
    target = "cuda"
    trials = 100
    task_name = "test"
    log_file = "tmp.log"
    builder = "local"
    runner = "local"
    
    schedule_option = auto_schedule.ScheduleOption(
        target, trials, task_name, log_file, builder, runner
    )
    
    sch, args = auto_schedule.auto_schedule(compute, schedule_option)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", help="test case", type=int, default=1)
    parser.add_argument("--all", help="test all", action="store_true")

    args = parser.parse_args()
    if args.all:
        for k, v in TEST_CASES.items():
            print("############################################")
            print("test", k)
            v()
            print("Pass!")
    else:
        assert args.case in TEST_CASES, "Can't find case %s." % (
            str(args.case))
        case = TEST_CASES[args.case]
        case()
