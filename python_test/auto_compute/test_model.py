import tvm
from tvm import auto_scheduler as ansor
from ditto import auto_compute as ac
from ditto import auto_schedule
from ditto.auto_compute.nn.model import resnet50


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


@register_test
def test1():
    A = ac.layer_tensor([1, 3, 224, 224], name="A", dtype="float16")
    ResNet50 = resnet50()
    outputs = ResNet50(A)
    print(outputs.layer)
    graph = ac.graph(A, outputs)
    print(graph)
    
    
@register_test
def test2():
    string = ("{\"nodes\": [{\"op\": \"null\", \"name\": \"A\", \"inputs\": []}, "
      + "{\"op\": \"null\", \"name\": \"B\", \"inputs\": []}, {\"op\": "
      + "\"tvm_op\", \"name\": \"elemwise_add\", \"attrs\": {\"flatten_data\": "
      + "\"1\", \"func_name\": \"elemwise_add\", \"num_inputs\": \"2\", "
      + "\"num_outputs\": \"1\"}, \"inputs\": [[0, 0, 0], [1, 0, 0]]}, {\"op\": "
      + "\"tvm_op\", \"name\": \"__copy_add_to_sub\", \"attrs\": "
      + "{\"flatten_data\": \"0\", \"func_name\": \"__copy\", \"num_inputs\": "
      + "\"1\", \"num_outputs\": \"1\"}, \"inputs\": [[2, 0, 0]]}, {\"op\": "
      + "\"null\", \"name\": \"C\", \"inputs\": []}, {\"op\": \"tvm_op\", "
      + "\"name\": \"elemwise_sub\", \"attrs\": {\"flatten_data\": \"0\", "
      + "\"func_name\": \"elemwise_sub\", \"num_inputs\": \"2\", "
      + "\"num_outputs\": \"1\"}, \"inputs\": [[3, 0, 0], [4, 0, 0]]}], "
      + "\"arg_nodes\": [0, 1, 4], \"node_row_ptr\": [0, 1, 2, 3, 4, 5, 6], "
      + "\"heads\": [[5, 0, 0]], \"attrs\": {\"storage_id\": [\"list_int\", [3, "
      + "4, 0, 1, 5, 2]], \"shape\": [\"list_shape\", [[4], [4], [4], [4], [4], "
      + "[4]]], \"device_index\": [\"list_int\", [2, 2, 2, 1, 1, 1]], \"dtype\": "
      + "[\"list_int\", [0, 0, 0, 0, 0, 0]], \"dltype\": [\"list_str\", "
      + "[\"float32\", \"float32\", \"float32\", \"float32\", \"float32\", "
      + "\"float32\"]]}}")
    print(string)
    


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
