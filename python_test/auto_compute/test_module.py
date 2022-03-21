import tvm
from tvm import auto_scheduler as ansor
from ditto import auto_compute as ac
from ditto import auto_schedule
from ditto.auto_compute.nn.module import Conv2d


from collections import OrderedDict


TEST_CASES = OrderedDict()


def register_test(func):
    name = func.__name__
    prefix = "test"
    assert name[: len(prefix)] == prefix
    try:
        number = int(name[len(prefix) :])

        def _inner(*args, **kwargs):
            print(func.__doc__)
            func(*args, **kwargs)

        assert number not in TEST_CASES, "Repeated test case number %d" % number
        TEST_CASES[number] = _inner
    except ValueError as e:
        print(e)
        print("Can't convert to number", name[len(prefix) :])


@register_test
def test1():
    A = ac.layer_tensor([1, 3, 224, 224], name="A", dtype="float16")
    conv_layer = Conv2d(
        3,
        64,
        7,
        stride=2,
        padding=3,
        dilation=1,
        groups=1,
        dtype="float16",
        out_dtype="float32",
        layout="NCHW",
    )
    C = conv_layer(A)
    print(C.layer)


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
        assert args.case in TEST_CASES, "Can't find case %s." % (str(args.case))
        case = TEST_CASES[args.case]
        case()
