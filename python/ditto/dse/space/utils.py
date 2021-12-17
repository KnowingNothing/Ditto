"""Utils for DSE space construction."""
import math


def get_factor_lst(value):
    assert isinstance(value, int)
    ret = []
    end = math.sqrt(value)
    for i in range(1, math.ceil(end)):
        if value % i == 0:
            ret.append(i)
            ret.append(value // i)
    if end - int(end) < 1e-10 and value % int(end) == 0:
        ret.append(int(end))

    return ret


def powerx_lst(x, left, right):
    ret = []
    beg = 1
    while beg < left:
        beg *= x
    while beg < right:
        ret.append(beg)
        beg = beg * x
    return ret


def any_factor_split(value, number, allow_non_divisible="off"):
    assert allow_non_divisible in ["off", "power2", "continuous"]
    ret = []
    assert isinstance(number, int)
    recursive_factor_split(value, [], number, ret, allow_non_divisible)
    return ret


def recursive_factor_split(left, cur, number, ret, policy):
    if number == 1:
        ret.append(cur + [left])
        return
    if policy == "power2":
        f_lst = get_factor_lst(left)
        f_lst.extend(powerx_lst(2, 1, left))
        f_lst = list(set(f_lst))
    elif policy == "continuous":
        f_lst = list(range(1, left + 1))
    else:
        f_lst = get_factor_lst(left)
        f_lst = sorted(f_lst)
    for f in f_lst:
        recursive_factor_split(left // f, cur + [f], number - 1, ret, policy)


def remap_factors(factor_lst):
    assert isinstance(factor_lst, (list, tuple))
    assert len(factor_lst) > 0
    sample = factor_lst[0]
    assert isinstance(sample, (list, tuple))
    assert len(sample) > 0
    dim = len(sample) - 1
    number_count = {i: set() for i in range(dim + 1)}
    # check the factor list
    for v in factor_lst:
        assert isinstance(v, (list, tuple))
        assert len(v) == dim + 1, dim
        for i, ele in enumerate(v):
            number_count[i].add(ele)
    num_factors = len(number_count[0])
    for k, v in number_count.items():
        assert len(v) == num_factors
    # remap the factor list
    sorted_factors = sorted(number_count[0])
    factor_map = {x: i for i, x in enumerate(sorted_factors)}
    reverse_map = {i: x for i, x in enumerate(sorted_factors)}
    ret = list(map(lambda factors: [factor_map[x]
               for x in factors], factor_lst))
    return ret, reverse_map, dim, num_factors - 1
