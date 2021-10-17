import math
import numpy as np
from itertools import permutations, product
from functools import reduce

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


def any_factor_split(value, number, allow_non_divisible="off"):
    assert allow_non_divisible in ["off", "power2", "continuous"]
    ret = []
    assert isinstance(number, int)
    recursive_factor_split(value, [], number, ret, allow_non_divisible)
    return ret