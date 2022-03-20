import pickle as pkl
import tvm
import sys
import numpy as np

def script(path, module):
    repeat = 10000
    container = np.load(path)
    data = [container[key] for key in container]
    dev = tvm.cpu()
    data_tvm = [tvm.nd.array(x, dev) for x in data]
    func = tvm.runtime.load_module(module)
    
    eval = func.time_evaluator(func.entry_name, dev, number = repeat)
    cost = eval(*data_tvm).mean
    print("elapsed time", cost)
    return cost

if __name__ == '__main__':
    cmd = sys.argv
    assert len(cmd) >= 3
    path, module = cmd[1], cmd[2]
    script (path, module)