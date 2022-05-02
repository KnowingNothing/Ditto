import subprocess
import tvm 
import numpy as np
import os
import regex as re
import pickle as pkl
import sys 
import command
import subprocess

application = ['chimera', 'pytorch']

def ceil(x, y):
    return (x + y - 1) // y


def uround(x, y):
    return int(ceil(x, y) * y)


shapes = [
    # (batch, M, N, K, L)
    (8, 512, 512 // 8, 512 // 8, 512),      # Bert-Small
    (12, 512, 768 // 12, 768 // 12, 512),   # Bert-Base
    (16, 512, 1024 // 16, 1024 // 16, 512), # Bert-Large
    (12, 256, 768 // 12, 768 // 12, 256),   # ViT-Base/14
    (16, 256, 1024 // 16, 1024 // 16, 256), # ViT-Large/14
    (16, 256, 1280 // 16, 1280 // 16, 256), # ViT-Huge/14
    (12, uround(196, 16), 768 // 12, 768 // 12, uround(196, 16)),   # ViT-Base/16
    (16, uround(196, 16), 1024 // 16, 1024 // 16, uround(196, 16)), # ViT-Large/16
    (16, uround(196, 16), 1280 // 16, 1280 // 16, uround(196, 16)), # ViT-Huge/16
    (1, 512, uround(49, 16), uround(49, 16), 256),  # Mixer-Small/32-S
    (1, 768, uround(49, 16), uround(49, 16), 384),  # Mixer-Base/32-S
    (1, 1024, uround(49, 16), uround(49, 16), 512), # Mixer-Large/32-S
]

import regex as re
import numpy as np
from sys import stdin

def main(begin, number, metric):
    res = {}
    fusionLevels = []
    subprocess.call(f"rm ./run_{metric}.sh".split())
    subprocess.call(f"rm ./run_likwid_{metric}.sh".split())
    subprocess.call(f"rm ./run_time_{metric}.sh".split())
    for i in range(begin, min(begin + num, len(shapes))):
        shape = shapes[i]   
        res[shape] = {}
        name = f'lib_bmm_bmm_{i}_[\d]*_([\d]*)_([\d]*)\.so'
        dataname = f"./data/data_bmm_bmm_{i}.npy"
        libname = str()
        libnames = []
        for file in os.listdir("./lib"):
            t = re.findall(name, file)
            if len(t):
                # /lib_bmm_bmm_{config['id']}_{iter}_{int(cost*1e6)}_{fusionLevel}
                libnames.append((int(t[0][0]), int(t[0][1]), file))
        print(i, libname)
        fusionLevel, libname = sorted(libnames, key = lambda i: i[0])[0][1:]
        libname = "./lib/" + libname 

        fusionLevels.append((shape, fusionLevel))

        assert libname is not None and dataname is not None 

        print("libname", libname)
        print("dataname", dataname)

        def helper(script, name):
            nonlocal i, res 
            if metric == "inst":
                cmd = f"perf stat -o ./res/{name}_{i}_INST.txt -e mem_inst_retired.all_loads,mem_load_retired.fb_hit,mem_load_retired.l1_hit,mem_load_retired.l1_miss,mem_load_retired.l2_hit,mem_load_retired.l2_miss,mem_load_retired.l3_hit,mem_load_retired.l3_miss,mem_load_retired.local_pmm,offcore_requests.all_data_rd -a "
            elif metric == "dm":
                cmd = f"perf stat -o ./res/{name}_{i}_DM.txt -e mem_inst_retired.all_loads,mem_inst_retired.all_stores,l1d.replacement,l2_trans.l2_wb,l2_lines_in.all,offcore_requests.all_data_rd -a "
            elif metric == "cycle":
                cmd = f"perf stat -o ./res/{name}_{i}_CYCLE.txt -e l1d_pend_miss.pending,cpu-cycles,cpu-clock -a "
            elif metric == "all":
                cmd = f"perf stat -o ./res/{name}_{i}_ALL.txt -e fp_arith_inst_retired.512b_packed_single,inst_retired.any,l1d_pend_miss.pending,cpu-cycles,cpu-clock,mem_inst_retired.all_stores,l1d.replacement,l2_trans.l2_wb,l2_lines_in.all,offcore_requests.all_data_rd,mem_inst_retired.all_loads,mem_load_retired.fb_hit,mem_load_retired.l1_hit,mem_load_retired.l1_miss,mem_load_retired.l2_hit,mem_load_retired.l2_miss,mem_load_retired.l3_hit,mem_load_retired.l3_miss,mem_load_retired.local_pmm,offcore_requests.all_data_rd -a "
            cmd += script
            with open(f"./run_{metric}.sh", 'a') as f:
                f.write(cmd)
                f.write('\n')
            cmd += script
            # s = subprocess.check_output(cmd.split()).decode('utf-8')
            # res[shape][name] = parser(s)
            # print(res[shape][name])
        def likwid_helper(script, name):
            groups = ['CACHES', 'CYCLE_STALLS', 'MEM_SP']
            for group in groups:
                cmd = f"likwid-perfctr -C 0-71 -m -g {group} -o ./likwid_res/{name}_{i}_{group}.csv " + script
                with open(f"./run_likwid_{metric}.sh", 'a') as f:
                    f.write(cmd + "\n")
        def time_helper(script, name):
            cmd = script
            with open(f"./run_time_{metric}.sh", 'a') as f:
                f.write(cmd + "\n")
        B, M, N, K, L = shape
        helper(f"python ./tvm_script.py {B} {M} {N} {K} {L} {libname}", 'Chimera')
        helper(f"python ./bmm_bmm_torch.py {B} {M} {N} {K} {L}", 'Pytorch')
        # time_helper(f"./chimera_cpp {B} {M} {N} {K} {L} {libname} {5000}", 'chimera')
        # helper(f"./chimera_cpp {B} {M} {N} {K} {L} {libname} {5000}", 'chimera')
        # helper(f"./MKL2MM_profile {B} {M} {N} {K} {L} {5000} {2995.2}", 'mkl')
        
        # helper(f"python ./bmm_torch.py {B} {M} {L} {K}", 'pytorch_1')
        # helper(f"python ./bmm_torch.py {B} {M} {N} {L}", 'pytorch_2')
        # helper(f"python ./bmm_bmm_torch.py {B} {M} {N} {K} {L}", 'pytorch')
    with open("fusionLevel.pkl", 'wb') as f:
        pkl.dump(fusionLevels, f)

if __name__ == "__main__":
    begin = 0
    num = len(shapes)
    if len(sys.argv) >= 2:
        begin = int(sys.argv[1])
    if len(sys.argv) >= 3:
        num = int(sys.argv[2])
    # main(begin, num, "cycle")
    # main(begin, num, "inst")
    # main(begin, num, "dm")
    main(begin, num, "all")