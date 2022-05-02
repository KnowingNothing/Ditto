import numpy as np
import regex as re
import pickle as pkl 


def parser_INST(s):
    d = {}
    for ss in s:
        m = {
            "mem_inst_retired.all_loads": "all_loads",
            "mem_load_retired.fb_hit": "fb_hit",
            "mem_load_retired.l1_hit": "l1_hit",
            "mem_load_retired.l1_miss": "l1_miss",
            "mem_load_retired.l2_hit": "l2_hit",
            "mem_load_retired.l2_miss": "l2_miss",
            "mem_load_retired.l3_hit": "l3_hit",
            "mem_load_retired.l3_miss": "l3_miss",
        }
        # p = "([0-9,\.]*)[\s]*(l1d\.replacement|l2_lines_in\.all|l2_trans\.l1d_wb|l2_trans\.l2_wb|cpu\-cycles)"
        p = "([\d,]*)[\s]*(mem_inst_retired\.all_loads|mem_load_retired\.fb_hit|mem_load_retired\.l1_hit|mem_load_retired\.l1_miss|mem_load_retired\.l2_hit|mem_load_retired\.l2_miss|mem_load_retired\.l3_hit|mem_load_retired\.l3_miss)"
        res = re.findall(p, ss)
        for (value, metric) in res:
            value = "".join(value.split(","))
            t = m[metric]
            if t not in d:
                d[t] = []
            d[t] = float(value) 
    d_ = {}
    d_['l1_hit_rate'] = (d['l1_hit']) / (d['l1_hit'] + d['l1_miss'] + d['fb_hit'])
    d_['l2_hit_rate'] = d['l2_hit'] / (d['l2_hit'] + d['l2_miss'])
    d_['l3_hit_rate'] = d['l3_hit'] / (d['l3_hit'] + d['l3_miss'])
    return d_


def parser_DM(s):
    print('parse', s)
    d = {}
    for ss in s:
        m = {
            "mem_inst_retired.all_loads": "all_loads",
            "mem_load_retired.all_stores": "fb_hit",
            "l1d.replacement": "l1_l2",
            "l2_lines_in.all": "l2_l3",
            "offcore_requests.all_data_rd": "L3_DRAM",
        }
        # p = "([0-9,\.]*)[\s]*(l1d\.replacement|l2_lines_in\.all|l2_trans\.l1d_wb|l2_trans\.l2_wb|cpu\-cycles)"
        p = "([\d,]*)[\s]*(mem_inst_retired\.all_loads|mem_load_retired\.all_stores|l1d\.replacement|l2_lines_in\.all|offcore_requests\.all_data_rd)"
        res = re.findall(p, ss)
        for (value, metric) in res:
            print(value, metric)
            value = "".join(value.split(","))
            t = m[metric]
            print(t)
            if t not in d:
                d[t] = []
            d[t] = float(value) 
    return d

def parser_CYCLE(s):
    print('parse', s)
    d = {}
    for ss in s:
        m = {
            "l1d_pend_miss.pending": "l1d_pending",
            "cpu-cycles": "cpu-cycles",
            "cpu-clock": "cpu-clock",
        }
        # p = "([0-9,\.]*)[\s]*(l1d\.replacement|l2_lines_in\.all|l2_trans\.l1d_wb|l2_trans\.l2_wb|cpu\-cycles)"
        p = "([\d,\.]*)[\sa-z]*(l1d_pend_miss\.pending|cpu-cycles|cpu-clock)"
        res = re.findall(p, ss)
        print('-----------begin-----------')
        print(ss)
        print('-----------end---------')
        for (value, metric) in res:
            print("value", value)
            print("metric", metric)
            value = "".join(value.split(","))
            t = m[metric]
            print(t)
            if t not in d:
                d[t] = []
            d[t] = float(value) 
    d['freq(GHz)'] = d['cpu-cycles'] / d['cpu-clock'] / 1e6
    return d

def parser_ALL(s):
    d = {}
    for ss in s:
        m = {
            "l1d_pend_miss.pending": "l1d_pending",
            "cpu-cycles": "cpu-cycles",
            "cpu-clock": "cpu-clock",
            "mem_load_retired.all_stores": "all_stores",
            "l1d.replacement": "l1d.replacement",
            "l2_lines_in.all": "l2_lines_in.all",
            "offcore_requests.all_data_rd": "offcore_requests.all_data_rd",
            "mem_inst_retired.all_loads": "all_loads",
            "mem_load_retired.fb_hit": "fb_hit",
            "mem_load_retired.l1_hit": "l1_hit",
            "mem_load_retired.l1_miss": "l1_miss",
            "mem_load_retired.l2_hit": "l2_hit",
            "mem_load_retired.l2_miss": "l2_miss",
            "mem_load_retired.l3_hit": "l3_hit",
            "mem_load_retired.l3_miss": "l3_miss",
        }
        # p = "([0-9,\.]*)[\s]*(l1d\.replacement|l2_lines_in\.all|l2_trans\.l1d_wb|l2_trans\.l2_wb|cpu\-cycles)"
        p = "([\d,\.]*)[\sa-z]*(l1d_pend_miss\.pending|cpu-cycles|cpu-clock|mem_inst_retired\.all_loads|mem_load_retired\.all_stores|l1d\.replacement|l2_lines_in\.all|offcore_requests\.all_data_rd|mem_load_retired\.fb_hit|mem_load_retired\.l1_hit|mem_load_retired\.l1_miss|mem_load_retired\.l2_hit|mem_load_retired\.l2_miss|mem_load_retired\.l3_hit|mem_load_retired\.l3_miss)"
        res = re.findall(p, ss)
        for (value, metric) in res:
            value = "".join(value.split(","))
            t = m[metric]
            if t not in d:
                d[t] = []
            d[t] = float(value) 
    d['freq(GHz)'] = d['cpu-cycles'] / d['cpu-clock'] / 1e6
    return d



def setGlobals(B, M, N, K, L, dtype):
    global MI, NI1, KI1, NI2, KI2
    MI = 6
    NI1 = 64 if N % 64 == 0 else 32 
    KI1 = 64 if K % 64 == 0 else K
    KI2 = NI1
    NI2 = 64 if L % 64 == 0 else 32
    M = uround(M, MI)
    N = uround(N, NI2)
    L = uround(L, NI1)
    return (B, M, N, K, L)

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

def main():
    apps = ['chimera', 'mkl']
    ret = {}
    for app in apps:
        ret[app] = []
        for iter in range(len(shapes)):
            shape = shapes[iter]
            # filename = app + '_' + str(iter) + '_INST.txt'
            # filename = app + '_' + str(iter) + '_DM.txt'
            # filename = app + '_' + str(iter) + '_CYCLE.txt'
            filename = './res/' + app + '_' + str(iter) + '_ALL.txt'
            with open(filename, 'r') as f:
                data = f.readlines()
            # d = parser_INST(data)
            d = parser_ALL(data)
            # print(d)
            ret[app].append(d)

    return ret 

def dumper(data):
    app = ['chimera', 'mkl']
    feat = data['chimera'][0].keys()
    for a in app:
        for f in feat:
            print()
            print(a, f)
            tmp = [i[f] for i in data[a]]
            for _ in tmp:
                print(_)

if __name__ == "__main__":
    ret = main()
    # with open("hit_rate_profile_INST.pkl", 'wb') as f:
    #     pkl.dump(ret, f)
    # with open("datamove_profile.pkl", 'wb') as f:
    #     pkl.dump(ret, f)
    # with open("cycle_profile.pkl", 'wb') as f:
    #     pkl.dump(ret, f)
    with open("all_profile.pkl", 'wb') as f:
        pkl.dump(ret, f)
    dumper(ret)
    # print(ret)
