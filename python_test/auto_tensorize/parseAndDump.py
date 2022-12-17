import regex as re
import numpy as np
from sys import stdin

if __name__ == "__main__":

    d = {}
    m = {
        "l1d.replacement": "R1",
        "l2_lines_in.all": "R2",
        "l2_trans.l1d_wb": "W1",
        "l2_trans.l2_wb": "W2",
        "cpu-cycles": "C",
    }
    # p = "([0-9,\.]*)[\s]*(l1d\.replacement|l2_lines_in\.all|l2_trans\.l1d_wb|l2_trans\.l2_wb|cpu\-cycles)"
    p = "S\d-D\d-C[\d]*[\s]*1[\s]*([\d,]*)[\s]*(l1d\.replacement|l2_lines_in\.all|l2_trans\.l1d_wb|l2_trans\.l2_wb|cpu\-cycles)"
    for line in stdin:
        res = re.findall(p, line)
        for (value, metric) in res:
            value = "".join(value.split(","))
            t = m[metric]
            if t not in d:
                d[t] = []
            if t == "C":
                v = float(value) / 1e9 / 2.2 / 10000
            else:
                v = float(value) * 64
            d[t].append(v)
    data = {}
    for t in d:
        a = np.array(d[t])
        data[t] = (float(a.mean), float(a.std))
    with open("./result/groundTruth.txt", "a") as f:
        data = [d["R1"][0], d["R1"][1], d["W1"][0], d["W1"][1], d["R2"][0], d['R2'][1], d["W2"][0], d["W2"][1], d["C"][0], d["C"][1]]
        data = [str(_) for _ in data]
        f.write(" ".join(data))
        f.write("\n")