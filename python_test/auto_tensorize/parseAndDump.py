import regex as re
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
    p = "([0-9,\.]*)[\s]*(l1d\.replacement|l2_lines_in\.all|l2_trans\.l1d_wb|l2_trans\.l2_wb|cpu\-cycles)"
    for line in stdin:
        res = re.findall(p, line)
        for (value, metric) in res:
            value = "".join(value.split(","))
            t = m[metric]
            if t == "C":
                d[t] = float(value) / 1e9 / 2.2 / 10000
            else:
                d[t] = float(value) * 64
    print(d)
    with open("./result/groudTruth.txt", "a") as f:
        data = [d["R1"], d["W1"], d["R2"], d["W2"], d["C"]]
        data = [str(_) for _ in data]
        f.write(" ".join(data))
        f.write("\n")
