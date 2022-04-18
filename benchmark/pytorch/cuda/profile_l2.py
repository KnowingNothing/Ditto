import subprocess

hit_rate = []

for case in range(12):
    p = subprocess.Popen(
        [
            "ncu",
            "--metrics",
            "lts__t_sector_hit_rate.pct",
            # "--metrics",
            # "dram__bytes_write.sum",
            "--print-summary",
            "per-gpu",
            "--csv",
            "python",
            "bmm_bmm_cuda.py",
            "--dtype",
            "float16",
            "--begin",
            f"{case}",
            "--num",
            "1",
            "--only_once"
        ],
        shell=False,
        stdout=subprocess.PIPE,
    )
    out, err = p.communicate()
    message = out.decode("utf-8")
    lines = message.split("\n")

    def helper(line_id, the_lines):
        res = []
        for line in the_lines[line_id+2:]:
            if line:
                elements = line.split(",")
                data = elements[-1][1:-1]
                data = float(data)
                res.append(data)
        return res

    for i, line in enumerate(lines):
        if line.startswith("==PROF== Disconnected"):
            res = helper(i, lines)
            hit_rate.append(res)
            break
        
for rate in hit_rate:
    print(",".join(map(str, rate)))
