import subprocess

utilization = []

for case in range(12):
    p = subprocess.Popen(
        [
            "ncu",
            "--metrics",
            "l1tex__data_pipe_lsu_wavefronts_mem_shared.avg.pct_of_peak_sustained_elapsed",
            # "--metrics",
            # "dram__bytes_write.sum",
            "--print-summary",
            "per-gpu",
            "--csv",
            "python",
            "bmm_bmm_cuda.py",
            "--in_dtype",
            "float16",
            "--acc_dtype",
            "float32",
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
            utilization.append(res)
            break
        
for util in utilization:
    print(",".join(map(str, util)))
