import subprocess

dram_read = []

for case in range(12):
    p = subprocess.Popen(
        [
            "ncu",
            "--metrics",
            "dram__bytes_read.sum",
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
        sum = 0
        for line in the_lines[line_id+2:]:
            if line:
                elements = line.split(",")
                data = elements[-1][1:-1]
                data = float(data)
                sum += data
        return sum

    for i, line in enumerate(lines):
        if line.startswith("==PROF== Disconnected"):
            read = helper(i, lines)
            dram_read.append(read)
            break
        
for read in dram_read:
    print(read)
