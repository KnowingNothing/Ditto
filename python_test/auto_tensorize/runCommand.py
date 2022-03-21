import command
import sys

if __name__ == "__main__":
    cmd = " ".join(sys.argv[1:])
    cmd_ = "perf stat -e l1d.replacement,l2_lines_in.all,l2_trans.l1d_wb,l2_trans.l2_wb,cpu-cycles -C 0 taskset -c 0 "
    cmd_ += cmd
    res = command.run(cmd_.split())
    print(str(res.output))
