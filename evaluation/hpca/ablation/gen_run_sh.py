import argparse


def gen_script(no_M, no_F, no_C):
	mode = 'survey' if no_C else 'best'
	search_type = 'stochastic' if no_C else 'normal'
	search_type = 'nofuse' if no_F else search_type
	topk = '100' if no_C else '5'
	script = "python ./bmm_bmm_cpu.py --server scccc "
	script += f"--search_type {search_type} "
	script += f"--mode {mode} "
	script += f"--mk_type mkl " if no_M else " "
	script += f"--topk {topk} "
	surfix = "no"
	surfix += "M" if no_M else ""
	surfix += "F" if no_F else ""
	surfix += "C" if no_C else ""
	if surfix == "no":
		surfix = "chimera"
	script += f"--output ./result/bmm_bmm_{surfix}"
	return script

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--noM", action="store_true")
	parser.add_argument("--noF", action="store_true")
	parser.add_argument("--noC", action="store_true")
	parser.add_argument("--store", action = "store_true")
	parser.add_argument("--all", action = "store_true")
	
	args = parser.parse_args()
	
	scripts = []
	if args.all:
		for no_M in [True, False]:
			for no_F in [True, False]:
				for no_C in [True, False]:
					scripts.append(gen_script(no_M, no_F, no_C))
	else:
		scripts.append(gen_script(args.no_M, args.no_F, args.no_C))
	
	for script in scripts:
		print(script)
	
	if args.store:
		with open("run.sh", 'w') as f:
			f.write("#!/bin/bash\n")
			f.write('\n'.join(scripts))
