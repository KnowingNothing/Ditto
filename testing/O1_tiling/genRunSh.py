import argparse 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("M", type=int)
    parser.add_argument("N", type=int)
    parser.add_argument("K", type=int)
    parser.add_argument("CACHE_SIZE", type = int)
    parser.add_argument("--factor", type = float, default = 0.5, required = False)
    parser.add_argument("--m", type = int, default = -1, required = False)
    parser.add_argument("--n", type = int, default = -1, required = False)
    parser.add_argument("--k", type = int, default = -1, required = False)

    args = parser.parse_args()
    
    tm_r = [args.m] if args.m > 0 else range(1, args.M+1)
    tn_r = [args.n] if args.n > 0 else range(1, args.N+1)
    tk_r = [args.k] if args.k > 0 else range(1, args.K+1)

    candidates = []

    for tm in tm_r:
        for tn in tn_r:
            for tk in tk_r:
                mem_use = tm * tn + tn * tk + tm * tk
                if mem_use >= args.CACHE_SIZE * args.factor and mem_use <= args.CACHE_SIZE:
                    candidates.append((tm, tn, tk))
    
    print(f"{len(candidates)} candidates.")
    
    insts = "#!/bin/bash\n"
    for candidate in candidates:
        inst = f"perf stat -o ./result/{args.M}_{args.N}_{args.K}_{candidate[0]}_{candidate[1]}_{candidate[2]}.txt -e l1d.replacement ./gemm {args.M} {args.N} {args.K} {candidate[0]} {candidate[1]} {candidate[2]}"
        insts += inst + "\n"
    
    filename = f"run_{args.M}_{args.N}_{args.K}_{args.m}_{args.n}_{args.k}.sh"
    
    print ("written into ", filename)

    with open(filename, 'w') as f:
        f.write(insts)
