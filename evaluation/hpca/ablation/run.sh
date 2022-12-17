#!/bin/bash
python ./bmm_bmm_cpu.py --server scccc --search_type nofuse --mode survey --mk_type mkl --topk 100 --output ./result/bmm_bmm_noMFC
python ./bmm_bmm_cpu.py --server scccc --search_type nofuse --mode best --mk_type mkl --topk 5 --output ./result/bmm_bmm_noMF
python ./bmm_bmm_cpu.py --server scccc --search_type stochastic --mode survey --mk_type mkl --topk 100 --output ./result/bmm_bmm_noMC
python ./bmm_bmm_cpu.py --server scccc --search_type normal --mode best --mk_type mkl --topk 5 --output ./result/bmm_bmm_noM
python ./bmm_bmm_cpu.py --server scccc --search_type nofuse --mode survey  --topk 100 --output ./result/bmm_bmm_noFC
python ./bmm_bmm_cpu.py --server scccc --search_type nofuse --mode best  --topk 5 --output ./result/bmm_bmm_noF
python ./bmm_bmm_cpu.py --server scccc --search_type stochastic --mode survey  --topk 100 --output ./result/bmm_bmm_noC
python ./bmm_bmm_cpu.py --server scccc --search_type normal --mode best  --topk 5 --output ./result/bmm_bmm_chimera