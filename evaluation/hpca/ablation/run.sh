#!/bin/bash

python ./bmm_bmm_cpu.py --store --server scccc --mode best --search_type normal --output bmm_bmm_chimera.pkl

python ./bmm_bmm_cpu.py --store --server scccc --mode best --search_type nofuse --output bmm_bmm_chimera_noF.pkl

python ./bmm_bmm_cpu.py --store --server scccc --mode best --search_type normal --output bmm_bmm_chimera_noM.pkl

python ./bmm_bmm_cpu.py --store --server scccc --mode best --search_type nofuse --output bmm_bmm_baseline.pkl 
