#!/usr/bin/bash
python ./bmm_bmm_nomk_fuse.py --mode perf
python ./bmm_bmm_mk_nofuse.py --mode perf
python ./bmm_bmm_nomk_nofuse.py --mode perf
python ./bmm_bmm_mk_fuse.py --mode perf