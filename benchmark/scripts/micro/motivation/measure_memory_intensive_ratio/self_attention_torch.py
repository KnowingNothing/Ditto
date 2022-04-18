import math
import json
from typing import NamedTuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autocast


# global mode: profile whole net or several layers
MODE = "whole-net"
# "whole-net", "gemm-chain"
PROFILE_LIST = []  # used to store time cost of layers
TENSOR_CORE = True


def split_last(x, shape):
    "split the last dimension to given shape"
    shape = list(shape)
    assert shape.count(-1) <= 1
    if -1 in shape:
        shape[shape.index(-1)] = int(x.size(-1) / -np.prod(shape))
    return x.view(*x.size()[:-1], *shape)


def merge_last(x, n_dims):
    "merge the last n_dims to a dimension"
    s = x.size()
    assert n_dims > 1 and n_dims < len(s)
    return x.view(*s[:-n_dims], -1)


class Config(NamedTuple):
    hidden: int = 768  # Dimension of Hidden Layer in Transformer Encoder
    n_heads: int = 768 // 64  # Numher of Heads in Multi-Headed Attention Layers
    # activ_fn: str = "gelu" # Non-linear Activation Function Type in Hidden Layers
    seq_len: int = 512

    @classmethod
    def from_dict(cls, d):
        return cls(**d)


class MultiHeadedSelfAttention(nn.Module):
    """Multi-Headed Dot Product Attention"""

    def __init__(self, cfg):
        super().__init__()
        self.proj_qkv = nn.Linear(cfg.hidden, cfg.hidden * 3)
        # self.proj_k = nn.Linear(cfg.hidden, cfg.hidden)
        # self.proj_v = nn.Linear(cfg.hidden, cfg.hidden)
        # self.drop = nn.Dropout(cfg.p_drop_attn)
        self.scores = None  # for visualization
        self.n_heads = cfg.n_heads

    def forward(self, x, mask=None):
        """
        x, q(query), k(key), v(value) : (B(batch_size), S(seq_len), D(dim))
        mask : (B(batch_size) x S(seq_len))
        * split D(dim) into (H(n_heads), W(width of head)) ; D = H * W
        """
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        # q, k, v = self.proj_q(x), self.proj_k(x), self.proj_v(x)
        qkv = self.proj_qkv(x)
        q, k, v = torch.tensor_split(qkv, 3, dim=-1)

        q, k, v = (split_last(x, (self.n_heads, -1)).transpose(1, 2) for x in [q, k, v])
        # (B, H, S, W) @ (B, H, W, S) -> (B, H, S, S) -softmax-> (B, H, S, S)
        if MODE == "gemm-chain":
            global PROFILE_LIST
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
        scores = q @ k.transpose(-2, -1) / np.sqrt(k.size(-1))
        if mask is not None:
            mask = mask[:, None, None, :].float()
            scores -= 10000.0 * (1.0 - mask)
        # scores = self.drop(F.softmax(scores, dim=-1))
        scores = F.softmax(scores, dim=-1)
        # (B, H, S, S) @ (B, H, S, W) -> (B, H, S, W) -trans-> (B, S, H, W)
        h = (scores @ v).transpose(1, 2).contiguous()
        if MODE == "gemm-chain":
            end.record()
            torch.cuda.synchronize()
            total = start.elapsed_time(end)
            PROFILE_LIST.append(total)
        # -merge-> (B, S, D)
        h = merge_last(h, 2)
        self.scores = scores
        return h


def perf_whole_net(config, batch_size, seq_len, verbose=False):
    global MODE
    MODE = "whole-net"
    cfg = Config.from_dict(config)
    model = MultiHeadedSelfAttention(cfg)
    model = model.cuda().eval()

    # Indices of input sequence tokens in the vocabulary. Indices can be obtained using Tokenizer
    if TENSOR_CORE:
        input_ids = torch.zeros([batch_size, seq_len, cfg.hidden], dtype=torch.float16)
    else:
        input_ids = torch.zeros([batch_size, seq_len, cfg.hidden], dtype=torch.float32)
    input_ids = input_ids.cuda()

    # output: BaseModelOutputWithPoolingAndCrossAttentions, [bs, seq-len, hid-dim]
    with torch.no_grad():
        if TENSOR_CORE:
            with autocast("cuda"):
                model(input_ids)
        else:
            model(input_ids)

    number = 100
    repeats = 1

    for i in range(repeats):
        time_record = []
        for j in range(number):
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()

            with torch.no_grad():
                if TENSOR_CORE:
                    with autocast("cuda"):
                        model(input_ids)
                else:
                    model(input_ids)

            end.record()
            torch.cuda.synchronize()
            total = start.elapsed_time(end)
            time_record.append(total)
        if i == repeats - 1:
            if verbose:
                print("Average Albert latency", np.mean(time_record), "ms")
                print("Median  Albert latency", np.median(time_record), "ms")
            return np.mean(time_record)


def perf_gemm_chain(config, batch_size, seq_len, verbose=False):
    global MODE
    MODE = "gemm-chain"
    cfg = Config.from_dict(config)
    model = MultiHeadedSelfAttention(cfg)
    model = model.cuda().eval()

    # Indices of input sequence tokens in the vocabulary. Indices can be obtained using Tokenizer
    if TENSOR_CORE:
        input_ids = torch.zeros([batch_size, seq_len, cfg.hidden], dtype=torch.float16)
    else:
        input_ids = torch.zeros([batch_size, seq_len, cfg.hidden], dtype=torch.float32)
    input_ids = input_ids.cuda()

    # output: BaseModelOutputWithPoolingAndCrossAttentions, [bs, seq-len, hid-dim]
    with torch.no_grad():
        if TENSOR_CORE:
            with autocast("cuda"):
                model(input_ids)
        else:
            model(input_ids)

    number = 100
    repeats = 1

    for i in range(repeats):
        time_record = []
        gemm_chain_record = []
        for j in range(number):
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()

            with torch.no_grad():
                if TENSOR_CORE:
                    with autocast("cuda"):
                        model(input_ids)
                else:
                    model(input_ids)

            end.record()
            torch.cuda.synchronize()
            total = start.elapsed_time(end)
            time_record.append(total)
            global PROFILE_LIST
            gemm_chain_cost = [x for x in PROFILE_LIST]
            PROFILE_LIST.clear()
            gemm_chain_record.append(sum(gemm_chain_cost))
        if i == repeats - 1:
            net_latency = np.mean(time_record)
            gemm_chain_latency = np.mean(gemm_chain_record)
            if verbose:
                # print("Average Albert latency", net_latency, "ms")
                print("Average Gemm-Chain latency", gemm_chain_latency, "ms")
                # print("Ratio:", gemm_chain_latency/net_latency * 100, "%")
            return gemm_chain_latency


configs = [
    {
        "hidden": 512,
        "n_heads": 8,
        "seq_len": 512,
    },
    {
        "hidden": 768,
        "n_heads": 12,
        "seq_len": 512,
    },
    {
        "hidden": 1024,
        "n_heads": 16,
        "seq_len": 512,
    },
    {
        "hidden": 768,
        "n_heads": 12,
        "seq_len": 196,
    },
    {
        "hidden": 1024,
        "n_heads": 16,
        "seq_len": 196,
    },
    {
        "hidden": 1280,
        "n_heads": 16,
        "seq_len": 196,
    },
]

if __name__ == "__main__":
    batch_size = 1
    print("Use PyTorch Tensor Core:", TENSOR_CORE)
    print("Ratio,Time,AI,FusedAI,Perf,OrgDRAM,FuseDRAM")
    record = []
    for config in configs:
        seq_len = config["seq_len"]
        hidden = config["hidden"]
        heads = config["n_heads"]
        net_cost = perf_whole_net(config, batch_size, seq_len)
        chain_cost = perf_gemm_chain(config, batch_size, seq_len)
        ratio = chain_cost / net_cost * 100
        gflop = (
            batch_size
            * (
                (seq_len * hidden * seq_len + seq_len * seq_len * hidden) * 2
                + (seq_len * seq_len * heads) * 3
            )
            / 1e9
        )
        first_gemm_dram = batch_size * (
            seq_len * hidden + hidden * seq_len + seq_len * seq_len * heads
        )
        softmax_dram = batch_size * (seq_len * seq_len * heads * 2)
        second_gemm_dram = batch_size * (
            seq_len * seq_len * heads + seq_len * hidden + seq_len * hidden
        )
        fuse_dram = batch_size * (
            seq_len * hidden + hidden * seq_len + seq_len * hidden + seq_len * hidden
        )
        if TENSOR_CORE:
            dram = (first_gemm_dram + softmax_dram + second_gemm_dram) * 2 / 1e9
            fuse_dram *= 2 / 1e9
        else:
            dram = (first_gemm_dram + softmax_dram + second_gemm_dram) * 4 / 1e9
            fuse_dram *= 4 / 1e9
        ai = gflop / dram
        fuse_ai = gflop / fuse_dram
        perf = gflop / chain_cost * 1e3
        print(f"{ratio},{chain_cost},{ai},{fuse_ai},{perf},{dram},{fuse_dram}")
        # record.append((seq_len * seq_len)/(seq_len + seq_len))
        record.append((seq_len * hidden // heads)/(seq_len + hidden//heads))
    print(min(record))
    print(max(record))
