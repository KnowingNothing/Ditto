# reference link: https://github.com/graykode/ALBERT-Pytorch/
"""
    Copyright 2019 Tae Hwan Jung
    ALBERT Implementation with forking
    Clean Pytorch Code from https://github.com/dhlee347/pytorchic-bert
"""

""" Transformer Model Classes & Config Class """

import math
import json
from typing import NamedTuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


""" Utils Functions """


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


# global mode: profile whole net or several layers
MODE = "whole-net"
# "whole-net", "gemm-chain"
PROFILE_LIST = []  # used to store time cost of layers


class Config(NamedTuple):
    "Configuration for BERT model"
    vocab_size: int = None  # Size of Vocabulary
    hidden: int = 768  # Dimension of Hidden Layer in Transformer Encoder
    hidden_ff: int = (
        768 * 4
    )  # Dimension of Intermediate Layers in Positionwise Feedforward Net
    embedding: int = 128  # Factorized embedding parameterization

    n_layers: int = 12  # Numher of Hidden Layers
    n_heads: int = 768 // 64  # Numher of Heads in Multi-Headed Attention Layers
    # activ_fn: str = "gelu" # Non-linear Activation Function Type in Hidden Layers
    max_len: int = 512  # Maximum Length for Positional Embeddings
    n_segments: int = 2  # Number of Sentence Segments

    @classmethod
    def from_json(cls, file):
        return cls(**json.load(open(file, "r")))

    @classmethod
    def from_dict(cls, d):
        return cls(**d)


def gelu(x):
    "Implementation of the gelu activation function by Hugging Face"
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class LayerNorm(nn.Module):
    "A layernorm module in the TF style (epsilon inside the square root)."

    def __init__(self, cfg, variance_epsilon=1e-12):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(cfg.hidden))
        self.beta = nn.Parameter(torch.zeros(cfg.hidden))
        self.variance_epsilon = variance_epsilon

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta


class Embeddings(nn.Module):
    "The embedding module from word, position and token_type embeddings."

    def __init__(self, cfg):
        super().__init__()
        # Original BERT Embedding
        # self.tok_embed = nn.Embedding(cfg.vocab_size, cfg.hidden) # token embedding

        # factorized embedding
        self.tok_embed1 = nn.Embedding(cfg.vocab_size, cfg.embedding)
        self.tok_embed2 = nn.Linear(cfg.embedding, cfg.hidden)

        self.pos_embed = nn.Embedding(cfg.max_len, cfg.hidden)  # position embedding
        self.seg_embed = nn.Embedding(
            cfg.n_segments, cfg.hidden
        )  # segment(token type) embedding

        self.norm = LayerNorm(cfg)
        # self.drop = nn.Dropout(cfg.p_drop_hidden)

    # segment embedding is not used
    def forward(self, x, seg):
        seq_len = x.size(1)
        pos = torch.arange(seq_len, dtype=torch.long, device=x.device)
        pos = pos.unsqueeze(0).expand_as(x)  # (S,) -> (B, S)

        # factorized embedding
        e = self.tok_embed1(x)
        e = self.tok_embed2(e)
        e = e + self.pos_embed(pos)  # + self.seg_embed(seg)
        # return self.drop(self.norm(e))
        return self.norm(e)


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

    def forward(self, x, mask):
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


class PositionWiseFeedForward(nn.Module):
    """FeedForward Neural Networks for each position"""

    def __init__(self, cfg):
        super().__init__()
        self.fc1 = nn.Linear(cfg.hidden, cfg.hidden_ff)
        self.fc2 = nn.Linear(cfg.hidden_ff, cfg.hidden)
        # self.activ = lambda x: activ_fn(cfg.activ_fn, x)

    def forward(self, x):
        # (B, S, D) -> (B, S, D_ff) -> (B, S, D)
        return self.fc2(gelu(self.fc1(x)))


class Transformer(nn.Module):
    """Transformer with Self-Attentive Blocks"""

    def __init__(self, cfg):
        super().__init__()
        self.embed = Embeddings(cfg)
        # Original BERT not used parameter-sharing strategies
        # self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])

        # To used parameter-sharing strategies
        self.n_layers = cfg.n_layers
        self.attn = MultiHeadedSelfAttention(cfg)
        self.proj = nn.Linear(cfg.hidden, cfg.hidden)
        self.norm1 = LayerNorm(cfg)
        self.pwff = PositionWiseFeedForward(cfg)
        self.norm2 = LayerNorm(cfg)
        # self.drop = nn.Dropout(cfg.p_drop_hidden)

    def forward(self, x, seg=None, mask=None):
        h = self.embed(x, seg)

        for _ in range(self.n_layers):
            # h = block(h, mask)
            h = self.attn(h, mask)
            h = self.norm1(h + self.proj(h))
            h = self.norm2(h + self.pwff(h))

        return h


albert_base_config = {
    "embedding": 128,
    "hidden": 768,
    "hidden_ff": 3072,
    "n_layers": 12,
    "n_heads": 12,
    "max_len": 512,
    "n_segments": 2,
    "vocab_size": 30522,
}


def perf_whole_net(batch_size, seq_len, verbose=False):
    global MODE
    MODE = "whole-net"
    cfg = Config.from_dict(albert_base_config)
    model = Transformer(cfg)
    model = model.cuda().eval()

    # Indices of input sequence tokens in the vocabulary. Indices can be obtained using Tokenizer
    input_ids = torch.zeros([batch_size, seq_len], dtype=torch.long)
    input_ids = input_ids.cuda()

    # output: BaseModelOutputWithPoolingAndCrossAttentions, [bs, seq-len, hid-dim]
    with torch.no_grad():
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


def perf_gemm_chain(batch_size, seq_len, verbose=False):
    global MODE
    MODE = "gemm-chain"
    cfg = Config.from_dict(albert_base_config)
    model = Transformer(cfg)
    model = model.cuda().eval()

    # Indices of input sequence tokens in the vocabulary. Indices can be obtained using Tokenizer
    input_ids = torch.zeros([batch_size, seq_len], dtype=torch.long)
    input_ids = input_ids.cuda()

    # output: BaseModelOutputWithPoolingAndCrossAttentions, [bs, seq-len, hid-dim]
    with torch.no_grad():
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


if __name__ == "__main__":
    batch_size = 1
    seq_len = 512
    net_cost = perf_whole_net(batch_size, seq_len)
    chain_cost = perf_gemm_chain(batch_size, seq_len)
    print("Ratio:", chain_cost / net_cost * 100, "%")
