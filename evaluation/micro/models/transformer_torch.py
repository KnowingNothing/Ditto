import torch
import numpy as np
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

records = []
profile_bmm = True

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)
        
        if profile_bmm:
            global records
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        if profile_bmm:
            end.record()
            torch.cuda.synchronize()
            total = start.elapsed_time(end)
            records.append(total)
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

    
if __name__ == "__main__":
    batch = 1
    
    # seq_len = 512
    # dim = 512
    # depth = 4
    # heads = 8
    # mlp_dim = 512
    
    # seq_len = 512
    # dim = 768
    # depth = 12
    # heads = 12
    # mlp_dim = 3072
    
    seq_len = 512
    dim = 1024
    depth = 24
    heads = 16
    mlp_dim = 3072
    dim_head = dim // heads
    model = Transformer(dim, depth, heads, dim_head, mlp_dim)
    model = model.cuda().eval().half()

    # Indices of input sequence tokens in the vocabulary. Indices can be obtained using BertTokenizer
    input_ids = torch.zeros([batch, seq_len, dim], dtype=torch.half)
    input_ids = input_ids.cuda()

    # output: BaseModelOutputWithPoolingAndCrossAttentions, [bs, seq-len, hid-dim]
    with torch.no_grad():
        output = model(input_ids)

    number = 10
    repeats = 10

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
            if profile_bmm:
                print("Best BMM latency", min(records), "ms")
                print("Average BMM latency", np.mean(records), "ms")
                print("Median  BMM latency", np.median(records), "ms")
            else:
                print("Best Transformer latency", min(time_record), "ms")
                print("Average Transformer latency", np.mean(time_record), "ms")
                print("Median  Transformer latency", np.median(time_record), "ms")