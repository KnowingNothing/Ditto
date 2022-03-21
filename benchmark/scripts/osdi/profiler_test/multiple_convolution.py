import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import argparse


if __name__ == "__main__":
    img = np.random.uniform(-1, 1, [1, 512, 14, 14]).astype("float32")
    img = torch.tensor(img).cuda()
    for i in range(1):
        conv = nn.Conv2d(512, 512, kernel_size=3, padding=1).cuda()
        img = conv(img)
