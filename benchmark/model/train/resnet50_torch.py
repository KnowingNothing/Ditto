import torch
import numpy as np
import torch.nn as nn
import math
from torch import optim
import torch.nn.functional as F
import resnet_torch as R
import argparse


def train_perf(device=0):
    model = R.resnet50().cuda("cuda:" + str(device))
    model.train()
    dtype = "float32"
    img = np.random.uniform(-1, 1, [batch, 3, 224, 224]).astype(dtype)
    img_tensor = torch.tensor(img).cuda("cuda:" + str(device))
    label_tensor = torch.empty(batch, dtype=torch.long).random_(
        1000).cuda("cuda:" + str(device))
    model(img_tensor)
    number = 10
    repeats = 10

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(model.parameters(), lr=0.002)

    for i in range(number):
        time_record = []
        for j in range(repeats):
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            optimizer.zero_grad()
            lltm_output = model(img_tensor)
            loss = criterion(lltm_output, label_tensor)

            start.record()
            loss.backward()
            end.record()

            optimizer.step()

            torch.cuda.synchronize()
            total = start.elapsed_time(end)
            time_record.append(total)
        print("Average training latency", np.mean(time_record))
        print("Median training latency", np.median(time_record))
    print("batch = ", batch)


def inference_perf(device=0):
    model = R.resnet50().cuda("cuda:" + str(device))
    model.eval()
    dtype = "float32"
    img = np.random.uniform(-1, 1, [batch, 3, 224, 224]).astype(dtype)
    img_tensor = torch.tensor(img).cuda("cuda:" + str(device))
    model(img_tensor)
    number = 10
    repeats = 10

    for i in range(repeats):
        time_record = []
        for j in range(number):
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record()
            output = model(img_tensor)
            end.record()

            torch.cuda.synchronize()
            total = start.elapsed_time(end)
            time_record.append(total)
        print("Average inference latency", np.mean(time_record))
        print("Median inference latency", np.median(time_record))
    print("batch = ", batch)


if __name__ == "__main__":
    device = 0
    for cudnn in [True, False]:
        for batch in [1, 16, 32, 64]:
            torch.backends.cudnn.enabled = cudnn
            train_perf(device)
            inference_perf(device)
            print("use cuDNN", cudnn)
            print()
