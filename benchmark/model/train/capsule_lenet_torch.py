import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from torch import optim


USE_CUDA = True if torch.cuda.is_available() else False


class ConvLayer(nn.Module):
    def __init__(self, in_channels=1, out_channels=256, kernel_size=9):
        super(ConvLayer, self).__init__()

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
        )

    def forward(self, x):
        return F.relu(self.conv(x))


class PrimaryCaps(nn.Module):
    def __init__(
        self,
        num_capsules=8,
        in_channels=256,
        out_channels=32,
        kernel_size=9,
        num_routes=32 * 6 * 6,
    ):
        super(PrimaryCaps, self).__init__()
        self.num_routes = num_routes
        self.capsules = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=2,
                    padding=0,
                )
                for _ in range(num_capsules)
            ]
        )

    def forward(self, x):
        u = [capsule(x) for capsule in self.capsules]
        u = torch.stack(u, dim=1)
        u = u.view(x.size(0), self.num_routes, -1)
        ret = self.squash(u)
        print(ret.shape)
        return ret

    def squash(self, input_tensor):
        squared_norm = (input_tensor ** 2).sum(-1, keepdim=True)
        output_tensor = (
            squared_norm
            * input_tensor
            / ((1.0 + squared_norm) * torch.sqrt(squared_norm))
        )
        return output_tensor


class DigitCaps(nn.Module):
    def __init__(
        self, num_capsules=10, num_routes=32 * 6 * 6, in_channels=8, out_channels=16
    ):
        super(DigitCaps, self).__init__()

        self.in_channels = in_channels
        self.num_routes = num_routes
        self.num_capsules = num_capsules

        self.W = nn.Parameter(
            torch.randn(1, num_routes, num_capsules, out_channels, in_channels)
        )

    def forward(self, x):
        batch_size = x.size(0)
        x = torch.stack([x] * self.num_capsules, dim=2).unsqueeze(4)
        print(x.shape)

        W = torch.cat([self.W] * batch_size, dim=0)
        print(W.shape)
        u_hat = torch.matmul(W, x)
        print(u_hat.shape)

        b_ij = Variable(torch.zeros(1, self.num_routes, self.num_capsules, 1))
        if USE_CUDA:
            b_ij = b_ij.cuda()

        num_iterations = 3
        for iteration in range(num_iterations):
            c_ij = F.softmax(b_ij, dim=1)
            print(c_ij.shape)
            c_ij = torch.cat([c_ij] * batch_size, dim=0).unsqueeze(4)
            print(c_ij.shape)
            s_j = (c_ij * u_hat).sum(dim=1, keepdim=True)
            print(s_j.shape)
            v_j = self.squash(s_j)

            if iteration < num_iterations - 1:
                a_ij = torch.matmul(
                    u_hat.transpose(3, 4), torch.cat([v_j] * self.num_routes, dim=1)
                )
                print("a_ij=", a_ij.shape)
                b_ij = b_ij + a_ij.squeeze(4).mean(dim=0, keepdim=True)

        return v_j.squeeze(1)

    def squash(self, input_tensor):
        squared_norm = (input_tensor ** 2).sum(-1, keepdim=True)
        output_tensor = (
            squared_norm
            * input_tensor
            / ((1.0 + squared_norm) * torch.sqrt(squared_norm))
        )
        return output_tensor


class CapsNet(nn.Module):
    def __init__(self, config=None):
        super(CapsNet, self).__init__()
        if config:
            self.conv_layer = ConvLayer(
                config.cnn_in_channels, config.cnn_out_channels, config.cnn_kernel_size
            )
            self.primary_capsules = PrimaryCaps(
                config.pc_num_capsules,
                config.pc_in_channels,
                config.pc_out_channels,
                config.pc_kernel_size,
                config.pc_num_routes,
            )
            self.digit_capsules = DigitCaps(
                config.dc_num_capsules,
                config.dc_num_routes,
                config.dc_in_channels,
                config.dc_out_channels,
            )
        else:
            self.conv_layer = ConvLayer()
            self.primary_capsules = PrimaryCaps()
            self.digit_capsules = DigitCaps()

        self.mse_loss = nn.MSELoss()

    def forward(self, data):
        output = self.digit_capsules(self.primary_capsules(self.conv_layer(data)))
        return output


def train_perf(batch_size=1, device=0):
    model = CapsNet().cuda("cuda:" + str(device))
    model.train()
    dtype = "float32"
    img = np.random.uniform(-1, 1, [batch_size, 1, 28, 28]).astype(dtype)
    img_tensor = torch.tensor(img).cuda("cuda:" + str(device))
    label_tensor = (
        torch.empty(batch_size, dtype=torch.long)
        .random_(1000)
        .cuda("cuda:" + str(device))
    )
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
            start.record()

            optimizer.zero_grad()
            lltm_output = model(img_tensor)
            loss = criterion(lltm_output, label_tensor)
            loss.backward()
            optimizer.step()

            end.record()
            torch.cuda.synchronize()
            total = start.elapsed_time(end)
            time_record.append(total)
        print("Average training latency", np.mean(time_record))
        print("Median training latency", np.median(time_record))
    print("batch = ", batch_size)


def inference_perf(batch_size=1, device=0):
    model = CapsNet().cuda("cuda:" + str(device))
    model.eval()
    dtype = "float32"
    img = np.random.uniform(-1, 1, [batch_size, 3, 224, 224]).astype(dtype)
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
    print("batch = ", batch_size)


if __name__ == "__main__":
    train_perf(batch_size=1)
