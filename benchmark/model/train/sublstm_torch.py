'''
Source code partially from https://github.com/pytorch/extension-cpp/blob/master/python/lltm.py
Tutorial at https://pytorch.org/tutorials/advanced/cpp_extension.html#motivation-and-example
'''
import math
import torch
import torch.nn.functional as F
import time
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
import numpy as np
import argparse

torch.manual_seed(42)


class subLSTM(torch.nn.Module):
    def __init__(self, input_size=28*28, state_size=128):
        super(subLSTM, self).__init__()
        self.input_size = input_size
        self.state_size = state_size
        self.weight_ih = torch.nn.Parameter(
            torch.Tensor(4*state_size, input_size))
        self.weight_hh = torch.nn.Parameter(
            torch.Tensor(4*state_size, state_size))
        self.bias_ih = torch.nn.Parameter(torch.Tensor(4*state_size))
        self.bias_hh = torch.nn.Parameter(torch.Tensor(4*state_size))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.state_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, +stdv)

    def forward(self, input, state):
        old_h, old_c = state
        gates = F.linear(input, self.weight_ih, self.bias_ih) + \
            F.linear(old_h, self.weight_hh, self.bias_hh)
        gates = torch.sigmoid(gates)
        in_gate, forget_gate, cell_gate, out_gate = gates.chunk(4, dim=1)

        # in_gate = torch.sigmoid(in_gate)
        # forget_gate = torch.sigmoid(forget_gate)
        # cell_gate = torch.sigmoid(cell_gate)
        # out_gate = torch.sigmoid(out_gate)
        # -> I maually merge these into gates = torch.sigmoid(gates)

        new_c = forget_gate * old_c + cell_gate - in_gate
        new_h = torch.sigmoid(new_c) - out_gate

        return new_h, new_c


class RnnsubLSTM(nn.Module):
    def __init__(self, in_dim=28*28, hidden_dim=128, n_class=10):
        super(RnnsubLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = subLSTM(in_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, n_class)
        self.hx = None

    def forward(self, x):
        if self.hx is None:
            zeros = Variable(torch.zeros(
                batch, self.hidden_dim, dtype=x.dtype, device=x.device))
            self.hx = (zeros, zeros)
        new_h, new_c = self.lstm(x, self.hx)
        self.hx = (Variable(new_h), Variable(new_c))
        out = self.classifier(new_h)
        return out


def MINST_train():
    # This profiling is inaccurate and deprecated!
    learning_rate = 1e-3
    num_epoches = 3

    train_dataset = datasets.MNIST(
        root='./data', train=True, transform=transforms.ToTensor(), download=False)

    test_dataset = datasets.MNIST(
        root='./data', train=False, transform=transforms.ToTensor(), download=False)

    train_loader = DataLoader(
        train_dataset, batch=batch, shuffle=True)
    test_loader = DataLoader(
        test_dataset, batch=batch, shuffle=False)

    model = RnnsubLSTM(28*28, 128, 10)  # 图片大小是28x28
    use_gpu = torch.cuda.is_available()
    assert use_gpu == True
    if use_gpu:
        model = model.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_time_record = []
    infer_time_record = []
    # 开始训练
    for epoch in range(num_epoches):
        print('epoch {}'.format(epoch + 1))
        print('*' * 10)
        running_loss = 0.0
        running_acc = 0.0
        for i, data in enumerate(train_loader, 1):
            img, label = data
            b, c, h, w = img.size()
            assert c == 1, 'channel must be 1'
            if b != batch:
                print("discarded for training")
                continue
            img = img.squeeze(1)
            img = img.view(batch, 28*28)
            if use_gpu:
                img = Variable(img).cuda()
                # print(img.size())
                label = Variable(label).cuda()
            else:
                img = Variable(img)
                label = Variable(label)

            start_train_time = time.time()
            # forward
            out = model(img)
            # compute loss
            # lable: torch.Size(batch)
            loss = criterion(out, label)

            # back-prop
            optimizer.zero_grad()
            loss.backward()

            train_time_record.append(time.time() - start_train_time)

            optimizer.step()

            running_loss += loss.item() * label.size(0)
            _, pred = torch.max(out, 1)
            num_correct = (pred == label).sum()
            running_acc += num_correct.item()

            if i % 30000 == 0:
                print('[{}/{}] Loss: {:.6f}, Acc: {:.6f}'.format(
                    epoch + 1, num_epoches, running_loss / (batch * i),
                    running_acc / (batch * i)))
        print('Finish {} epoch, Loss: {:.6f}, Acc: {:.6f}'.format(
            epoch + 1, running_loss / (len(train_dataset)), running_acc / (len(
                train_dataset))))
        with torch.no_grad():
            # model.eval()
            eval_loss = 0.
            eval_acc = 0.
            for data in test_loader:
                img, label = data
                b, c, h, w = img.size()
                assert c == 1, 'channel must be 1'
                if b != batch:
                    print("discarded for testing")
                    continue
                img = img.squeeze(1)
                img = img.view(batch, 28*28)
                if use_gpu:
                    img = Variable(img, volatile=True).cuda()
                    label = Variable(label, volatile=True).cuda()
                else:
                    img = Variable(img, volatile=True)
                    label = Variable(label, volatile=True)

                start_infer_time = time.time()
                # only forward
                out = model(img)

                infer_time_record.append(time.time() - start_infer_time)

                loss = criterion(out, label)
                eval_loss += loss.item() * label.size(0)
                _, pred = torch.max(out, 1)
                num_correct = (pred == label).sum()
                eval_acc += num_correct.item()
            print('Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / (len(
                test_dataset)), eval_acc / (len(test_dataset))))
    print("train time median:", np.median(train_time_record),
          "max", np.max(train_time_record))
    print("infer time meidan:", np.median(infer_time_record),
          "max", np.max(infer_time_record))


def train_perf(device=0):
    model = RnnsubLSTM(28*28, 128, 10).cuda("cuda:" + str(device))
    model.train()
    dtype = "float32"
    img = np.random.uniform(-1, 1, [batch, 28*28]).astype(dtype)
    img_tensor = torch.tensor(img).cuda("cuda:" + str(device))
    label_tensor = torch.empty(batch, dtype=torch.long).random_(
        10).cuda("cuda:" + str(device))
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
            images, reconstructions = 0, 0
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
    model = RnnsubLSTM(28*28, 128, 10).cuda("cuda:" + str(device))
    model.eval()
    dtype = "float32"
    img = np.random.uniform(-1, 1, [batch, 28*28]).astype(dtype)
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
