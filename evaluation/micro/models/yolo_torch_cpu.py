import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import time

records = []
profile_conv = False

class YOLONetV1(nn.Module):
    def __init__(self, channel=3, height=448, width=448):
        super(YOLONetV1, self).__init__()

        self.conv1 = nn.Conv2d(channel, 64, 7, stride=2, padding=3)
        self.pool1 = nn.AvgPool2d(2, stride=2)

        self.conv2 = nn.Conv2d(64, 192, 3, stride=2, padding=1)

        self.conv3 = nn.Conv2d(192, 128, 1)
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv5 = nn.Conv2d(256, 256, 1)
        self.conv6 = nn.Conv2d(256, 512, 3, padding=1)

        self.pool6 = nn.AvgPool2d(2, stride=2)

        self.conv7 = nn.Conv2d(512, 256, 1)
        self.conv8 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv9 = nn.Conv2d(512, 256, 1)
        self.conv10 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv11 = nn.Conv2d(512, 256, 1)
        self.conv12 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv13 = nn.Conv2d(512, 256, 1)
        self.conv14 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv15 = nn.Conv2d(512, 512, 1)
        self.conv16 = nn.Conv2d(512, 1024, 3, padding=1)

        self.pool16 = nn.AvgPool2d(2, stride=2)

        self.conv17 = nn.Conv2d(1024, 512, 1)
        self.conv18 = nn.Conv2d(512, 1024, 3, padding=1)
        self.conv19 = nn.Conv2d(1024, 512, 1)
        self.conv20 = nn.Conv2d(512, 1024, 3, padding=1)
        self.conv21 = nn.Conv2d(1024, 1024, 3, padding=1)
        self.conv22 = nn.Conv2d(1024, 1024, 3, stride=2, padding=1)
        self.conv23 = nn.Conv2d(1024, 1024, 3, padding=1)
        self.conv24 = nn.Conv2d(1024, 1024, 3, padding=1)

        self.fc25 = nn.Linear(1024 * 7 * 7, 4096)

        self.fc26 = nn.Linear(4096, 1470)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))

        x = self.pool1(x)

        if profile_conv:
            start = time.time()
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        if profile_conv:
            end = time.time()
            global records
            records.append((end - start) * 1e3)

        x = self.relu(x)
        x = self.relu(self.conv4(x))
        x = self.relu(self.conv5(x))
        x = self.relu(self.conv6(x))

        x = self.pool6(x)

        x = self.relu(self.conv7(x))
        x = self.relu(self.conv8(x))
        x = self.relu(self.conv9(x))
        x = self.relu(self.conv10(x))
        x = self.relu(self.conv11(x))
        x = self.relu(self.conv12(x))
        x = self.relu(self.conv13(x))
        x = self.relu(self.conv14(x))
        x = self.relu(self.conv15(x))
        x = self.relu(self.conv16(x))

        x = self.pool16(x)

        x = self.relu(self.conv17(x))
        x = self.relu(self.conv18(x))
        x = self.relu(self.conv19(x))
        x = self.relu(self.conv20(x))
        x = self.relu(self.conv21(x))
        x = self.relu(self.conv22(x))
        x = self.relu(self.conv23(x))
        x = self.relu(self.conv24(x))

        x = torch.flatten(x, start_dim=1)

        x = self.relu(self.fc25(x))

        x = self.fc26(x)

        return x


def yolo_v1():
    model = YOLONetV1()
    return model


if __name__ == "__main__":
    dev = torch.device("cpu")
    model = yolo_v1().to(dev)
    batch = 1
    dtype = "float32"
    img = np.random.uniform(-1, 1, [batch, 3, 448, 448]).astype(dtype)
    img_tensor = torch.tensor(img).to(dev)
    model(img_tensor)
    number = 100
    repeats = 10
    start = time.time()
    for i in range(repeats):
        for j in range(number):
            output = model(img_tensor)
    end = time.time()
    total = (end - start) * 1e3
    if profile_conv:
        print("Average cost for conv chain:", np.mean(records), "ms.")
    else:
        print("Average cost for one iteration:", total / (repeats * number), "ms.")
