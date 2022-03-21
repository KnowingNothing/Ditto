from ..module import (
    Module,
    Conv2d,
    BatchNorm2d,
    ReLU,
    AvgPool2d,
    GlobalAvgPool2d,
    Linear,
    Sequential,
    ShuffleChannel,
    Add,
    BatchFlatten,
    CatChannel,
)


def conv_bn(
    in_channel,
    out_channel,
    kernel_size,
    strides=1,
    padding=0,
    dilation=1,
    groups=1,
    use_bias=False,
    dtype="float32",
    out_dtype="float32",
):
    return Sequential(
        Conv2d(
            in_channel,
            out_channel,
            kernel_size,
            stride=strides,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=use_bias,
            dtype=dtype,
            out_dtype=out_dtype,
        ),
        BatchNorm2d(out_channel, dtype=dtype, out_dtype=out_dtype),
        ReLU(),
    )


class ShuffleNetUnitA(Module):
    """ShuffleNet unit for stride=1"""

    def __init__(
        self, in_channels, out_channels, groups=3, dtype="float32", out_dtype="float32"
    ):
        super(ShuffleNetUnitA, self).__init__()
        assert in_channels == out_channels
        assert out_channels % 4 == 0
        bottleneck_channels = out_channels // 4
        self.groups = groups
        self.group_conv1 = Conv2d(
            in_channels,
            bottleneck_channels,
            1,
            groups=groups,
            stride=1,
            dtype=dtype,
            out_dtype=out_dtype,
        )
        self.bn2 = BatchNorm2d(bottleneck_channels, dtype=dtype, out_dtype=out_dtype)
        self.depthwise_conv3 = Conv2d(
            bottleneck_channels,
            bottleneck_channels,
            3,
            padding=1,
            stride=1,
            groups=bottleneck_channels,
            dtype=dtype,
            out_dtype=out_dtype,
        )
        self.bn4 = BatchNorm2d(bottleneck_channels, dtype=dtype, out_dtype=out_dtype)
        self.group_conv5 = Conv2d(
            bottleneck_channels,
            out_channels,
            1,
            stride=1,
            groups=groups,
            dtype=dtype,
            out_dtype=out_dtype,
        )
        self.bn6 = BatchNorm2d(out_channels, dtype=dtype, out_dtype=out_dtype)

        self.add = Add()
        self.relu = ReLU()
        self.shuffle = ShuffleChannel(self.groups)

    def forward(self, x):
        out = self.group_conv1(x)
        out = self.relu(self.bn2(out))
        out = self.shuffle(out)
        out = self.depthwise_conv3(out)
        out = self.bn4(out)
        out = self.group_conv5(out)
        out = self.bn6(out)
        out = self.add(x, out)
        out = self.relu(out)
        return out


class ShuffleNetUnitB(Module):
    """ShuffleNet unit for stride=2"""

    def __init__(
        self, in_channels, out_channels, groups=3, dtype="float32", out_dtype="float32"
    ):
        super(ShuffleNetUnitB, self).__init__()
        out_channels -= in_channels
        assert out_channels % 4 == 0
        bottleneck_channels = out_channels // 4
        self.groups = groups
        self.group_conv1 = Conv2d(
            in_channels,
            bottleneck_channels,
            1,
            groups=groups,
            stride=1,
            dtype=dtype,
            out_dtype=out_dtype,
        )
        self.bn2 = BatchNorm2d(bottleneck_channels, dtype=dtype, out_dtype=out_dtype)
        self.depthwise_conv3 = Conv2d(
            bottleneck_channels,
            bottleneck_channels,
            3,
            padding=1,
            stride=2,
            groups=bottleneck_channels,
            dtype=dtype,
            out_dtype=out_dtype,
        )
        self.bn4 = BatchNorm2d(bottleneck_channels, dtype=dtype, out_dtype=out_dtype)
        self.group_conv5 = Conv2d(
            bottleneck_channels,
            out_channels,
            1,
            stride=1,
            groups=groups,
            dtype=dtype,
            out_dtype=out_dtype,
        )
        self.bn6 = BatchNorm2d(out_channels, dtype=dtype, out_dtype=out_dtype)

        self.relu = ReLU()
        self.shuffle = ShuffleChannel(self.groups)
        self.avgpool2d = AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.cat = CatChannel()

    def forward(self, x):
        out = self.group_conv1(x)
        out = self.relu(self.bn2(out))
        out = self.shuffle(out)
        out = self.depthwise_conv3(out)
        out = self.bn4(out)
        out = self.group_conv5(out)
        out = self.bn6(out)
        x = self.avgpool2d(x)
        out = self.relu(self.cat(x, out))
        return out


class ShuffleNet(Module):
    """ShuffleNet for groups=3"""

    def __init__(
        self,
        groups=3,
        in_channels=3,
        num_classes=1000,
        dtype="float32",
        out_dtype="float32",
    ):
        super(ShuffleNet, self).__init__()

        self.conv1 = Conv2d(
            in_channels, 24, 3, stride=2, padding=1, dtype=dtype, out_dtype=out_dtype
        )
        stage2_seq = [
            ShuffleNetUnitB(24, 240, groups=3, dtype=dtype, out_dtype=out_dtype)
        ] + [
            ShuffleNetUnitA(240, 240, groups=3, dtype=dtype, out_dtype=out_dtype)
            for i in range(3)
        ]
        self.stage2 = Sequential(*stage2_seq)
        stage3_seq = [
            ShuffleNetUnitB(240, 480, groups=3, dtype=dtype, out_dtype=out_dtype)
        ] + [
            ShuffleNetUnitA(480, 480, groups=3, dtype=dtype, out_dtype=out_dtype)
            for i in range(7)
        ]
        self.stage3 = Sequential(*stage3_seq)
        stage4_seq = [
            ShuffleNetUnitB(480, 960, groups=3, dtype=dtype, out_dtype=out_dtype)
        ] + [
            ShuffleNetUnitA(960, 960, groups=3, dtype=dtype, out_dtype=out_dtype)
            for i in range(3)
        ]
        self.stage4 = Sequential(*stage4_seq)
        self.fc = Linear(960, num_classes, dtype=dtype, out_dtype=out_dtype)

        self.flatten = BatchFlatten()
        self.avgpool2d1 = AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.avgpool2d2 = AvgPool2d(kernel_size=7, stride=7, padding=0)

    def forward(self, x):
        net = self.conv1(x)
        # TODO: change to max_pool2d(net, 3, stride=2, padding=1)
        net = self.avgpool2d1(net)
        net = self.stage2(net)
        net = self.stage3(net)
        net = self.stage4(net)
        net = self.avgpool2d2(net)
        net = self.flatten(net)
        net = self.fc(net)
        return net
