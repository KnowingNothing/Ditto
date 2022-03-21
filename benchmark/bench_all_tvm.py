import argparse
import math

import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

from bench_tvm_utils import bench_network


batch_size = 1


""" MILSTM """


class MILSTM_Cell(nn.Module):
    def __init__(self, input_size=28 * 28, hidden_size=1024):
        super(MILSTM_Cell, self).__init__()

        self.hidden_size = hidden_size
        # lstm weights
        self.weight_fh = nn.Linear(hidden_size, hidden_size)
        self.weight_ih = nn.Linear(hidden_size, hidden_size)
        self.weight_zh = nn.Linear(hidden_size, hidden_size)
        self.weight_oh = nn.Linear(hidden_size, hidden_size)
        self.weight_fx = nn.Linear(input_size, hidden_size)
        self.weight_ix = nn.Linear(input_size, hidden_size)
        self.weight_zx = nn.Linear(input_size, hidden_size)
        self.weight_ox = nn.Linear(input_size, hidden_size)
        # alphas and betas
        self.alpha_f = nn.Parameter(torch.ones(1, hidden_size))
        self.beta_f1 = nn.Parameter(torch.ones(1, hidden_size))
        self.beta_f2 = nn.Parameter(torch.ones(1, hidden_size))

        self.alpha_i = nn.Parameter(torch.ones(1, hidden_size))
        self.beta_i1 = nn.Parameter(torch.ones(1, hidden_size))
        self.beta_i2 = nn.Parameter(torch.ones(1, hidden_size))

        self.alpha_o = nn.Parameter(torch.ones(1, hidden_size))
        self.beta_o1 = nn.Parameter(torch.ones(1, hidden_size))
        self.beta_o2 = nn.Parameter(torch.ones(1, hidden_size))

        self.alpha_z = nn.Parameter(torch.ones(1, hidden_size))
        self.alpha_z = nn.Parameter(torch.ones(1, hidden_size))
        self.beta_z1 = nn.Parameter(torch.ones(1, hidden_size))
        self.beta_z2 = nn.Parameter(torch.ones(1, hidden_size))

    def forward(self, inp, h_0, c_0):
        # inp : [batch, 28*28]
        # gates : [batch, hidden_size]

        # forget gate
        f_g = torch.sigmoid(
            self.alpha_f * self.weight_fx(inp) * self.weight_fh(h_0)
            + (self.beta_f1 * self.weight_fx(inp))
            + (self.beta_f2 * self.weight_fh(h_0))
        )
        # input gate
        i_g = torch.sigmoid(
            self.alpha_i * self.weight_ix(inp) * self.weight_ih(h_0)
            + (self.beta_i1 * self.weight_ix(inp))
            + (self.beta_i2 * self.weight_ih(h_0))
        )
        # output gate
        o_g = torch.sigmoid(
            self.alpha_o * self.weight_ox(inp) * self.weight_oh(h_0)
            + (self.beta_o1 * self.weight_ox(inp))
            + (self.beta_o2 * self.weight_oh(h_0))
        )
        # block input
        z_t = torch.tanh(
            self.alpha_z * self.weight_zx(inp) * self.weight_zh(h_0)
            + (self.beta_z1 * self.weight_zx(inp))
            + (self.beta_z2 * self.weight_zh(h_0))
        )
        # current cell state
        cx = f_g * c_0 + i_g * z_t
        # hidden state
        hx = o_g * torch.tanh(cx)

        return hx, cx


# NOTE(yicheng): disabled internal state updating of RNN for TorchScript tracing,
# should not affect the accuracy of benchmarking. same for other RNN models


class MILSTM(nn.Module):
    def __init__(self, input_size=28 * 28, hidden_size=1024, n_class=10):
        super(MILSTM, self).__init__()
        self.n_class = n_class
        self.hidden_size = hidden_size
        self.lstm = MILSTM_Cell(input_size=input_size, hidden_size=hidden_size)
        self.classifier = nn.Linear(hidden_size, n_class)
        self.h = torch.zeros(batch_size, self.hidden_size)
        self.c = torch.zeros(batch_size, self.hidden_size)

    def forward(self, x):
        # if self.h is None:
        #     zeroh = Variable(torch.zeros(batch_size, self.hidden_size, dtype=x.dtype, device=x.device))
        #     zeroc = Variable(torch.zeros(batch_size, self.hidden_size, dtype=x.dtype, device=x.device))
        #     self.h = zeroh
        #     self.c = zeroc
        new_h, new_c = self.lstm(x, self.h, self.c)
        # self.h, self.c = Variable(new_h), Variable(new_c)
        out = self.classifier(new_h)
        return out


""" scRNN """


class SCRNNCell(nn.Module):
    def __init__(self, input_size=28 * 28, num_units=128, context_units=64, alpha=0.5):
        super(SCRNNCell, self).__init__()
        self._input_size = input_size
        self._num_units = num_units
        self._context_units = context_units
        self._alpha = alpha
        self.B = nn.Parameter(torch.empty(input_size, context_units))
        self.V = nn.Parameter(torch.empty(context_units, num_units))
        self.U = nn.Parameter(torch.empty(num_units, num_units))
        self.fc = nn.Linear(
            context_units + input_size + num_units, num_units, bias=False
        )
        self.reset_parameters()

    def forward(self, inputs, state_h, state_c):
        context_state = (1 - self._alpha) * (inputs @ self.B) + self._alpha * state_c
        concated = torch.cat([context_state, inputs, state_h], dim=1)
        hidden_state = torch.sigmoid(self.fc(concated))
        new_h = hidden_state @ self.U + context_state @ self.V
        return new_h, context_state

    def reset_parameters(self):
        for weight in self.parameters():
            nn.init.xavier_uniform_(weight, gain=1.0)


class SCRNN(nn.Module):
    def __init__(self, num_units=128, context_units=64, n_class=10):
        super(SCRNN, self).__init__()
        self.num_units = num_units
        self.context_units = context_units
        self.lstm = SCRNNCell(num_units=num_units, context_units=context_units)
        self.classifier = nn.Linear(num_units, n_class)
        self.h = torch.zeros(batch_size, self.num_units)
        self.c = torch.zeros(batch_size, self.context_units)

    def forward(self, x):
        # if self.h is None:
        #     zeroh = Variable(torch.zeros(batch_size, self.num_units, dtype=x.dtype, device=x.device))
        #     zeroc = Variable(torch.zeros(batch_size, self.context_units, dtype=x.dtype, device=x.device))
        #     self.h = zeroh
        #     self.c = zeroc
        new_h, new_c = self.lstm(x, self.h, self.c)
        # self.h, self.c = Variable(new_h), Variable(new_c)
        out = self.classifier(new_h)
        return out


""" Capsule Network """


class ConvLayer(nn.Module):
    def __init__(self, in_channels=1, out_channels=256):
        super(ConvLayer, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=9, stride=1, padding=0
        )

    def forward(self, x):
        conved = self.conv(x)
        features = F.relu(conved)
        return features


class PrimaryCaps(nn.Module):
    def __init__(self, num_capsules=8, in_channels=256, out_channels=32):
        super(PrimaryCaps, self).__init__()
        self.capsules = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=9,
                    stride=2,
                    padding=0,
                )
                for _ in range(num_capsules)
            ]
        )

    def forward(self, x):
        # get batch size of inputs
        u = [capsule(x).view(batch_size, 32 * 6 * 6, 1) for capsule in self.capsules]
        u = torch.cat(u, dim=-1)
        u_squash = self.squash(u)
        return u_squash

    def squash(self, input_tensor):
        squared_norm = (input_tensor ** 2).sum(dim=-1, keepdim=True)
        scale = squared_norm / (1 + squared_norm)  # normalization coeff
        output_tensor = scale * input_tensor / torch.sqrt(squared_norm)
        return output_tensor


def softmax(input_tensor, dim=2):
    transposed_input = input_tensor.transpose(dim, len(input_tensor.size()) - 1)
    softmaxed_output = F.softmax(
        transposed_input.contiguous().view(-1, transposed_input.size(-1)), dim=-1
    )
    return softmaxed_output.view(*transposed_input.size()).transpose(
        dim, len(input_tensor.size()) - 1
    )


def dynamic_routing(b_ij, u_hat, squash, routing_iterations=3):
    for iteration in range(routing_iterations):
        c_ij = softmax(b_ij, dim=2)
        s_j = (c_ij * u_hat).sum(dim=2, keepdim=True)
        v_j = squash(s_j)
        if iteration < routing_iterations - 1:
            a_ij = (u_hat * v_j).sum(dim=-1, keepdim=True)
            b_ij = b_ij + a_ij
    return v_j


class DigitCaps(nn.Module):
    def __init__(
        self,
        num_capsules=10,
        previous_layer_nodes=32 * 6 * 6,
        in_channels=8,
        out_channels=16,
    ):
        super(DigitCaps, self).__init__()
        self.num_capsules = num_capsules
        self.previous_layer_nodes = previous_layer_nodes
        self.in_channels = in_channels

        self.W = nn.Parameter(
            torch.randn(num_capsules, previous_layer_nodes, in_channels, out_channels)
        )

    def forward(self, u):
        u = u[None, :, :, None, :]
        W = self.W[:, None, :, :, :]

        # NOTE(yicheng): pytorch broadcasts the first dimension of u during matmul automatically.
        # This is not properly handled by relay and will cause errors during model conversion.
        # Thus we explicitly perform broadcast here.
        u = u.expand(W.shape[0], -1, -1, -1, -1)

        u_hat = torch.matmul(u, W)
        b_ij = torch.zeros_like(u_hat)

        v_j = dynamic_routing(b_ij, u_hat, self.squash, routing_iterations=3)

        return v_j

    def squash(self, input_tensor):
        squared_norm = (input_tensor ** 2).sum(dim=-1, keepdim=True)
        scale = squared_norm / (1 + squared_norm)
        output_tensor = scale * input_tensor / torch.sqrt(squared_norm)
        return output_tensor


class CapsuleNetwork(nn.Module):
    def __init__(self):
        super(CapsuleNetwork, self).__init__()
        self.conv_layer = ConvLayer()
        self.primary_capsules = PrimaryCaps()
        self.digit_capsules = DigitCaps()

    def forward(self, images):
        primary_caps_output = self.primary_capsules(self.conv_layer(images))
        caps_output = self.digit_capsules(primary_caps_output).squeeze().transpose(0, 1)
        # squeeze can will delete all 1 in dims, which is unexpected
        if batch_size == 1:
            caps_output = caps_output.reshape(batch_size, 10, 16)
        return caps_output


""" LLTM """


class LLTM(torch.nn.Module):
    def __init__(self, input_features, state_size):
        super(LLTM, self).__init__()
        self.input_features = input_features
        self.state_size = state_size
        # 3 * state_size for input gate, output gate and candidate cell gate.
        # input_features + state_size because we will multiply with [input, h].
        self.weights = torch.nn.Parameter(
            torch.Tensor(3 * state_size, input_features + state_size)
        )
        self.bias = torch.nn.Parameter(torch.Tensor(1, 3 * state_size))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.state_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, +stdv)

    def forward(self, input, state):
        old_h, old_cell = state
        X = torch.cat([old_h, input], dim=1)

        # Compute the input, output and candidate cell gates with one MM.
        gate_weights = F.linear(X, self.weights, self.bias)
        # Split the combined gate weight matrix into its components.
        gates = gate_weights.chunk(3, dim=1)

        input_gate = torch.sigmoid(gates[0])
        output_gate = torch.sigmoid(gates[1])
        # Here we use an ELU instead of the usual tanh.
        candidate_cell = F.elu(gates[2])

        # Compute the new cell state.
        new_cell = old_cell + candidate_cell * input_gate
        # Compute the new hidden state and output.
        new_h = torch.tanh(new_cell) * output_gate

        return new_h, new_cell


class RnnLLTM(nn.Module):
    def __init__(self, in_dim=28 * 28, hidden_dim=128, n_class=10):
        super(RnnLLTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = LLTM(in_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, n_class)
        self.hx = (
            torch.zeros(batch_size, self.hidden_dim),
            torch.zeros(batch_size, self.hidden_dim),
        )

    def forward(self, x):
        # if self.hx is None:
        #     zeros = Variable(torch.zeros(batch_size, self.hidden_dim,dtype=x.dtype, device=x.device))
        #     self.hx = (zeros, zeros)
        new_h, new_c = self.lstm(x, self.hx)
        # self.hx = (Variable(new_h), Variable(new_c))
        out = self.classifier(new_h)
        return out


""" ShuffleNet """


# NOTE(yicheng): originally disabled, turned on to circumvent a complicated issue in the model conversion process
# to guarantee strict equivalence, we ensure manually refactor BN-without-tracking into smaller operations
bn_track_running_stats = True


def shuffle_channels(x, groups):
    """shuffle channels of a 4-D Tensor"""
    batch_size, channels, height, width = x.size()
    assert channels % groups == 0
    channels_per_group = channels // groups
    # split into groups
    x = x.view(batch_size, groups, channels_per_group, height, width)
    # transpose 1, 2 axis
    x = x.transpose(1, 2).contiguous()
    # reshape into orignal
    x = x.view(batch_size, channels, height, width)
    return x


class ShuffleNetUnitA(nn.Module):
    """ShuffleNet unit for stride=1"""

    def __init__(self, in_channels, out_channels, groups=3):
        super(ShuffleNetUnitA, self).__init__()
        assert in_channels == out_channels
        assert out_channels % 4 == 0
        bottleneck_channels = out_channels // 4
        self.groups = groups
        self.group_conv1 = nn.Conv2d(
            in_channels, bottleneck_channels, 1, groups=groups, stride=1
        )
        self.bn2 = nn.BatchNorm2d(
            bottleneck_channels, track_running_stats=bn_track_running_stats
        )
        self.depthwise_conv3 = nn.Conv2d(
            bottleneck_channels,
            bottleneck_channels,
            3,
            padding=1,
            stride=1,
            groups=bottleneck_channels,
        )
        self.bn4 = nn.BatchNorm2d(
            bottleneck_channels, track_running_stats=bn_track_running_stats
        )
        self.group_conv5 = nn.Conv2d(
            bottleneck_channels, out_channels, 1, stride=1, groups=groups
        )
        self.bn6 = nn.BatchNorm2d(
            out_channels, track_running_stats=bn_track_running_stats
        )

    def forward(self, x):
        out = self.group_conv1(x)
        out = F.relu(self.bn2(out))
        out = shuffle_channels(out, groups=self.groups)
        out = self.depthwise_conv3(out)
        out = self.bn4(out)
        out = self.group_conv5(out)
        out = self.bn6(out)
        out = F.relu(x + out)
        return out


class ShuffleNetUnitB(nn.Module):
    """ShuffleNet unit for stride=2"""

    def __init__(self, in_channels, out_channels, groups=3):
        super(ShuffleNetUnitB, self).__init__()
        out_channels -= in_channels
        assert out_channels % 4 == 0
        bottleneck_channels = out_channels // 4
        self.groups = groups
        self.group_conv1 = nn.Conv2d(
            in_channels, bottleneck_channels, 1, groups=groups, stride=1
        )
        self.bn2 = nn.BatchNorm2d(
            bottleneck_channels, track_running_stats=bn_track_running_stats
        )
        self.depthwise_conv3 = nn.Conv2d(
            bottleneck_channels,
            bottleneck_channels,
            3,
            padding=1,
            stride=2,
            groups=bottleneck_channels,
        )
        self.bn4 = nn.BatchNorm2d(
            bottleneck_channels, track_running_stats=bn_track_running_stats
        )
        self.group_conv5 = nn.Conv2d(
            bottleneck_channels, out_channels, 1, stride=1, groups=groups
        )
        self.bn6 = nn.BatchNorm2d(
            out_channels, track_running_stats=bn_track_running_stats
        )

    def forward(self, x):
        out = self.group_conv1(x)
        out = F.relu(self.bn2(out))
        out = shuffle_channels(out, groups=self.groups)
        out = self.depthwise_conv3(out)
        out = self.bn4(out)
        out = self.group_conv5(out)
        out = self.bn6(out)
        x = F.avg_pool2d(x, 3, stride=2, padding=1)
        out = F.relu(torch.cat([x, out], dim=1))
        return out


class ShuffleNet(nn.Module):
    """ShuffleNet for groups=3"""

    def __init__(self, groups=3, in_channels=3, num_classes=1000):
        super(ShuffleNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, 24, 3, stride=2, padding=1)
        stage2_seq = [ShuffleNetUnitB(24, 240, groups=3)] + [
            ShuffleNetUnitA(240, 240, groups=3) for i in range(3)
        ]
        self.stage2 = nn.Sequential(*stage2_seq)
        stage3_seq = [ShuffleNetUnitB(240, 480, groups=3)] + [
            ShuffleNetUnitA(480, 480, groups=3) for i in range(7)
        ]
        self.stage3 = nn.Sequential(*stage3_seq)
        stage4_seq = [ShuffleNetUnitB(480, 960, groups=3)] + [
            ShuffleNetUnitA(960, 960, groups=3) for i in range(3)
        ]
        self.stage4 = nn.Sequential(*stage4_seq)
        self.fc = nn.Linear(960, num_classes)

    def forward(self, x):
        net = self.conv1(x)
        net = F.avg_pool2d(net, 3, stride=2, padding=1)
        net = self.stage2(net)
        net = self.stage3(net)
        net = self.stage4(net)
        net = F.avg_pool2d(net, 7)
        net = net.view(net.size(0), -1)
        net = self.fc(net)
        logits = F.softmax(net)
        return logits


""" subLSTM """


class subLSTM(torch.nn.Module):
    def __init__(self, input_size=28 * 28, state_size=128):
        super(subLSTM, self).__init__()
        self.input_size = input_size
        self.state_size = state_size
        self.weight_ih = torch.nn.Parameter(torch.Tensor(4 * state_size, input_size))
        self.weight_hh = torch.nn.Parameter(torch.Tensor(4 * state_size, state_size))
        self.bias_ih = torch.nn.Parameter(torch.Tensor(4 * state_size))
        self.bias_hh = torch.nn.Parameter(torch.Tensor(4 * state_size))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.state_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, +stdv)

    def forward(self, input, state):
        old_h, old_c = state
        gates = F.linear(input, self.weight_ih, self.bias_ih) + F.linear(
            old_h, self.weight_hh, self.bias_hh
        )
        gates = torch.sigmoid(gates)
        in_gate, forget_gate, cell_gate, out_gate = gates.chunk(4, dim=1)

        new_c = forget_gate * old_c + cell_gate - in_gate
        new_h = torch.sigmoid(new_c) - out_gate

        return new_h, new_c


class RnnsubLSTM(nn.Module):
    def __init__(self, in_dim=28 * 28, hidden_dim=128, n_class=10):
        super(RnnsubLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = subLSTM(in_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, n_class)
        self.hx = (
            torch.zeros(batch_size, self.hidden_dim),
            torch.zeros(batch_size, self.hidden_dim),
        )

    def forward(self, x):
        # if self.hx is None:
        #     zeros = Variable(torch.zeros(batch_size, self.hidden_dim,dtype=x.dtype, device=x.device))
        #     self.hx = (zeros, zeros)
        new_h, new_c = self.lstm(x, self.hx)
        # self.hx = (Variable(new_h), Variable(new_c))
        out = self.classifier(new_h)
        return out


# TODO: BERT


MODEL_FACTORY = {
    # model-name: (model-class, input-shapes),
    "milstm": (MILSTM, [(batch_size, 28 * 28)]),
    "scrnn": (SCRNN, [(batch_size, 28 * 28)]),
    "capsule": (CapsuleNetwork, [(batch_size, 1, 28, 28)]),
    "lltm": (RnnLLTM, [(batch_size, 28 * 28)]),
    "shufflenet": (ShuffleNet, [(batch_size, 3, 224, 224)]),
    "sublstm": (RnnsubLSTM, [(batch_size, 28 * 28)]),
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--trial", type=int, default=2000)
    args = parser.parse_args()

    model_name = args.model
    assert model_name in MODEL_FACTORY, f"Unsupported model: {model_name}"

    ModelClass, input_shapes = MODEL_FACTORY[model_name]
    model = ModelClass().eval()  # initialize model on CPU for tracing only
    bench_network(model, input_shapes, model_name, n_trial=args.trial)
