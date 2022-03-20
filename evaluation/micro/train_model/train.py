import torch
import torch.nn as nn
import numpy as np
import json


def load_dataset():
    data = []
    ls = []
    ps = []
    rs = []
    costs = []
    for name in ["F1", "F3"]:
        with open(
            f"../gen_data/dataset_attention_batch_gemm_chain_{name}.csv", "r"
        ) as fin:
            fin.readline()
            for line in fin:
                (
                    fid,
                    valid,
                    Tm,
                    Tn,
                    Tk,
                    Tl,
                    locality,
                    parallelism,
                    recomputation,
                    cost,
                ) = line.strip().split(",")
                fid = int(fid)
                valid = bool(valid)
                Tm = int(Tm)
                Tn = int(Tn)
                Tk = int(Tk)
                Tl = int(Tl)
                locality = float(locality)
                parallelism = float(parallelism)
                recomputation = float(recomputation)
                cost = float(cost)
                if locality == -float("inf") or cost >= 1e5:
                    continue
                ls.append(locality)
                ps.append(parallelism)
                rs.append(recomputation)
                costs.append(cost)

    def normalize(lst):
        # lst = np.log(lst)
        max_v = np.max(lst)
        min_v = np.min(lst)
        print(min(lst), max(lst), np.mean(lst))
        lst = (lst - min_v) / (max_v + 1e-15)
        return lst

    ls = normalize(ls)
    ps = normalize(ps)
    rs = normalize(rs)
    # costs = normalize(costs)
    print(type(costs))
    costs = np.array(costs) * 100
    data = list(zip(ls, ps, rs, costs))
    return data


def split_train_test_data(dataset: list, fold=0.2):
    datasize = len(dataset)
    num_test = int(datasize * fold)
    num_train = datasize - num_test
    np.random.shuffle(dataset)
    return dataset[:num_train], dataset[:num_test]


class Batcher(object):
    def __init__(self, dataset, batch=16):
        self._datasize = len(dataset)
        self._iters = (self._datasize + batch - 1) // batch
        self._data = []
        for i in range(self._iters):
            sliced = torch.FloatTensor(dataset[i * batch : (i + 1) * batch])
            self._data.append((sliced[:, 0:3], sliced[:, 3:4]))
        self._ptr = 0

    def __iter__(self):
        self._ptr = 0
        # np.random.shuffle(self._data)
        return self

    def __next__(self):
        if self._ptr < self._iters:
            self._ptr += 1
            return self._data[self._ptr - 1]
        else:
            raise StopIteration


class PerfModel(nn.Module):
    def __init__(self):
        super(PerfModel, self).__init__()
        self.l1 = nn.Linear(3, 128)
        self.l2 = nn.Linear(128, 128)
        self.pred = nn.Linear(128, 1)

    def forward(self, x):
        return torch.relu(self.pred(torch.relu(self.l2(torch.relu(self.l1(x))))))


def cal_rank_acc(pred: list, label: list):
    assert len(pred) == len(label)
    acc = 0
    total = 0
    for i in range(len(pred)):
        for j in range(i + 1, len(pred)):
            total += 1
            if (abs(label[i] - label[j]) < 1e-5):
                acc += int((pred[i] - pred[j]) < 1e-5)
            else:
                acc += int((pred[i] - pred[j]) * (label[i] - label[j]) > 0)
    return float(acc) / float(total)


def cal_abs_acc(pred: list, label: list):
    assert len(pred) == len(label)
    acc = 0
    total = 0
    for i in range(len(pred)):
        total += 1
        acc += int(abs((pred[i] - label[i]) / label[i]) < 1e-3)
    return float(acc) / float(total)


def get_tri_mask(size):
    mask = [[int(i > j) for j in range(size)] for i in range(size)]
    return torch.FloatTensor(mask)


def rank_loss(pred, label, mask):
    # pred = pred.squeeze()
    # label = label.squeeze()
    diff = pred - pred.T
    target = (label - label.T)
    target_sign = target.sign()
    # print(diff)
    # print(target)
    # print(mask)
    return torch.sum(torch.max(torch.zeros_like(diff), target_sign * (-diff + target) * (-diff + target) * mask)) # + torch.sum((pred - label) * (pred - label))


def train():
    datasets = load_dataset()
    trainset, testset = split_train_test_data(datasets, fold=0.2)
    # _, testset = split_train_test_data(datasets, fold=1.0)

    batcher = Batcher(trainset, batch=4)
    tester = Batcher(testset, batch=len(testset))

    model = PerfModel()
    torch.nn.init.ones_(model.l1.weight)
    torch.nn.init.ones_(model.pred.weight)
    # torch.nn.init.xavier_uniform(model.pred.weight)
    model.train()
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for iter in range(1000):
        for batch_inp, batch_label in batcher:
            optimizer.zero_grad()
            preds = model(batch_inp)
            # print(preds)
            mask = get_tri_mask(preds.shape[0])
            loss = loss_fn(preds, batch_label)
            # loss = rank_loss(preds, batch_label, mask)
            loss.backward()
            optimizer.step()
        if iter % 100 == 0:
            print(loss.item())

    # test
    model.eval()
    for inp, label in tester:
        preds = model(inp)
        print(preds)
        rank_acc = cal_rank_acc(
            preds.squeeze().detach().numpy().tolist(),
            label.squeeze().detach().numpy().tolist(),
        )
        abs_acc = cal_abs_acc(
            preds.squeeze().detach().numpy().tolist(),
            label.squeeze().detach().numpy().tolist(),
        )
        print("Rank accuracy is", rank_acc)
        print("Absolute accuracy is", abs_acc)

    weights = {
        "l1": model.l1.weight.detach().numpy().tolist(),
        "pred": model.pred.weight.detach().numpy().tolist(),
    }
    string = json.dumps(weights)
    with open("weights.json", "w") as fout:
        fout.write(string)


if __name__ == "__main__":
    train()
