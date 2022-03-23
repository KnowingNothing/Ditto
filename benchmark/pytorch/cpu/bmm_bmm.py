import time
import torch 
import numpy as np

PARALLEL = 20
REPEAT = 10000

def test_torch(batch, M, N, K, L):
    num = REPEAT
    per = 100
    num = ((num-1) // per + 1) * per
    cost = 0
    for outest in range(num // per + 1):
        Q_np = []
        K_np = []
        V_np = []
        Qtensor = []
        Ktensor = []
        Vtensor = []
        QKV_np = []
        for trial in range(per):
            Q_np.append(np.random.uniform(
                size=(batch, M, K)).astype(np.float32))
            K_np.append(np.random.uniform(
                size=(batch, K, L)).astype(np.float32))
            V_np.append(np.random.uniform(
                size=(batch, L, N)).astype(np.float32))
            Qtensor.append(torch.from_numpy(Q_np[trial]))
            Ktensor.append(torch.from_numpy(K_np[trial]))
            Vtensor.append(torch.from_numpy(V_np[trial]))
            QKV_np.append(np.random.uniform(
                size=(batch, M, N)).astype(np.float32))
            for i in range(batch):
                QKV_np[trial][i] = Q_np[trial][i].dot(
                    K_np[trial][i]).dot(V_np[trial][i])

        QKV = []

        start = time.time()
        for trial in range(per):
            QK = torch.bmm(Qtensor[trial], Ktensor[trial])
            # QK_relu = torch.nn.ReLU()(QK)
            QKV.append(torch.bmm(QK, Vtensor[trial]))

        end = time.time()
        cost_tmp = (end - start) / per
        if outest > 0:
            cost += cost_tmp
#        print("%d: %g" % (outest, cost_tmp))

        for trial in range(per):
            np.testing.assert_allclose(
                QKV_np[trial], QKV[trial].numpy(), rtol=1e-3)

    cost /= num // per
    wl = batch * ( M * K * L + M * L * N)
    ratioToPeak = wl / PARALLEL / 1e9 / cost / 2.2 / 35.2
    print(ratioToPeak)
    return ratioToPeak

shapes = [
    # (batch, M, N, K, L)
    (8, 512, 512 // 8, 512 // 8, 512),  # Bert-Small
    (12, 512, 768 // 12, 768 // 12, 512),  # Bert-Base
    (16, 512, 1024 // 16, 1024 // 16, 512),  # Bert-Large
    (12, 256, 768 // 12, 768 // 12, 256),  # ViT-Base/14
    (16, 256, 1024 // 16, 1024 // 16, 256),  # ViT-Large/14
    (16, 256, 1280 // 16, 1280 // 16, 256),  # ViT-Huge/14
]

if __name__ == "__main__":
    res = []
    torch.set_num_threads(20)
    for shape in shapes:
        cost = test_torch(*shape)
        res.append((shape, cost))
    print(res)