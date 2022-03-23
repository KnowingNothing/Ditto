import subprocess
import regex as re
REPEAT = 10000
def test_mkl(batch, M, N, K, L):
    shapeS = [batch, M, N, K, L, REPEAT]
    shapeS = [str(_) for _ in shapeS]
    shapeS = ' '.join(shapeS)
    cmd = "./MKL2MM " + shapeS
    s = subprocess.check_output(cmd.split()).decode('utf-8')
    ratioToPeak = re.findall('ratioToPeak: ([\d\.]*)', s)
    return float(ratioToPeak[0])

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
    subprocess.Popen(["make"]).wait()
    res = []
    for shape in shapes:
        cost = test_mkl(*shape)
        res.append((shape, cost))
    print(res)