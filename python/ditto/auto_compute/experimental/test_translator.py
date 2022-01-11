from translator import *


def test_gemm():
    # A[i, j] = B[i, k] * C[k, j]
    i, j, k = Iter('i', (0, 32)), Iter('j', (0, 32)), Iter('k', (0, 32), True)
    B_access = TensorAccess.new(Tensor('B', [32, 32]), [([(i, 1)], 0), ([(k, 1)], 0)])
    C_access = TensorAccess.new(Tensor('C', [32, 32], True), [k, j])
    A_access = TensorAccess.new(Tensor('A', [32, 32]), [([(i, 1)], 0), ([(j, 1)], 0)])
    compute_expr = ComputeExpr(A_access, operands=[B_access, C_access])
    print(compute_expr)

    stage = Stage(iters=[i, j, k], compute_expr=compute_expr)
    state = State(stages=[stage])

    state, [i0, i1] = state.split(0, i, [4,])
    print(state.stages[0].compute_expr)

    state, __ = state.eliminate(0, i1)
    print(state.stages[0].compute_expr)  # A[1*j, 1*i.0] = B[4*i.0, 1*k] * C[1*k, 1*j]

    module = StageEinsum.from_compute_expr(state.stages[0].compute_expr)
    print(module.einsum_eq)
    for name, param in module.named_parameters():
        print(name, param.shape)

    inputs = {
        k: torch.randn(*shape)
        for k, shape in module.input_cfg.items()
    }
    output = module(**inputs)
    print(output.shape)


def test_local_attention():
    # S[i1, j1] = sum(Q[i1, k1] * K[j1, k1], k1)
    # O[i2, j2] = sum(S[i2, k2] * V[j2, k2], k2)
    
    i1, j1, k1 = Iter('i1', (0, 32)), Iter('j1', (0, 32)), Iter('k1', (0, 16), True)
    Q_access = TensorAccess.new(Tensor('Q', [32, 16]), [([(i1, 1)], 0), ([(k1, 1)], 0)])
    K_access = TensorAccess.new(Tensor('K', [32, 16]), [([(j1, 1)], 0), ([(k1, 1)], 0)])
    S_access1 = TensorAccess.new(Tensor('S', [32, 32], True), [i1, j1])
    gemm1_expr = ComputeExpr(S_access1, operands=[Q_access, K_access])
    stage1 = Stage(iters=[i1, j1, k1], compute_expr=gemm1_expr)

    i2, j2, k2 = Iter('i2', (0, 32)), Iter('j2', (0, 16)), Iter('k2', (0, 32), True)
    S_access2 = TensorAccess.new(S_access1.tensor, [i2, k2])
    V_access = TensorAccess.new(Tensor('V', [16, 32]), [([(j2, 1)], 0), ([(k2, 1)], 0)])
    O_access = TensorAccess.new(Tensor('O', [32, 16]), [([(i2, 1)], 0), ([(j2, 1)], 0)])
    gemm2_expr = ComputeExpr(O_access, operands=[S_access2, V_access])
    stage2 = Stage(iters=[i2, j2, k2], compute_expr=gemm2_expr)

    state = State(stages=[stage1, stage2])

    print(state.stages[0].compute_expr)
    print(state.stages[1].compute_expr)
    print(state.sync_groups)

    state, [i1o, i1i] = state.split(0, i1, [4,])
    state, [j1o, j1i] = state.split(0, j1, [4,])
    print(state.stages[0].compute_expr)
    print(state.stages[1].compute_expr)
    print(state.sync_groups)

    state, shared_it = state.iter_share(0, [i1o, j1o])
    print(state.stages[0].compute_expr)
    print(state.stages[1].compute_expr)
    print(state.sync_groups)

    mod = StateEinsum.from_state(state)
    print(mod.input_cfg)
    inputs = {
        k: torch.randn(*shape)
        for k, shape in mod.input_cfg.items()
    }
    output = mod(**inputs)
    print({k: v.shape for k, v in output.items()})


def test_conv2d():
    # Y[64, 32, 32], X[16, 32, 32]
    # Y[oc, oh, ow] = X[ic, ih, iw] * W[oc, oh, ow, ic, ih, iw]
    oc, oh, ow = Iter('oc', (0, 64)), Iter('oh', (0, 32)), Iter('ow', (0, 32))
    ic, ih, iw = Iter('ic', (0, 16), True), Iter('ih', (0, 32), True), Iter('iw', (0, 32), True)
    
    X_access = TensorAccess.new(
        Tensor('X', [16, 32, 32]), 
        [([(ic, 1)], 0), ([(ih, 1)], 0), ([(iw, 1)], 0)]
    )
    W_access = TensorAccess.new(
        Tensor('W', [64, 32, 32, 16, 32, 32], True), 
        [oc, oh, ow, ic, ih, iw]
    )
    Y_access = TensorAccess.new(
        Tensor('Y', [64, 32, 32]), 
        [([(oc, 1)], 0), ([(oh, 1)], 0), ([(ow, 1)], 0)]
    )
    
    conv_expr = ComputeExpr(Y_access, operands=[X_access, W_access])
    stage = Stage(iters=[oc, oh, ow, ic, ih, iw], compute_expr=conv_expr)
    state = State(stages=[stage])
    print(state.stages[0].compute_expr)

    # 1. h = share(ih, oh), w = share(iw, ow)
    # Y[oc, h, w] = X[ic, h, w] * W[oc, h, w, ic]
    state, h = state.iter_share(0, [ih, oh])
    state, w = state.iter_share(0, [iw, ow])
    print(state.stages[0].compute_expr)

    # 2. kh = swin(), kw = swin()
    # Y[oc, h, w] = X[ic, h+kh, w+2*kw] * W[oc, h, w, ic, kh, kw]
    state, kh = state.sliding_window(0, 0, 1, 3, 1)
    state, kw = state.sliding_window(0, 0, 2, 3, 2)
    print(state.stages[0].compute_expr)

    mod = StateEinsum.from_state(state)
    inputs = {
        k: torch.randn(*shape)
        for k, shape in mod.input_cfg.items()
    }
    output = mod(**inputs)
    print({k: v.shape for k, v in output.items()})


if __name__ == '__main__':
    test_gemm()
    test_local_attention()
    test_conv2d()
