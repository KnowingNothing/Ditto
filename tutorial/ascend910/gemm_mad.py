from tbe import tvm
import ctypes
import sys
ctypes.CDLL("/usr/local/Ascend/ascend-toolkit/5.0.2.alpha001/arm64-linux/atc/lib64/libc_sec.so")
sys.path.append("/usr/local/Ascend/ascend-toolkit/latest/opp/op_impl/built-in/ai_core/tbe")
from impl.util.platform_adapter import tbe as platform_tbe


def ceil(x, factor):
    return (x + factor - 1) // factor


def print_ir(sch):
    """
    print ir for input sch
    :param process: tag
    :param sch: schedule
    :return: IR process
    """
    sch = sch.normalize()
    bounds = tvm.schedule.InferBound(sch)
    stmt = tvm.schedule.ScheduleOps(sch, bounds, True)
    print(stmt)


def tile_axes(sch, op, axis, factors):
    ret = []
    for f in reversed(factors[1:]):
        axis, inner = sch[op].split(axis, factor=f)
        ret.append(inner)
    ret.append(axis)
    return list(reversed(ret))


def gemm(M, N, K):
    MI, NI, KI = 16, 16, 16
    OM = ceil(M, 16)
    OK = ceil(K, 16)
    ON = ceil(N, 16)

    gA = tvm.placeholder([M, K], name="gA", dtype="float16")
    gB = tvm.placeholder([K, N], name="gB", dtype="float16")

    ubA = tvm.compute([M, K], lambda i, j: gA[i, j], name="ubA")  # local.UB
    ubFracA = tvm.compute([OM, K, MI], lambda i1, j, i2: ubA[i1 * MI + i2, j], name="ubFracA")  # local.UB
    l1A = tvm.compute([OM, K, MI], lambda i1, j, i2: ubFracA[i1, j, i2], name="l1A")  # local.L1
    l0A = tvm.compute([OM, OK, MI, KI], lambda i1, k1, i2, k2: l1A[i1, k1 * KI + k2, i2], name="l0A")  # local.L0A

    ubB = tvm.compute([K, N], lambda k, j: gB[k, j], name="ubB")  # local.UB
    ubFracB = tvm.compute([OK, N, KI], lambda k1, j, k2: ubB[k1 * KI + k2, j], name="ubFracB")  # local.UB
    l1B = tvm.compute([OK, N, KI], lambda k1, j, k2: ubFracB[k1, j, k2], name="l1B")  # local.L1
    l0B = tvm.compute([OK, ON, NI, KI], lambda k1, j1, j2, k2: l1B[k1, j1 * NI + j2, k2])  # local.L0B

    rk1 = tvm.reduce_axis([0, OK], name="rk1")
    rk2 = tvm.reduce_axis([0, KI], name="rk2")
    l0C = tvm.compute(
        [ON, OM, MI, NI],
        lambda j1, i1, i2, j2:
            tvm.sum((l0A[i1, rk1, i2, rk2] * l0B[rk1, j1, j2, rk2]).astype("float32"), axis=[rk1, rk2]),
        name="l0c"
    )

    ubC = tvm.compute(
        [ON, OM, MI, NI],
        lambda j1, i1, i2, j2: l0C[j1, i1, i2, j2],
        name="ubC"
    )

    ubCastC = tvm.compute(
        [ON, OM, MI, NI],
        lambda j1, i1, i2, j2: ubC[j1, i1, i2, j2].astype("float16"),
        name="ubCastC"
    )

    ubNdC = tvm.compute(
        [OM * MI, ON * NI],
        lambda i, j: ubCastC[j//NI, i//MI, i%MI, j%NI],
        name="ubNdC"
    )

    gC = tvm.compute(
        [OM * MI, ON * NI],
        lambda i, j: ubNdC[i, j],
        name="gC"
    )

    return [gA, gB], [gC]


def main():
    ins, outs = gemm(1024, 1024, 1024)
    
    gA, gB = ins
    gC, = outs
    sch = tvm.create_schedule(gC.op)

    # get all the stages
    ubNdC = gC.op.input_tensors[0]
    ubCastC = ubNdC.op.input_tensors[0]
    ubC = ubCastC.op.input_tensors[0]
    l0C = ubC.op.input_tensors[0]
    l0A, l0B = l0C.op.input_tensors
    l1A = l0A.op.input_tensors[0]
    l1B = l0B.op.input_tensors[0]
    ubFracA = l1A.op.input_tensors[0]
    ubFracB = l1B.op.input_tensors[0]
    ubA = ubFracA.op.input_tensors[0]
    ubB = ubFracB.op.input_tensors[0]

    # set buffer scope
    ub_buffers = [ubNdC, ubCastC, ubC, ubFracA, ubFracB, ubA, ubB]
    l1_buffers = [l1A, l1B]
    for ub in ub_buffers:
        sch[ub].set_scope("local.UB")
    for l1 in l1_buffers:
        sch[l1].set_scope("local.L1")
    sch[l0A].set_scope("local.L0A")
    sch[l0B].set_scope("local.L0B")
    sch[l0C].set_scope("local.L0C")


    # tiling and fusion
    # these are tunable factors
    m_factors = [16, 4, 16]
    n_factors = [2, 1, 512]

    vec_m_factors = [1, 16]
    vec_n_factors = [32, 16]

    l0_m_factors = [1, 4]
    l0_n_factors = [1, 32]
    l0_k_factors = [4, 8, 2]

    l1_B_k1_factors = [16, 1]
    l1_B_k2_factors = [1, 16]
    l1_B_n_factors = [1, 512]

    l1_A_m1_factors = [1, 4]
    l1_A_m2_factors = [1, 16]
    l1_A_k_factors = [2, 512]

    # first, tile the gC
    i, j = sch[gC].op.axis
    i1, i2, i3 = tile_axes(sch, gC, i, m_factors)
    j1, j2, j3 = tile_axes(sch, gC, j, n_factors)
    sch[gC].reorder(j1, i1, j2, i2, i3, j3)
    global_C_ub_pos = i2
    global_C_l0_pos = j2
    global_A_l1_pos = j2
    global_C_dma_pos = i3
    sch[gC].emit_insn(global_C_dma_pos, "dma_copy")

    # then fuse and tile ubNdC
    sch[ubNdC].compute_at(sch[gC], global_C_ub_pos)
    i, j = sch[ubNdC].op.axis
    i1, i2 = tile_axes(sch, ubNdC, i, vec_m_factors)
    j1, j2 = tile_axes(sch, ubNdC, j, vec_n_factors)
    sch[ubNdC].reorder(i1, j1, i2, j2)
    ub_C_vector_auto_pos = i2
    sch[ubNdC].emit_insn(ub_C_vector_auto_pos, "vector_auto")

    # third, fuse ubCastC
    sch[ubCastC].compute_at(sch[gC], global_C_ub_pos)
    ub_C_dma_pos = sch[ubCastC].op.axis[0]
    sch[ubCastC].emit_insn(ub_C_dma_pos, "dma_copy")

    # forth, fuse ubC
    sch[ubC].compute_inline()

    # fifth, fuse and tile l0C
    sch[l0C].compute_at(sch[gC], global_C_l0_pos)
    n1, m1, m2, n2 = sch[l0C].op.axis
    k1, k2 = sch[l0C].op.reduce_axis
    m11, m12 = tile_axes(sch, l0C, m1, l0_m_factors)
    n11, n12 = tile_axes(sch, l0C, n1, l0_n_factors)
    k11, k12, k13 = tile_axes(sch, l0C, k1, l0_k_factors)
    sch[l0C].reorder(n11, m11, k11, k12, n12, m12, m2, n2, k13, k2)
    l0_A_l0_pos = k12
    l0_B_l0_pos = k12
    l0_B_l1_pos = k11
    l0_mad_pos = n12
    l0_mad_dict = {
                "mad_pattern": 0, # GEMM mode
                "k_outer": [k11, k12]
            }
    sch[l0C].emit_insn(l0_mad_pos, "mad", l0_mad_dict)

    # sixth, fuse l0A and l0B
    sch[l0A].compute_at(sch[l0C], l0_A_l0_pos)
    sch[l0B].compute_at(sch[l0C], l0_B_l0_pos)
    l0_A_dma_pos = sch[l0A].op.axis[0]
    l0_B_dma_pos = sch[l0B].op.axis[0]
    sch[l0A].emit_insn(l0_A_dma_pos, "dma_copy")
    sch[l0B].emit_insn(l0_B_dma_pos, "dma_copy")

    # seventh, fuse and tile l1B
    sch[l1B].compute_at(sch[l0C], l0_B_l1_pos)
    k1, j, k2 = sch[l1B].op.axis
    k11, k12 = tile_axes(sch, l1B, k1, l1_B_k1_factors)
    j1, j2 = tile_axes(sch, l1B, j, l1_B_n_factors)
    k21, k22 = tile_axes(sch, l1B, k2, l1_B_k2_factors)
    sch[l1B].reorder(k11, j1, k21, k12, j2, k22)
    l1_B_ub_pos = k21
    l1_B_dma_pos = k12
    sch[l1B].emit_insn(l1_B_dma_pos, "dma_copy")
    
    # eighth, fuse ubFracB
    sch[ubFracB].compute_at(sch[l1B], l1_B_ub_pos)
    ub_B_vnchwconv_pos = sch[ubFracB].op.axis[-2]
    sch[ubFracB].emit_insn(ub_B_vnchwconv_pos, "vnchwconv")

    # ninth, fuse ubB
    sch[ubB].compute_at(sch[l1B], l1_B_ub_pos)
    ub_B_dma_pos = sch[ubB].op.axis[0]
    sch[ubB].emit_insn(ub_B_dma_pos, "dma_copy")

    # tenth, fuse and tile l1A
    sch[l1A].compute_at(sch[gC], global_A_l1_pos)
    i1, k, i2 = sch[l1A].op.axis
    i11, i12 = tile_axes(sch, l1A, i1, l1_A_m1_factors)
    k1, k2 = tile_axes(sch, l1A, k, l1_A_k_factors)
    i21, i22 = tile_axes(sch, l1A, i2, l1_A_m2_factors)
    sch[l1A].reorder(i11, k1, i21, i12, k2, i22)
    l1_A_ub_pos = i21
    l1_A_dma_pos = i12
    sch[l1A].emit_insn(l1_A_dma_pos, "dma_copy")

    # eleventh, fuse ubFracA
    sch[ubFracA].compute_at(sch[l1A], l1_A_ub_pos)
    ub_A_vnchwconv_pos = sch[ubFracA].op.axis[-2]
    sch[ubFracA].emit_insn(ub_A_vnchwconv_pos, "vnchwconv")

    # twelfth, fuse ubA
    sch[ubA].compute_at(sch[l1A], l1_A_ub_pos)
    ub_A_dma_pos = sch[ubA].op.axis[0]
    sch[ubA].emit_insn(ub_A_dma_pos, "dma_copy")


    print_ir(sch)


    # print(tvm.lower(sch, [gA, gB, gC], simple_mode=True))

    # tvm.build(sch, [gA, gB, gC], "cce")
    sch.cce_special = {}
    # spec_node_list
    sch.cce_special["tensor_list"] = [gA, gB, gC]
    # the origin out tensor list
    sch.cce_special["orign_out_tensor"] = [gC]
    # the real out tensor list
    sch.cce_special["real_out_tensor"] = [gC]
    # func = tvm.build(sch, [A, B, gemm], "cce")
    platform_tbe.build(sch, {"print_ir": True, "name": "Gemm", "tensor_list": [gA, gB, gC]})


if __name__ == "__main__":
    main()

