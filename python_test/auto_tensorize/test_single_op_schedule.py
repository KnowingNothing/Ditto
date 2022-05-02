import time
import tvm
from ditto import auto_tensorize as at
from ditto.hardware.hw_param import hardware_param
from ditto import auto_compute as ac
import numpy as np
import tvm.te as te
import random
import math

MI = 6
NI = 16
KI = 16


def BatchGemm(
    batch=1, M=516, K=64, L=512, in_dtype="float32", acc_dtype="float32"
):
    assert M % MI == 0
    assert K % KI == 0
    assert L % NI == 0
    A = tvm.te.placeholder([batch, M, K], name="A", dtype=in_dtype)
    B = tvm.te.placeholder([batch, K, L], name="B", dtype=in_dtype)

    rko = tvm.te.reduce_axis([0, K // KI], "rko")
    rki = tvm.te.reduce_axis([0, KI], "rki")
    C = tvm.te.compute(
        [batch, M // MI, L // NI, MI, NI],
        lambda b, mo, lo, mi, li: tvm.te.sum(
            A[b, mo * MI + mi, rko * KI + rki].astype(acc_dtype)
            * B[b, rko * KI + rki, lo * NI + li].astype(acc_dtype),
            axis=[rko, rki],
        ),
        name="C",
    )
    return [A, B], [C], [C.op.axis[-2], C.op.axis[-1], rki]


class MicroKernel:
    def __init__(self, N=1, K=4, H=3, W=8, C=8, R=3, S=3) -> None:
        self.N = N
        self.K = K
        self.H = H
        self.W = W
        self.C = C
        self.R = R
        self.S = S

    def RoundUp(self, N, K, H, W, C, R, S):
        def f(a, b): return (a + b - 1) // b * b
        return [f(N, self.N), f(K, self.K), f(H, self.H), f(W, self.W), f(C, self.C), f(R, self.R), f(S, self.S)]

    def Verify(self, N, K, H, W, C, R, S):
        return N % self.N == 0 and K % self.K == 0 and H % self.H == 0 and W % self.W == 0 and R % self.R == 0 and S % self.S == 0


def intrin_conv(mk):
    N_, K_, H_, W_, C_, R_, S_ = mk.N, mk.K, mk.H, mk.W, mk.C, mk.R, mk.S
    assert N_ == 1
    assert K_ == 4
    assert H_ == 3
    assert W_ == 8
    assert R_ == 3
    assert S_ == 3
    In_ = te.placeholder([N_, C_, H_ + R_ - 1, W_ + S_-1],
                         name='in', dtype="float32")
    Weight_ = te.placeholder([K_, C_, R_, S_], name='weight', dtype="float32")
    r_ = tvm.te.reduce_axis((0, R_), name='rr')
    s_ = tvm.te.reduce_axis((0, S_), name='rs')
    co = tvm.te.reduce_axis((0, 1), name='rco')
    c_ = tvm.te.reduce_axis((0, C_), name='rci')
    Out_ = te.compute([1, N_, 1, K_, 1, H_, 1, W_], lambda no, n_, ko, k_, ho, h_, wo, w_:
                      te.sum(In_[n_, c_, h_ + r_, w_+s_] * Weight_[k_, c_, r_, s_], axis=[co, c_, r_, s_]), name='out')
    #  [21119070, 158790, 402, 1],  value [350, 50, 10, 1],
    varnames = ['chrws', 'hrws', 'ws', 'crs', 'rs', 's',
                'khw', 'hw', 'w', '_nkhw', '_khw', '_hw', '_w']
    vars = {varname: tvm.te.var(varname) for varname in varnames}
    In_b = tvm.tir.decl_buffer(In_.shape, In_.dtype, name='In', offset_factor=1, strides=[
                               vars['chrws'], vars['hrws'], vars['ws'], 1])  # N C H+R-1 W+S-1
    Weight_b = tvm.tir.decl_buffer(Weight_.shape, Weight_.dtype, name='Weight', offset_factor=1, strides=[
                                   vars['crs'], vars['rs'], vars['s'], 1])  # K C R S
    Out_b = tvm.tir.decl_buffer(Out_.shape, Out_.dtype, name='Out', offset_factor=1, strides=[
                                vars['_nkhw'], vars['khw'], vars['_khw'], vars['hw'], vars['_hw'], vars['w'], vars['_w'], 1])  # N K H W

    def intrin_func(ins, outs):
        In__, Weight__ = ins
        Out__ = outs[0]

        def _body():
            ib = tvm.tir.ir_builder.create()
            # extern "C" int conv_update_avx2(float *In, float *Weight, float *Out, int HW_s_, int W_s_, int C_in_,
            # int HRWS_s_, int WS_s_, int CRS_s_, int RS_s_, int S_s_){
            ib.emit(
                tvm.tir.call_extern(
                    "int32",
                    "conv_update_avx2",
                    In__.access_ptr('r'),
                    Weight__.access_ptr('r'),
                    Out__.access_ptr('w'),
                    vars['hw'], vars['w'], C_, vars['hrws'], vars['ws'], vars['crs'], vars['rs'], vars['s']
                )
                # extern "C" int conv_update(float *In, float *Weight, float *Out, int KHW, int HW, int W, int CHRWS, int HRWS, int WS, int CRS, int RS, int S,
                # int N_, int K_, int H_, int W_, int C_, int R_, int S_)
                # tvm.tir.call_extern(
                #     "int32",
                #     "conv_update",
                #     In__.access_ptr('r'),
                #     Weight__.access_ptr('r'),
                #     Out__.access_ptr('w'),
                #     vars['khw'], vars['hw'], vars['w'], vars['chrws'], vars['hrws'], vars['ws'], vars['crs'],vars['rs'],vars['s'],C_
                # )
            )
            return ib.get()

        def _reduce_reset():
            # long long conv_reset(float *Out, long long K, long long H, long long W,
            #            long long N_, long long K_, long long H_, long long W_)
            ib = tvm.tir.ir_builder.create()
            ib.emit(tvm.tir.call_extern("int32", "conv_reset", Out__.access_ptr(
                'w'), vars['khw'], vars['hw'], vars['w'], N_, K_, H_, W_))
            return ib.get()

        def _reduce_update():
            return _body()

        return _body(), _reduce_reset(), _reduce_update()
    return te.decl_tensor_intrin(Out_.op, intrin_func, binds={In_: In_b, Weight_: Weight_b, Out_: Out_b})


def test_singleOpSchedule(B=1, M=516, L=512, K=64, config={}, prefix="", ):
    ins, outs, tensorizeAxes = BatchGemm(B, M, K, L)
    layer = ac.layer([outs[0].op], inputs=ins)

    cacheSizes = [32*16, 32 * 1024, 256*1024,
                  35840*1024, 35840 * 1024 * 1024]
    bandwidth = [293.72, 81.72, 38.54,  13.14]
    tensorWeight = [1.0, 2.0]
    searchType = 'normal'
    mode = 'best'
    parallelism = 20

    if 'cacheSizes' in config:
        cacheSizes = config['cacheSizes']
    if 'bandwidth' in config:
        bandwidth = config['bandwidth']
    if 'tensorWeight' in config:
        tensorWeight = config['tensorWeight']
    if 'searchType' in config:
        searchType = config['searchType']
    if 'mode' in config:
        mode = config['mode']
    if 'parallelism' in config:
        parallelism = config['parallelism']

    CPU = hardware_param(
        register_per_processor_kb=-1,
        shared_memory_per_group_kb=-1,
        shared_memory_bandwidth_gbs=-1,
        global_memory_gb=-1,
        global_memory_bandwidth_gbs=-1,
        num_processors_per_group=-1,
        num_groups=parallelism,
        fp32_peak_perf_gflops=-1,
        launch_latency_s=-1,
        cacheSizes=cacheSizes,
        bandwidth=bandwidth,
        tensorWeight=tensorWeight,
        platform="CPU"
    )

    packed = at.cpu_intrin("gemm", (MI, NI, KI), dtype="float32", prefix="gemm1")
    code = kernels_impl('avx2')
    context = at.single_op_schedule(
        outs[0].op, tensorizeAxes, CPU, searchType=searchType, mode=mode)
    ave_ratioToPeak = 0
    for i in range(context.size):
        print(f"step{i}")
        sch = tvm.te.create_schedule(outs[0].op)
        sch = context.run(i, sch, outs[0].op, tensorizeAxes, packed.compute_intrinsic, code,
                          f"/home/CORP.PKUSC.ORG/gulang2020/workspace/Ditto/python_test/auto_tensorize/{prefix}data.txt")
        func = tvm.build(sch, layer.schedule_tensors, name="bmm")
        inputs_np = [
            np.random.uniform(-1, 1, [int(x)
                              for x in y.shape]).astype("float32")
            for y in ins
        ]

        outputs_np = [
            np.random.uniform(-1, 1, [int(x)
                              for x in y.shape]).astype("float32")
            for y in outs
        ]
        ctx = tvm.cpu()
        inputs_tvm = [tvm.nd.array(x, ctx) for x in inputs_np]
        outputs_tvm = [tvm.nd.array(x, ctx) for x in outputs_np]

        evaluator = func.time_evaluator(
            func.entry_name, ctx, min_repeat_ms=100, repeat=10)
        cost = evaluator(*inputs_tvm, *outputs_tvm).mean * 1e3
        computation = B * M * K * L
        ratioToPeak = computation / 1e6 / cost / 35.2 / parallelism
        with open(f'{prefix}res.txt', 'a') as f:
            f.write(
                f'{i} {cost} {computation / 1e6 / cost / 35.2 / parallelism}\n')
        ave_ratioToPeak += ratioToPeak
        print(f"GEMM(B={B},M={M},L={L},K={K},parallelism={parallelism}): ")
        print("cost: ", cost)
        print("ratio to peak: ", ratioToPeak)
    ave_ratioToPeak /= context.size
    return ave_ratioToPeak


def kernels_impl(instructionSet="avx512"):
    cc_code = """
        #include <immintrin.h>
        #include <stdio.h>

        extern "C" int gemm1_update(float *C, float *A, float *B, int K_, int N_, int c_stride_) {
            long long K = K_;
            long long N = N_;
            long long c_stride = c_stride_;
            __asm__(
                // AT&T syntax: src dst 
                "mov %%[A], %%%%rax;"
                "mov %%[B], %%%%rbx;"
                "mov %%[C], %%%%rcx;"
                "mov %%[K], %%%%rsi;"
                "mov %%[N], %%%%rdi;"

                "mov %%[c_stride], %%%%rdx;"
                "xor %%%%r8, %%%%r8;"
                "vmovaps 0(%%%%rcx, %%%%r8, 4), %%%%__regName__4;"
                "vmovaps __nBytePerReg__(%%%%rcx, %%%%r8, 4), %%%%__regName__5;"
                "add %%%%rdx, %%%%r8;"
                "vmovaps 0(%%%%rcx, %%%%r8, 4), %%%%__regName__6;"
                "vmovaps __nBytePerReg__(%%%%rcx, %%%%r8, 4), %%%%__regName__7;"
                "add %%%%rdx, %%%%r8;"
                "vmovaps 0(%%%%rcx, %%%%r8, 4), %%%%__regName__8;"
                "vmovaps __nBytePerReg__(%%%%rcx, %%%%r8, 4), %%%%__regName__9;"
                "add %%%%rdx, %%%%r8;"
                "vmovaps 0(%%%%rcx, %%%%r8, 4), %%%%__regName__10;"
                "vmovaps __nBytePerReg__(%%%%rcx, %%%%r8, 4), %%%%__regName__11;"
                "add %%%%rdx, %%%%r8;"
                "vmovaps 0(%%%%rcx, %%%%r8, 4), %%%%__regName__12;"
                "vmovaps __nBytePerReg__(%%%%rcx, %%%%r8, 4), %%%%__regName__13;"
                "add %%%%rdx, %%%%r8;"
                "vmovaps 0(%%%%rcx, %%%%r8, 4), %%%%__regName__14;"
                "vmovaps __nBytePerReg__(%%%%rcx, %%%%r8, 4), %%%%__regName__15;"

                "mov %%[K], %%%%rdx;"
                "mov $0, %%%%r8;"
                "lea (%%%%r8, %%%%rdx), %%%%r9;"
                "lea (%%%%r9, %%%%rdx), %%%%r10;"
                "lea (%%%%r10, %%%%rdx), %%%%r11;"
                "lea (%%%%r11, %%%%rdx), %%%%r12;"
                "lea (%%%%r12, %%%%rdx), %%%%r13;"

                "mov $%d, %%%%rdx;"
                "test %%%%rdx, %%%%rdx;"
                "jz .store%%=;"
                

            ".compute%%=:"
                "vmovaps 0(%%%%rbx), %%%%__regName__2;"
                "vmovaps __nBytePerReg__(%%%%rbx), %%%%__regName__3;"
                "vbroadcastss 0(%%%%rax, %%%%r8, 4), %%%%__regName__0;"
                "vbroadcastss 0(%%%%rax, %%%%r9, 4), %%%%__regName__1;"
                "vfmadd231ps %%%%__regName__2, %%%%__regName__0, %%%%__regName__4;"
                "vfmadd231ps %%%%__regName__3, %%%%__regName__0, %%%%__regName__5;"
                "vfmadd231ps %%%%__regName__2, %%%%__regName__1, %%%%__regName__6;"
                "vfmadd231ps %%%%__regName__3, %%%%__regName__1, %%%%__regName__7;"
                "vbroadcastss 0(%%%%rax, %%%%r10, 4), %%%%__regName__0;"
                "vbroadcastss 0(%%%%rax, %%%%r11, 4), %%%%__regName__1;"
                "vfmadd231ps %%%%__regName__2, %%%%__regName__0, %%%%__regName__8;"
                "vfmadd231ps %%%%__regName__3, %%%%__regName__0, %%%%__regName__9;"
                "vfmadd231ps %%%%__regName__2, %%%%__regName__1, %%%%__regName__10;"
                "vfmadd231ps %%%%__regName__3, %%%%__regName__1, %%%%__regName__11;"
                "vbroadcastss 0(%%%%rax, %%%%r12, 4), %%%%__regName__0;"
                "vbroadcastss 0(%%%%rax, %%%%r13, 4), %%%%__regName__1;"
                "vfmadd231ps %%%%__regName__2, %%%%__regName__0, %%%%__regName__12;"
                "vfmadd231ps %%%%__regName__3, %%%%__regName__0, %%%%__regName__13;"
                "vfmadd231ps %%%%__regName__2, %%%%__regName__1, %%%%__regName__14;"
                "vfmadd231ps %%%%__regName__3, %%%%__regName__1, %%%%__regName__15;"

                "lea 4(%%%%rax), %%%%rax;"
                "lea 0(%%%%rbx, %%%%rdi, 4), %%%%rbx;"
                "sub $1, %%%%rdx;"
                "jnz .compute%%=;"
                
            ".store%%=:"
                // store result into C
                "mov %%[c_stride], %%%%rdx;"
                "xor %%%%r8, %%%%r8;"
                "vmovaps %%%%__regName__4, 0(%%%%rcx, %%%%r8, 4);"
                "vmovaps %%%%__regName__5, __nBytePerReg__(%%%%rcx, %%%%r8, 4);"
                "add %%%%rdx, %%%%r8;"
                "vmovaps %%%%__regName__6, 0(%%%%rcx, %%%%r8, 4);"
                "vmovaps %%%%__regName__7, __nBytePerReg__(%%%%rcx, %%%%r8, 4);"
                "add %%%%rdx, %%%%r8;"
                "vmovaps %%%%__regName__8, 0(%%%%rcx, %%%%r8, 4);"
                "vmovaps %%%%__regName__9, __nBytePerReg__(%%%%rcx, %%%%r8, 4);"
                "add %%%%rdx, %%%%r8;"
                "vmovaps %%%%__regName__10, 0(%%%%rcx, %%%%r8, 4);"
                "vmovaps %%%%__regName__11, __nBytePerReg__(%%%%rcx, %%%%r8, 4);"
                "add %%%%rdx, %%%%r8;"
                "vmovaps %%%%__regName__12, 0(%%%%rcx, %%%%r8, 4);"
                "vmovaps %%%%__regName__13, __nBytePerReg__(%%%%rcx, %%%%r8, 4);"
                "add %%%%rdx, %%%%r8;"
                "vmovaps %%%%__regName__14, 0(%%%%rcx, %%%%r8, 4);"
                "vmovaps %%%%__regName__15, __nBytePerReg__(%%%%rcx, %%%%r8, 4);"
                :
                :[A] "m" (A),
                [B] "m" (B),
                [C] "m" (C),
                [K] "m" (K),
                [N] "m" (N),
                [c_stride] "m" (c_stride)
                :"rax", "rbx", "rcx", "rdx", "rsi", "rdi", "r8", "r9", "r10", "r11", "r12", "r13", "__regName__0", "__regName__1", "__regName__2", "__regName__3", "__regName__4", "__regName__5", "__regName__6", "__regName__7", "__regName__8", "__regName__9", "__regName__10", "__regName__11", "__regName__12", "__regName__13", "__regName__14", "__regName__15"
            );
            return 0;
        }
        extern "C" int gemm2_update(float *C, float *A, float *B, int K_, int N_, int c_stride_) {
            long long K = K_;
            long long N = N_;
            long long c_stride = c_stride_;
            __asm__(
                // AT&T syntax: src dst 
                "mov %%[A], %%%%rax;"
                "mov %%[B], %%%%rbx;"
                "mov %%[C], %%%%rcx;"
                "mov %%[K], %%%%rsi;"
                "mov %%[N], %%%%rdi;"

                "mov %%[c_stride], %%%%rdx;"
                "xor %%%%r8, %%%%r8;"
                "vmovaps 0(%%%%rcx, %%%%r8, 4), %%%%__regName__4;"
                "vmovaps __nBytePerReg__(%%%%rcx, %%%%r8, 4), %%%%__regName__5;"
                "add %%%%rdx, %%%%r8;"
                "vmovaps 0(%%%%rcx, %%%%r8, 4), %%%%__regName__6;"
                "vmovaps __nBytePerReg__(%%%%rcx, %%%%r8, 4), %%%%__regName__7;"
                "add %%%%rdx, %%%%r8;"
                "vmovaps 0(%%%%rcx, %%%%r8, 4), %%%%__regName__8;"
                "vmovaps __nBytePerReg__(%%%%rcx, %%%%r8, 4), %%%%__regName__9;"
                "add %%%%rdx, %%%%r8;"
                "vmovaps 0(%%%%rcx, %%%%r8, 4), %%%%__regName__10;"
                "vmovaps __nBytePerReg__(%%%%rcx, %%%%r8, 4), %%%%__regName__11;"
                "add %%%%rdx, %%%%r8;"
                "vmovaps 0(%%%%rcx, %%%%r8, 4), %%%%__regName__12;"
                "vmovaps __nBytePerReg__(%%%%rcx, %%%%r8, 4), %%%%__regName__13;"
                "add %%%%rdx, %%%%r8;"
                "vmovaps 0(%%%%rcx, %%%%r8, 4), %%%%__regName__14;"
                "vmovaps __nBytePerReg__(%%%%rcx, %%%%r8, 4), %%%%__regName__15;"

                "mov %%[K], %%%%rdx;"
                "mov $0, %%%%r8;"
                "lea (%%%%r8, %%%%rdx), %%%%r9;"
                "lea (%%%%r9, %%%%rdx), %%%%r10;"
                "lea (%%%%r10, %%%%rdx), %%%%r11;"
                "lea (%%%%r11, %%%%rdx), %%%%r12;"
                "lea (%%%%r12, %%%%rdx), %%%%r13;"

                "mov $%d, %%%%rdx;"
                "test %%%%rdx, %%%%rdx;"
                "jz .store%%=;"
                

            ".compute%%=:"
                "vmovaps 0(%%%%rbx), %%%%__regName__2;"
                "vmovaps __nBytePerReg__(%%%%rbx), %%%%__regName__3;"
                "vbroadcastss 0(%%%%rax, %%%%r8, 4), %%%%__regName__0;"
                "vbroadcastss 0(%%%%rax, %%%%r9, 4), %%%%__regName__1;"
                "vfmadd231ps %%%%__regName__2, %%%%__regName__0, %%%%__regName__4;"
                "vfmadd231ps %%%%__regName__3, %%%%__regName__0, %%%%__regName__5;"
                "vfmadd231ps %%%%__regName__2, %%%%__regName__1, %%%%__regName__6;"
                "vfmadd231ps %%%%__regName__3, %%%%__regName__1, %%%%__regName__7;"
                "vbroadcastss 0(%%%%rax, %%%%r10, 4), %%%%__regName__0;"
                "vbroadcastss 0(%%%%rax, %%%%r11, 4), %%%%__regName__1;"
                "vfmadd231ps %%%%__regName__2, %%%%__regName__0, %%%%__regName__8;"
                "vfmadd231ps %%%%__regName__3, %%%%__regName__0, %%%%__regName__9;"
                "vfmadd231ps %%%%__regName__2, %%%%__regName__1, %%%%__regName__10;"
                "vfmadd231ps %%%%__regName__3, %%%%__regName__1, %%%%__regName__11;"
                "vbroadcastss 0(%%%%rax, %%%%r12, 4), %%%%__regName__0;"
                "vbroadcastss 0(%%%%rax, %%%%r13, 4), %%%%__regName__1;"
                "vfmadd231ps %%%%__regName__2, %%%%__regName__0, %%%%__regName__12;"
                "vfmadd231ps %%%%__regName__3, %%%%__regName__0, %%%%__regName__13;"
                "vfmadd231ps %%%%__regName__2, %%%%__regName__1, %%%%__regName__14;"
                "vfmadd231ps %%%%__regName__3, %%%%__regName__1, %%%%__regName__15;"

                "lea 4(%%%%rax), %%%%rax;"
                "lea 0(%%%%rbx, %%%%rdi, 4), %%%%rbx;"
                "sub $1, %%%%rdx;"
                "jnz .compute%%=;"
                
            ".store%%=:"
                // store result into C
                "mov %%[c_stride], %%%%rdx;"
                "xor %%%%r8, %%%%r8;"
                "vmovaps %%%%__regName__4, 0(%%%%rcx, %%%%r8, 4);"
                "vmovaps %%%%__regName__5, __nBytePerReg__(%%%%rcx, %%%%r8, 4);"
                "add %%%%rdx, %%%%r8;"
                "vmovaps %%%%__regName__6, 0(%%%%rcx, %%%%r8, 4);"
                "vmovaps %%%%__regName__7, __nBytePerReg__(%%%%rcx, %%%%r8, 4);"
                "add %%%%rdx, %%%%r8;"
                "vmovaps %%%%__regName__8, 0(%%%%rcx, %%%%r8, 4);"
                "vmovaps %%%%__regName__9, __nBytePerReg__(%%%%rcx, %%%%r8, 4);"
                "add %%%%rdx, %%%%r8;"
                "vmovaps %%%%__regName__10, 0(%%%%rcx, %%%%r8, 4);"
                "vmovaps %%%%__regName__11, __nBytePerReg__(%%%%rcx, %%%%r8, 4);"
                "add %%%%rdx, %%%%r8;"
                "vmovaps %%%%__regName__12, 0(%%%%rcx, %%%%r8, 4);"
                "vmovaps %%%%__regName__13, __nBytePerReg__(%%%%rcx, %%%%r8, 4);"
                "add %%%%rdx, %%%%r8;"
                "vmovaps %%%%__regName__14, 0(%%%%rcx, %%%%r8, 4);"
                "vmovaps %%%%__regName__15, __nBytePerReg__(%%%%rcx, %%%%r8, 4);"
                :
                :[A] "m" (A),
                [B] "m" (B),
                [C] "m" (C),
                [K] "m" (K),
                [N] "m" (N),
                [c_stride] "m" (c_stride)
                :"rax", "rbx", "rcx", "rdx", "rsi", "rdi", "r8", "r9", "r10", "r11", "r12", "r13", "__regName__0", "__regName__1", "__regName__2", "__regName__3", "__regName__4", "__regName__5", "__regName__6", "__regName__7", "__regName__8", "__regName__9", "__regName__10", "__regName__11", "__regName__12", "__regName__13", "__regName__14", "__regName__15"
            );
            return 0;
        }
        extern "C" int gemm1_reset(float *cc, int stride) {
            #pragma unroll
            for (int i = 0; i < %d; ++i) 
                #pragma unroll
                for (int j = 0; j < %d; ++j) {
                    cc[i*stride+j] = 0.0;
                }
            return 0;
        }
        extern "C" int gemm2_reset(float *cc, int stride) {
            #pragma unroll
            for (int i = 0; i < %d; ++i) 
                #pragma unroll
                for (int j = 0; j < %d; ++j) {
                    cc[i*stride+j] = 0.0;
                }
            return 0;
        }

    """ % (KI, KI, MI, NI, MI, NI)
    if instructionSet == "avx2":
        cc_code = cc_code.replace("__regName__", "ymm")
        cc_code = cc_code.replace("__nBytePerReg__", "32")
    elif instructionSet == "avx512":
        cc_code = cc_code.replace("__regName__", "zmm")
        cc_code = cc_code.replace("__nBytePerReg__", "64")
    from tvm.contrib import utils, clang

    temp = utils.tempdir()
    ll_path = temp.relpath("temp.ll")
    # Create LLVM ir from c source code
    ll_code = clang.create_llvm(cc_code, output=ll_path, options=[
                                "-mavx", "-mavx512f", "-msse"])
    return ll_code


def conv(
    N=1,
    C0=128,
    P0=384,
    Q0=384,
    C1=64,
    R1=3,
    S1=3,
    padding1=1,
    stride1=1,
    in_dtype="float32",
    mk1=MicroKernel(),
):
    '''
            N1 K4 H3 W8 C1 R3 S3
    Conv1   N C1 P0 Q0 C0 R1 S1
    Conv1   N C2 P1 Q1 C1 R2 S2 
    '''
    P1 = (P0 + 2 * padding1 - R1) // stride1 + 1
    Q1 = (Q0 + 2 * padding1 - S1) // stride1 + 1
    assert C1 % mk1.K == 0
    assert P0 % mk1.H == 0
    assert Q0 % mk1.W == 0
    assert C0 % mk1.C == 0

    Img = tvm.te.placeholder([N, C0, P0, Q0], dtype=in_dtype, name="Img")
    Weight1 = tvm.te.placeholder(
        [C1, C0, R1, S1], dtype=in_dtype, name="Weight1")

    Pad1 = tvm.te.compute([N, C0, P0 + 2 * padding1, Q0 + 2 * padding1],
                          lambda n, c, p, q: tvm.tir.if_then_else(tvm.tir.all(
                              p >= padding1, p < P0+padding1, q >= padding1, q < Q0 + padding1), Img[n, c, p-padding1, q-padding1], tvm.tir.const(0, in_dtype)),
                          name="pad1"
                          )

    r1 = tvm.te.reduce_axis((0, R1), name='rr1')
    s1 = tvm.te.reduce_axis((0, S1), name='rs1')
    co1 = tvm.te.reduce_axis((0, C0 // mk1.C), name='rco1')
    ci1 = tvm.te.reduce_axis((0, mk1.C), name='rci1')
    Conv1 = tvm.te.compute([N//mk1.N, mk1.N, C1//mk1.K, mk1.K, P1//mk1.H, mk1.H, Q1//mk1.W, mk1.W],
                           lambda no1, ni1, ko1, ki1, ho1, hi1, wo1, wi1: tvm.te.sum(
        Pad1[no1 * mk1.N + ni1, co1 * mk1.C + ci1, ho1 * mk1.H + hi1 + r1, wo1 * mk1.W + wi1 + s1] *
        Weight1[ko1 * mk1.K + ki1, co1 * mk1.C + ci1, r1, s1], axis=[co1, ci1, r1, s1]),
        name='conv1'
    )
    Conv1TensorizeAxes = [Conv1.op.axis[1], Conv1.op.axis[3],
                          Conv1.op.axis[5], Conv1.op.axis[7], ci1, r1, s1]
    # return (N,C0,P0,Q0,C1,R1,S1,C2,R2,S2,P1,Q1,P2,Q2),(Img, Weight1, Weight2), (Pad1, Conv1, Conv1_rfact, Conv1_relu, Pad2, Conv2)
    return [Img, Weight1], [Conv1], Conv1TensorizeAxes


def cal_rank_acc(pred: list, label: list):
    assert len(pred) == len(label)
    acc = 0
    total = 0
    for i in range(len(pred)):
        for j in range(i + 1, len(pred)):
            total += 1
            if abs(label[i] - label[j]) < 1e-5:
                acc += int((pred[i]-pred[j]) < 1e-5)
            else:
                acc += int((pred[i] - pred[j]) * (label[i] - label[j]) > 0)
    return float(acc) / float(total)


index = 0

ConvData = [[1, 32, 114, 112, 32, 3, 3],
            [1, 128, 57, 56, 128, 3, 3],
            [1, 256, 30, 32, 256, 3, 3],
            [1, 512, 15, 16, 512, 3, 3],
            [1, 1024, 9, 8, 1024, 3, 3],
            [1, 64, 57, 56, 64, 3, 3],
            [1, 128, 30, 32, 128, 3, 3],
            [1, 256, 15, 16, 256, 3, 3],
            [1, 512, 9, 8, 512, 3, 3],
            [1, 32, 546, 544, 8, 3, 3],
            [1, 64, 273, 272, 32, 3, 3],
            [1, 128, 138, 136, 64, 3, 3],
            [1, 256, 69, 72, 128, 3, 3],
            [1, 512, 36, 40, 256, 3, 3],
            [1, 1024, 18, 24, 512, 3, 3]]
GEMMData = [
    # (1,  258,  256,  256),
    # (1, 2052,   32,  128),
    # (1, 1026,  256,   32),
    #(1,  132, 1024, 2048),
    (1,  516,  512,   64)]


def time_cost_gemm(param, parallelism=20, prefix=""):
    cacheSizes, bandwidth, tensorWeight = param[:3], param[3:6], param[6:]
    cacheSizes = [32 * 16] + cacheSizes
    bandwidth = [293.0] + bandwidth

    t0 = time.time()
    gemm_cost = 0
    gemm_valid = 0
    for shape in GEMMData:
        B, M, L, K = shape
        B = parallelism
        try:
            cost = test_singleOpSchedule(
                B, M, L, K, "", {'cacheSizes': cacheSizes, 'bandwidth': bandwidth, 'tensorWeight': tensorWeight, "searchType": 'normal', 'mode': 'best', 'parallelism': parallelism})
            gemm_cost += cost
            gemm_valid += 1
        except:
            print(f"error with data{shape}")
    gemm_cost /= gemm_valid
    print(f"gemm cost is {gemm_cost}")
    t1 = time.time()
    print(f"test takes {t1-t0} time")
    tmp = ['%.2f' % _ for _ in param]
    print('param: ', ' '.join(tmp), 'cost: ', gemm_cost)
    return gemm_cost


def sas():
    CONVR1S1Data = [[1, 516, 64, 512],
                    [1, 132, 128, 128],
                    [1, 258, 256, 256],
                    [1, 516, 512, 512],
                    [1, 1026, 1024, 1024],
                    [1, 2052, 2048, 2048],
                    [1, 132, 1024, 2048],
                    [1, 2052, 32, 128],
                    [1, 1026, 256, 32],
                    [1, 516, 64, 16],
                    [1, 36, 1024, 32],
                    [1, 1764, 128, 1760],
                    [1, 7860, 64, 2560],
                    [1, 5124, 9136, 2560],
                    [1, 3072, 128, 1024],
                    [1, 5124, 9136, 2560],
                    [1, 3072, 128, 1024],
                    [1, 786, 512, 2304],
                    [1, 198, 1024, 4608]]
    # 20077.01 262144.00 150453156.18 188.55 245.84 1.90 1.60 10.92
    cacheSizes_ = [32 * 1024, 256*1024,
                   35840*1024]
    bandwidth_ = [81.72, 38.54,  13.14]
    tensorWeight_ = [1.0, 2.0]
    param = cacheSizes_ + bandwidth_ + tensorWeight_
    # performance = []
    # for _ in range(len(ConvData) + len(GEMMData)):
    #     performance.append([])

    def rank_acc(param, prefix=""):
        cacheSizes, bandwidth, tensorWeight = param[:3], param[3:5], param[5:]
        cacheSizes = [32 * 16] + cacheSizes
        bandwidth = [293.0] + bandwidth
        x, y = [], []
        gemm_cost = 0
        conv_cost = 0
        gemm_valid = 0
        conv_valid = 0
        config = {
            'cacheSizes': cacheSizes,
            'bandwidth': bandwidth,
            'tensorWeight': tensorWeight,
            'searchType': 'normal',
        }
        global index
        for idx in range(len(ConvData)):
            prefix = f'rank_acc_{index}'
            shape = ConvData[idx]
            try:
                _, data_path, res_path = test_singleOpScheduleConv(
                    *shape, prefix=prefix, config=config)
                conv_valid += 1
            except:
                print(f"conv data {idx} error")
                continue
            with open(data_path, 'r') as f:
                for line in f.readlines():
                    line = [float(i) for i in line.split()]
                    x.append(sum(line[1:]))
            with open(res_path, 'r') as f:
                for line in f.readlines():
                    y.append(float(line.split()[-1]))
            if len(x) < len(y):
                y = y[:len(x)]
            else:
                x = x[:len(y)]
            conv_cost += cal_rank_acc(x, y)
            index += 1
        conv_cost /= conv_valid
        tmp = ['%.2f' % _ for _ in param]
        print('param: ', ' '.join(tmp), 'cost: ', conv_cost)
        with open("./result/performance.txt", 'a') as f:
            f.write('param: ')
            f.write(' '.join(tmp))
            f.write('cost: ')
            f.write('%.4f' % conv_cost)
            f.write('\n')
        return conv_cost

    def time_cost(param, prefix=""):
        cacheSizes, bandwidth, tensorWeight = param[:3], param[3:5], param[5:]
        cacheSizes = [32 * 16] + cacheSizes
        bandwidth = [293.0] + bandwidth
        ave_cost = 0

        valid = 0
        t0 = time.time()
        conv_cost = 0
        for idx in range(len(ConvData)):
            try:
                shape = ConvData[idx]
                cost = test_singleOpScheduleConv(
                    *shape, "", {'cacheSizes': cacheSizes, 'bandwidth': bandwidth, 'tensorWeight': tensorWeight, "searchType": 'normal'})
                ave_cost += cost
                conv_cost += cost
                performance[idx].append(cost)
                valid += 1
            except:
                print(f"error with data{idx}")
        print(f"conv cost is {conv_cost}")
        gemm_cost = 0
        gemm_valid = 0
        for idx in range(len(GEMMData)):
            try:
                shape = GEMMData[idx]
                cost = test_singleOpScheduleConv(
                    *shape, "", {'cacheSizes': cacheSizes, 'bandwidth': bandwidth, 'tensorWeight': tensorWeight, "searchType": 'best'})
                ave_cost += cost
                gemm_cost += cost
                performance[idx + len(ConvData)].append(cost)
                valid += 1
                gemm_valid += 1
            except:
                print(f"error with data{idx}")
        gemm_cost /= gemm_valid
        print(f"gemm cost is {conv_cost}")
        t1 = time.time()
        print(f"test takes {t1-t0} time")
        ave_cost /= valid
        tmp = ['%.2f' % _ for _ in param]
        print('param: ', ' '.join(tmp), 'cost: ', ave_cost)
        return ave_cost

    eval = time_cost_gemm
    E = eval(param=param)  # time_cost(param=param)
    best_param = param
    best_E = 0
    for k in np.arange(0, 1, 0.05):
        T = 1 - k
        new_param = param.copy()
        i = random.randint(0, len(param)-1)
        for fac in [1 - math.sqrt(T) * 0.5, 1 / (1 - math.sqrt(T) * 0.5)]:
            param_ = param.copy()
            param_[i] *= fac
            E_ = eval(param=param_)
            if math.exp(-(E-E_) / T) > random.random():
                new_param = param_.copy()
                E = E_
            if best_E < E_:
                best_E = E_
                best_param = param_.copy()
        param = new_param
    with open("./result/param.txt", 'a') as f:
        f.write(str(best_E))
        f.write(' ')
        f.write(' '.join([str(i) for i in best_param]))
        f.write('\n')
    # with open("./result/performace.txt", 'a') as f:
    #     for line in performance:
    #         f.write(' '.join(line))
    #         f.write('\n')


def test_singleOpScheduleConv(N, C1, P0, Q0, C0, R1, S1, prefix="", config={}):
    mk = MicroKernel()
    ins, outs, tensorizeAxes = conv(N, C0, P0, Q0, C1, R1, S1, mk1=mk)
    layer = ac.layer([outs[0].op], inputs=ins)
    cacheSizes = [32*16, 32 * 1024, 256*1024,
                  35840*1024, 35840 * 1024 * 1024]
    bandwidth = [293.72, 81.72, 38.54,  13.14]
    tensorWeight = [1.0, 2.0]
    searchType = "stochastic"
    if 'cacheSizes' in config:
        cacheSizes = config['cacheSizes']
    if 'bandwidth' in config:
        bandwidth = config['bandwidth']
    if 'tensorWeight' in config:
        tensorWeight = config['tensorWeight']
    if 'searchType' in config:
        searchType = config['searchType']
    CPU = hardware_param(
        register_per_processor_kb=-1,
        shared_memory_per_group_kb=-1,
        shared_memory_bandwidth_gbs=-1,
        global_memory_gb=-1,
        global_memory_bandwidth_gbs=-1,
        num_processors_per_group=-1,
        num_groups=20,
        fp32_peak_perf_gflops=-1,
        launch_latency_s=-1,
        cacheSizes=cacheSizes,
        bandwidth=bandwidth,
        tensorWeight=tensorWeight,
        platform="CPU"
    )
    intrin = intrin_conv(mk)
    code = open('./sconvKernel').read()
    context = at.single_op_schedule(outs[0].op, tensorizeAxes, CPU, searchType)
    ave_gflops = 0
    data_path = f"/home/CORP.PKUSC.ORG/gulang2020/workspace/Ditto/python_test/auto_tensorize/result/{prefix}conv_data.txt"
    res_path = f'./result/{prefix}conv_res.txt'
    for i in range(context.size):
        sch = tvm.te.create_schedule(outs[0].op)
        sch = context.run(i, sch, outs[0].op, tensorizeAxes, intrin, code,
                          data_path)
        func = tvm.build(sch, layer.schedule_tensors, name="bmm")
        inputs_np = [
            np.random.uniform(-1, 1, [int(x)
                              for x in y.shape]).astype("float32")
            for y in ins
        ]

        outputs_np = [
            np.random.uniform(-1, 1, [int(x)
                              for x in y.shape]).astype("float32")
            for y in outs
        ]
        ctx = tvm.cpu()
        inputs_tvm = [tvm.nd.array(x, ctx) for x in inputs_np]
        outputs_tvm = [tvm.nd.array(x, ctx) for x in outputs_np]

        evaluator = func.time_evaluator(
            func.entry_name, ctx, min_repeat_ms=100, repeat=10)
        cost = evaluator(*inputs_tvm, *outputs_tvm).mean * 1e3
        computation = N*C0*C1*P0*Q0*R1*S1
        with open(res_path, 'a') as f:
            f.write(
                f'{i} {computation / 1e6 / cost / 35.2} {35.2 * cost * 1e6 / computation}\n')
        ave_gflops += computation / 1e6 / cost / 35.2
    ave_gflops /= context.size
    return ave_gflops, data_path, res_path


if __name__ == "__main__":
    # for B in [1,2,4,8]:
    #     test_singleOpScheduleConv(B = B, prefix = "2")
    # sas()
    # t0 = time.time()
    # test_singleOpScheduleConv(1, 512, 36, 32, 256, 3,
    #                           3, prefix="", config={'searchType': 'best'})
    # t1 = time.time()
    # print('time: ', t1 - t0)
    # test_singleOpSchedule(1)
    cacheSizes = [32*16, 32 * 1024, 256*1024,
                  25600 * 1024 * 2]
    bandwidth = [293.00, 81.55, 38.84, 13]
    tensorWeight = [1.0, 2.0]
    time_cost_gemm(param=cacheSizes[1:] +
                   bandwidth[1:]+tensorWeight, parallelism=20)
