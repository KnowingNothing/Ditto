import pytest
import tvm
import tvm.testing
from ditto import auto_compute as ac
from ditto import auto_tensorize as at
from ditto import hardware as hw
from ditto import utils
import numpy as np
import sys
import time
import tvm.te as te
NI = 1
KI = 4
HI = 3
WI = 16
CI = 32
def conv(N, K, H, W, C, R, S):
    assert N % NI == 0
    assert K % KI == 0
    assert H % HI == 0
    assert W % WI == 0
    assert C % CI == 0
    In = te.placeholder([N, C, H + R - 1, W + S - 1], dtype = "float32")
    Weight = te.placeholder([K, C, R, S], dtype = "float32")
    rco = te.reduce_axis((0, C // CI))
    rci = te.reduce_axis((0, CI))
    rr = te.reduce_axis((0,R)) 
    rs = te.reduce_axis((0,S)) 
    Out = te.compute([N // NI, NI, K // KI, KI, H // HI, HI, W // WI, WI], \
    lambda no, ni, ko, ki, ho, hi, wo, wi: 
       te.sum(In[no*NI + ni, rco * CI + rci, (ho * HI + hi)+ rr, (wo * WI + wi) + rs] * 
            Weight[ko * KI + ki, rco * CI +rci, rr, rs], axis = [rco, rci, rr, rs]))
    
    return [In, Weight], [Out]

def conv_notensorize(N, K, H, W, C, R, S):
    assert N % NI == 0
    assert K % KI == 0
    assert H % HI == 0
    assert W % WI == 0
    assert C % CI == 0
    In = te.placeholder([N, C, H + R - 1, W + S - 1], dtype = "float32")
    Weight = te.placeholder([K, C, R, S], dtype = "float32")
    c = te.reduce_axis((0, C))
    r = te.reduce_axis((0,R)) 
    s = te.reduce_axis((0,S)) 
    print("shape: ", N, K, H, W, C, R, S)
    Out = te.compute([N, K, H, W], \
    lambda n,k,h,w: 
       te.sum(In[n,c,h+r,w+s] * 
            Weight[k,c,r,s], axis = [c, r, s]))
    
    return [In, Weight], [Out]

def intrin_conv(shape, dtype, prefix):
    """
        In = tvm.te.placeholder([N, C, H + R - 1, W + S - 1], dtype=in_dtype, name='In' + surfix)
    Weight = tvm.te.placeholder([K, C, R, S], dtype=in_dtype, name='Weight' + surfix)
    r = tvm.te.reduce_axis((0, R), name='rr' + surfix)
    s = tvm.te.reduce_axis((0, S), name='rs' + surfix)
    c = tvm.te.reduce_axis((0, C), name='rc' + surfix)
    Out = tvm.te.compute([N, K, H, W], lambda n,k,h,w:
        tvm.te.sum(In[n,c,h+r,w+s] * Weight[k,c,r,s], axis=[c,r,s]),
        name = 'Ou' + surfix
    )"""
    assert(len(shape) == 7)
    N_, K_, H_, W_, C_, R_, S_ = shape
    In_ = te.placeholder([N_, C_, H_ + R_ - 1, W_ + S_ - 1], name="in", dtype=dtype)
    Weight_ = te.placeholder([K_, C_, R_, S_], name="weight", dtype=dtype)
    r_ = tvm.te.reduce_axis((0, R_), name="rr")
    s_ = tvm.te.reduce_axis((0, S_), name="rs")
    c_ = tvm.te.reduce_axis((0, C_), name="rci")
    Out_ = te.compute(
        [N_, K_, H_, W_],
        lambda n_, k_, h_, w_: te.sum(
            In_[n_, c_, h_ + r_, w_ + s_] * Weight_[k_, c_, r_, s_],
            axis=[c_, r_, s_],
        ),
        name="out",
    )
    #  [21119070, 158790, 402, 1],  value [350, 50, 10, 1],
    varnames = [
        "chrws",
        "hrws",
        "ws",
        "crs",
        "rs",
        "s",
        "khw",
        "hw",
        "w",
    ]
    vars = {varname: tvm.te.var(varname) for varname in varnames}
    In_b = tvm.tir.decl_buffer(
        In_.shape,
        In_.dtype,
        name="In",
        data_alignment=64,
        offset_factor=1,
        strides=[vars["chrws"], vars["hrws"], vars["ws"], 1],
    )  # N C H+R-1 W+S-1
    Weight_b = tvm.tir.decl_buffer(
        Weight_.shape,
        Weight_.dtype,
        name="Weight",
        data_alignment=64,
        offset_factor=1,
        strides=[vars["crs"], vars["rs"], vars["s"], 1],
    )  # K C R S
    Out_b = tvm.tir.decl_buffer(
        Out_.shape,
        Out_.dtype,
        name="Out",
        data_alignment=64,
        offset_factor=1,
        strides=[
            vars["khw"],
            vars["hw"],
            vars["w"],
            1,
        ],
    )  # N K H W

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
                    prefix + "conv_update_avx2",
                    In__.access_ptr("r"),
                    Weight__.access_ptr("r"),
                    Out__.access_ptr("w"),
                    vars["hw"],
                    vars["w"],
                    C_,
                    vars["hrws"],
                    vars["ws"],
                    vars["crs"],
                    vars["rs"],
                    vars["s"],
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
            ib.emit(
                tvm.tir.call_extern(
                    "int32",
                    prefix + "conv_reset",
                    Out__.access_ptr("w"),
                    vars["khw"],
                    vars["hw"],
                    vars["w"],
                    N_,
                    K_,
                    H_,
                    W_,
                )
            )
            return ib.get()

        def _reduce_update():
            return _body()

        return _body(), _reduce_reset(), _reduce_update()

    return te.decl_tensor_intrin(
        Out_.op, intrin_func, binds={In_: In_b, Weight_: Weight_b, Out_: Out_b}
    )

def conv_impl():
    cc_code = """
#include <unistd.h>
    extern "C" int conv_update_avx2(float *In, float *Weight, float *Out, int HW_s_, int W_s_, int C_in_,
                                int HRWS_s_, int WS_s_, int CRS_s_, int RS_s_, int S_s_)
{
    getpid();
    unsigned long long HW_s = HW_s_;
    unsigned long long W_s = W_s_;
    unsigned long long C_in = C_in_;
    unsigned long long HRWS_s = HRWS_s_;
    unsigned long long WS_s = WS_s_;
    unsigned long long CRS_s = CRS_s_;
    unsigned long long RS_s = RS_s_;
    unsigned long long S_s = S_s_;
    
    __asm__ __volatile__(
        "mov %[W], %%rax;"
        "mov %[HW], %%rbx;"

        "mov %[Out], %%rdi;" // Out[k * HW]
        "vmovups (%%rdi), %%zmm4;"
        "mov %%rax, %%rdx;"
        "vmovups (%%rdi, %%rdx, 4), %%zmm5;"
        "add %%rax, %%rdx;"
        "vmovups (%%rdi, %%rdx, 4), %%zmm6;"

        "lea (%%rdi, %% rbx, 4), %%rdi;"
        "vmovups (%%rdi), %%zmm7;"
        "mov %%rax, %%rdx;"
        "vmovups (%%rdi, %%rdx, 4), %%zmm8;"
        "add %%rax, %%rdx;"
        "vmovups (%%rdi, %%rdx, 4), %%zmm9;"

        "lea (%%rdi, %% rbx, 4), %%rdi;"
        "vmovups (%%rdi), %%zmm10;"
        "mov %%rax, %%rdx;"
        "vmovups (%%rdi, %%rdx, 4), %%zmm11;"
        "add %%rax, %%rdx;"
        "vmovups (%%rdi, %%rdx, 4), %%zmm12;"

        "lea (%%rdi, %% rbx, 4), %%rdi;"
        "vmovups (%%rdi), %%zmm13;"
        "mov %%rax, %%rdx;"
        "vmovups (%%rdi, %%rdx, 4), %%zmm14;"
        "add %%rax, %%rdx;"
        "vmovups (%%rdi, %%rdx, 4), %%zmm15;"

        // calculation
        /*
        for n in [0, 1): "eliminate"
            for c in [0, C):
                for r in [0, 3): # unroll

                    for s in [0, 3): # unroll
                        for h in [0, 3):#unroll
                            zmm_h = In[c*(H+R-1)*(W+S-1)+(h+r)*(W+S-1)+(s)](0:7)
                        for k in [0, 4):#unroll
                            zmm_3 = bdcast(Weight[k*CRS+c*RS+r*S+s])
                            for h in [0, 3): #unroll
                                fma(zmm_h, zmm_3, zmm_{3k+h+4})

        regs:
            values:
                In[c*(H+R-1)*(W+S-1)]                           Base_In_c       %%rax   1
                In[c0*(H+R-1)*(W+S-1)+r*(W+S-1)+s]              Base_In         %%rbx   0
                In[c0*(H+R-1)*(W+S-1)+r0*(W+S-1)+s0+h*(W+S-1)]  hWS             %%rcx   0
                Weight[c*RS]                                    Base_Weight_C   %%rdx   1
                Weight[c0*RS+r*S+s]                             Base_Weight     %%rdi   0
                Weight[c0*RS+r0*S+s0+k*CRS]                     kCRS            %%rsi   0
                c                                                               %%r13   1
            strides:
                HRWS                                                            %%r8    1
                WS                                                              %%r9    1
                CRS                                                             %%r10   1
                RS                                                              %%r11   1
                S                                                               %%r12   1
        */
        "mov %[C_in], %%r13;"
        "mov %[In], %%rax;"
        "mov %[Weight], %%rdx;"
        "mov %[HRWS], %%r8;"
        "mov %[WS], %%r9;"
        "mov %[CRS], %%r10;"
        "mov %[RS], %%r11;"
        "mov %[S], %%r12;"
        ".compute%=:;"

        "mov %%rax, %%rbx;"
        "mov %%rdx, %%rdi;"

        /*cr=0s=0(h)(kh) begin*/
        "mov %%rbx, %%rcx;"
        "vmovups (%%rcx), %%zmm0;"
        "lea (%%rcx, %%r9, 4), %%rcx;"
        "vmovups (%%rcx), %%zmm1;"
        "lea (%%rcx, %%r9, 4), %%rcx;"
        "vmovups (%%rcx), %%zmm2;"

        "mov %%rdi, %%rsi;"
        "vbroadcastss (%%rsi), %%zmm3;"
        "vfmadd231ps %%zmm0, %%zmm3, %%zmm4;"
        "vfmadd231ps %%zmm1, %%zmm3, %%zmm5;"
        "vfmadd231ps %%zmm2, %%zmm3, %%zmm6;"

        "lea (%%rsi, %%r10, 4), %%rsi;"
        "vbroadcastss (%%rsi), %%zmm3;"
        "vfmadd231ps %%zmm0, %%zmm3, %%zmm7;"
        "vfmadd231ps %%zmm1, %%zmm3, %%zmm8;"
        "vfmadd231ps %%zmm2, %%zmm3, %%zmm9;"

        "lea (%%rsi, %%r10, 4), %%rsi;"
        "vbroadcastss (%%rsi), %%zmm3;"
        "vfmadd231ps %%zmm0, %%zmm3, %%zmm10;"
        "vfmadd231ps %%zmm1, %%zmm3, %%zmm11;"
        "vfmadd231ps %%zmm2, %%zmm3, %%zmm12;"

        "lea (%%rsi, %%r10, 4), %%rsi;"
        "vbroadcastss (%%rsi), %%zmm3;"
        "vfmadd231ps %%zmm0, %%zmm3, %%zmm13;"
        "vfmadd231ps %%zmm1, %%zmm3, %%zmm14;"
        "vfmadd231ps %%zmm2, %%zmm3, %%zmm15;"
        /*cr=0s=0(h)(kh) end*/

        /*cr=0s=1(h)(kh) begin*/
        "lea 4(%%rbx), %%rcx;"
        "vmovups (%%rcx), %%zmm0;"
        "lea (%%rcx, %%r9, 4), %%rcx;"
        "vmovups (%%rcx), %%zmm1;"
        "lea (%%rcx, %%r9, 4), %%rcx;"
        "vmovups (%%rcx), %%zmm2;"

        "lea 4(%%rdi), %%rsi;"
        "vbroadcastss (%%rsi), %%zmm3;"
        "vfmadd231ps %%zmm0, %%zmm3, %%zmm4;"
        "vfmadd231ps %%zmm1, %%zmm3, %%zmm5;"
        "vfmadd231ps %%zmm2, %%zmm3, %%zmm6;"

        "lea (%%rsi, %%r10, 4), %%rsi;"
        "vbroadcastss (%%rsi), %%zmm3;"
        "vfmadd231ps %%zmm0, %%zmm3, %%zmm7;"
        "vfmadd231ps %%zmm1, %%zmm3, %%zmm8;"
        "vfmadd231ps %%zmm2, %%zmm3, %%zmm9;"

        "lea (%%rsi, %%r10, 4), %%rsi;"
        "vbroadcastss (%%rsi), %%zmm3;"
        "vfmadd231ps %%zmm0, %%zmm3, %%zmm10;"
        "vfmadd231ps %%zmm1, %%zmm3, %%zmm11;"
        "vfmadd231ps %%zmm2, %%zmm3, %%zmm12;"

        "lea (%%rsi, %%r10, 4), %%rsi;"
        "vbroadcastss (%%rsi), %%zmm3;"
        "vfmadd231ps %%zmm0, %%zmm3, %%zmm13;"
        "vfmadd231ps %%zmm1, %%zmm3, %%zmm14;"
        "vfmadd231ps %%zmm2, %%zmm3, %%zmm15;"
        /*cr=0s=1(h)(kh) end*/

        /*cr=0s=2(h)(kh) begin*/
        "lea 8(%%rbx), %%rcx;"
        "vmovups (%%rcx), %%zmm0;"
        "lea (%%rcx, %%r9, 4), %%rcx;"
        "vmovups (%%rcx), %%zmm1;"
        "lea (%%rcx, %%r9, 4), %%rcx;"
        "vmovups (%%rcx), %%zmm2;"

        "lea 8(%%rdi), %%rsi;"
        "vbroadcastss (%%rsi), %%zmm3;"
        "vfmadd231ps %%zmm0, %%zmm3, %%zmm4;"
        "vfmadd231ps %%zmm1, %%zmm3, %%zmm5;"
        "vfmadd231ps %%zmm2, %%zmm3, %%zmm6;"

        "lea (%%rsi, %%r10, 4), %%rsi;"
        "vbroadcastss (%%rsi), %%zmm3;"
        "vfmadd231ps %%zmm0, %%zmm3, %%zmm7;"
        "vfmadd231ps %%zmm1, %%zmm3, %%zmm8;"
        "vfmadd231ps %%zmm2, %%zmm3, %%zmm9;"

        "lea (%%rsi, %%r10, 4), %%rsi;"
        "vbroadcastss (%%rsi), %%zmm3;"
        "vfmadd231ps %%zmm0, %%zmm3, %%zmm10;"
        "vfmadd231ps %%zmm1, %%zmm3, %%zmm11;"
        "vfmadd231ps %%zmm2, %%zmm3, %%zmm12;"

        "lea (%%rsi, %%r10, 4), %%rsi;"
        "vbroadcastss (%%rsi), %%zmm3;"
        "vfmadd231ps %%zmm0, %%zmm3, %%zmm13;"
        "vfmadd231ps %%zmm1, %%zmm3, %%zmm14;"
        "vfmadd231ps %%zmm2, %%zmm3, %%zmm15;"
        /*cr=0s=2(h)(kh) end*/

        /*cr=1s=0(h)(kh) begin*/
        "lea (%%rbx, %%r9, 4), %%rbx;"
        "mov %%rbx, %%rcx;"
        "vmovups (%%rcx), %%zmm0;"
        "lea (%%rcx, %%r9, 4), %%rcx;"
        "vmovups (%%rcx), %%zmm1;"
        "lea (%%rcx, %%r9, 4), %%rcx;"
        "vmovups (%%rcx), %%zmm2;"

        "lea (%%rdi, %%r12, 4), %%rdi;"
        "mov %%rdi, %%rsi;"
        "vbroadcastss (%%rsi), %%zmm3;"
        "vfmadd231ps %%zmm0, %%zmm3, %%zmm4;"
        "vfmadd231ps %%zmm1, %%zmm3, %%zmm5;"
        "vfmadd231ps %%zmm2, %%zmm3, %%zmm6;"

        "lea (%%rsi, %%r10, 4), %%rsi;"
        "vbroadcastss (%%rsi), %%zmm3;"
        "vfmadd231ps %%zmm0, %%zmm3, %%zmm7;"
        "vfmadd231ps %%zmm1, %%zmm3, %%zmm8;"
        "vfmadd231ps %%zmm2, %%zmm3, %%zmm9;"

        "lea (%%rsi, %%r10, 4), %%rsi;"
        "vbroadcastss (%%rsi), %%zmm3;"
        "vfmadd231ps %%zmm0, %%zmm3, %%zmm10;"
        "vfmadd231ps %%zmm1, %%zmm3, %%zmm11;"
        "vfmadd231ps %%zmm2, %%zmm3, %%zmm12;"

        "lea (%%rsi, %%r10, 4), %%rsi;"
        "vbroadcastss (%%rsi), %%zmm3;"
        "vfmadd231ps %%zmm0, %%zmm3, %%zmm13;"
        "vfmadd231ps %%zmm1, %%zmm3, %%zmm14;"
        "vfmadd231ps %%zmm2, %%zmm3, %%zmm15;"
        /*cr=1s=0(h)(kh) end*/

        /*cr=1s=1(h)(kh) begin*/
        "lea 4(%%rbx), %%rcx;"
        "vmovups (%%rcx), %%zmm0;"
        "lea (%%rcx, %%r9, 4), %%rcx;"
        "vmovups (%%rcx), %%zmm1;"
        "lea (%%rcx, %%r9, 4), %%rcx;"
        "vmovups (%%rcx), %%zmm2;"

        "lea 4(%%rdi), %%rsi;"
        "vbroadcastss (%%rsi), %%zmm3;"
        "vfmadd231ps %%zmm0, %%zmm3, %%zmm4;"
        "vfmadd231ps %%zmm1, %%zmm3, %%zmm5;"
        "vfmadd231ps %%zmm2, %%zmm3, %%zmm6;"

        "lea (%%rsi, %%r10, 4), %%rsi;"
        "vbroadcastss (%%rsi), %%zmm3;"
        "vfmadd231ps %%zmm0, %%zmm3, %%zmm7;"
        "vfmadd231ps %%zmm1, %%zmm3, %%zmm8;"
        "vfmadd231ps %%zmm2, %%zmm3, %%zmm9;"

        "lea (%%rsi, %%r10, 4), %%rsi;"
        "vbroadcastss (%%rsi), %%zmm3;"
        "vfmadd231ps %%zmm0, %%zmm3, %%zmm10;"
        "vfmadd231ps %%zmm1, %%zmm3, %%zmm11;"
        "vfmadd231ps %%zmm2, %%zmm3, %%zmm12;"

        "lea (%%rsi, %%r10, 4), %%rsi;"
        "vbroadcastss (%%rsi), %%zmm3;"
        "vfmadd231ps %%zmm0, %%zmm3, %%zmm13;"
        "vfmadd231ps %%zmm1, %%zmm3, %%zmm14;"
        "vfmadd231ps %%zmm2, %%zmm3, %%zmm15;"
        /*cr=1s=1(h)(kh) end*/

        /*cr=1s=2(h)(kh) begin*/
        "lea 8(%%rbx), %%rcx;"
        "vmovups (%%rcx), %%zmm0;"
        "lea (%%rcx, %%r9, 4), %%rcx;"
        "vmovups (%%rcx), %%zmm1;"
        "lea (%%rcx, %%r9, 4), %%rcx;"
        "vmovups (%%rcx), %%zmm2;"

        "lea 8(%%rdi), %%rsi;"
        "vbroadcastss (%%rsi), %%zmm3;"
        "vfmadd231ps %%zmm0, %%zmm3, %%zmm4;"
        "vfmadd231ps %%zmm1, %%zmm3, %%zmm5;"
        "vfmadd231ps %%zmm2, %%zmm3, %%zmm6;"

        "lea (%%rsi, %%r10, 4), %%rsi;"
        "vbroadcastss (%%rsi), %%zmm3;"
        "vfmadd231ps %%zmm0, %%zmm3, %%zmm7;"
        "vfmadd231ps %%zmm1, %%zmm3, %%zmm8;"
        "vfmadd231ps %%zmm2, %%zmm3, %%zmm9;"

        "lea (%%rsi, %%r10, 4), %%rsi;"
        "vbroadcastss (%%rsi), %%zmm3;"
        "vfmadd231ps %%zmm0, %%zmm3, %%zmm10;"
        "vfmadd231ps %%zmm1, %%zmm3, %%zmm11;"
        "vfmadd231ps %%zmm2, %%zmm3, %%zmm12;"

        "lea (%%rsi, %%r10, 4), %%rsi;"
        "vbroadcastss (%%rsi), %%zmm3;"
        "vfmadd231ps %%zmm0, %%zmm3, %%zmm13;"
        "vfmadd231ps %%zmm1, %%zmm3, %%zmm14;"
        "vfmadd231ps %%zmm2, %%zmm3, %%zmm15;"
        /*cr=1s=2(h)(kh) end*/

        /*cr=2s=0(h)(kh) begin*/
        "lea (%%rbx, %%r9, 4), %%rbx;"
        "mov %%rbx, %%rcx;"
        "vmovups (%%rcx), %%zmm0;"
        "lea (%%rcx, %%r9, 4), %%rcx;"
        "vmovups (%%rcx), %%zmm1;"
        "lea (%%rcx, %%r9, 4), %%rcx;"
        "vmovups (%%rcx), %%zmm2;"

        "lea (%%rdi, %%r12, 4), %%rdi;"
        "mov %%rdi, %%rsi;"
        "vbroadcastss (%%rsi), %%zmm3;"
        "vfmadd231ps %%zmm0, %%zmm3, %%zmm4;"
        "vfmadd231ps %%zmm1, %%zmm3, %%zmm5;"
        "vfmadd231ps %%zmm2, %%zmm3, %%zmm6;"

        "lea (%%rsi, %%r10, 4), %%rsi;"
        "vbroadcastss (%%rsi), %%zmm3;"
        "vfmadd231ps %%zmm0, %%zmm3, %%zmm7;"
        "vfmadd231ps %%zmm1, %%zmm3, %%zmm8;"
        "vfmadd231ps %%zmm2, %%zmm3, %%zmm9;"

        "lea (%%rsi, %%r10, 4), %%rsi;"
        "vbroadcastss (%%rsi), %%zmm3;"
        "vfmadd231ps %%zmm0, %%zmm3, %%zmm10;"
        "vfmadd231ps %%zmm1, %%zmm3, %%zmm11;"
        "vfmadd231ps %%zmm2, %%zmm3, %%zmm12;"

        "lea (%%rsi, %%r10, 4), %%rsi;"
        "vbroadcastss (%%rsi), %%zmm3;"
        "vfmadd231ps %%zmm0, %%zmm3, %%zmm13;"
        "vfmadd231ps %%zmm1, %%zmm3, %%zmm14;"
        "vfmadd231ps %%zmm2, %%zmm3, %%zmm15;"
        /*cr=0s=0(h)(kh) end*/

        /*cr=2s=1(h)(kh) begin*/
        "lea 4(%%rbx), %%rcx;"
        "vmovups (%%rcx), %%zmm0;"
        "lea (%%rcx, %%r9, 4), %%rcx;"
        "vmovups (%%rcx), %%zmm1;"
        "lea (%%rcx, %%r9, 4), %%rcx;"
        "vmovups (%%rcx), %%zmm2;"

        "lea 4(%%rdi), %%rsi;"
        "vbroadcastss (%%rsi), %%zmm3;"
        "vfmadd231ps %%zmm0, %%zmm3, %%zmm4;"
        "vfmadd231ps %%zmm1, %%zmm3, %%zmm5;"
        "vfmadd231ps %%zmm2, %%zmm3, %%zmm6;"

        "lea (%%rsi, %%r10, 4), %%rsi;"
        "vbroadcastss (%%rsi), %%zmm3;"
        "vfmadd231ps %%zmm0, %%zmm3, %%zmm7;"
        "vfmadd231ps %%zmm1, %%zmm3, %%zmm8;"
        "vfmadd231ps %%zmm2, %%zmm3, %%zmm9;"

        "lea (%%rsi, %%r10, 4), %%rsi;"
        "vbroadcastss (%%rsi), %%zmm3;"
        "vfmadd231ps %%zmm0, %%zmm3, %%zmm10;"
        "vfmadd231ps %%zmm1, %%zmm3, %%zmm11;"
        "vfmadd231ps %%zmm2, %%zmm3, %%zmm12;"

        "lea (%%rsi, %%r10, 4), %%rsi;"
        "vbroadcastss (%%rsi), %%zmm3;"
        "vfmadd231ps %%zmm0, %%zmm3, %%zmm13;"
        "vfmadd231ps %%zmm1, %%zmm3, %%zmm14;"
        "vfmadd231ps %%zmm2, %%zmm3, %%zmm15;"
        /*cr=0s=1(h)(kh) end*/

        /*cr=2s=2(h)(kh) begin*/
        "lea 8(%%rbx), %%rcx;"
        "vmovups (%%rcx), %%zmm0;"
        "lea (%%rcx, %%r9, 4), %%rcx;"
        "vmovups (%%rcx), %%zmm1;"
        "lea (%%rcx, %%r9, 4), %%rcx;"
        "vmovups (%%rcx), %%zmm2;"

        "lea 8(%%rdi), %%rsi;"
        "vbroadcastss (%%rsi), %%zmm3;"
        "vfmadd231ps %%zmm0, %%zmm3, %%zmm4;"
        "vfmadd231ps %%zmm1, %%zmm3, %%zmm5;"
        "vfmadd231ps %%zmm2, %%zmm3, %%zmm6;"

        "lea (%%rsi, %%r10, 4), %%rsi;"
        "vbroadcastss (%%rsi), %%zmm3;"
        "vfmadd231ps %%zmm0, %%zmm3, %%zmm7;"
        "vfmadd231ps %%zmm1, %%zmm3, %%zmm8;"
        "vfmadd231ps %%zmm2, %%zmm3, %%zmm9;"

        "lea (%%rsi, %%r10, 4), %%rsi;"
        "vbroadcastss (%%rsi), %%zmm3;"
        "vfmadd231ps %%zmm0, %%zmm3, %%zmm10;"
        "vfmadd231ps %%zmm1, %%zmm3, %%zmm11;"
        "vfmadd231ps %%zmm2, %%zmm3, %%zmm12;"

        "lea (%%rsi, %%r10, 4), %%rsi;"
        "vbroadcastss (%%rsi), %%zmm3;"
        "vfmadd231ps %%zmm0, %%zmm3, %%zmm13;"
        "vfmadd231ps %%zmm1, %%zmm3, %%zmm14;"
        "vfmadd231ps %%zmm2, %%zmm3, %%zmm15;"
        /*cr=0s=2(h)(kh) end*/

        "lea (%%rax, %%r8, 4), %%rax;"
        "lea (%%rdx, %%r11, 4), %%rdx;"

        "sub $1, %%r13;"
        "jnz .compute%=;"

        // store Out
        "mov %[W], %%rax;"
        "mov %[HW], %%rbx;"

        "mov %[Out], %%rdi;" // Out[k * HW]
        "vmovups %%zmm4, (%%rdi);"
        "mov %%rax, %%rdx;"
        "vmovups %%zmm5, (%%rdi, %%rdx, 4);"
        "add %%rax, %%rdx;"
        "vmovups %%zmm6, (%%rdi, %%rdx, 4);"

        "lea (%%rdi, %% rbx, 4), %%rdi;"
        "vmovups  %%zmm7, (%%rdi);"
        "mov %%rax, %%rdx;"
        "vmovups %%zmm8, (%%rdi, %%rdx, 4);"
        "add %%rax, %%rdx;"
        "vmovups %%zmm9, (%%rdi, %%rdx, 4);"

        "lea (%%rdi, %% rbx, 4), %%rdi;"
        "vmovups %%zmm10, (%%rdi);"
        "mov %%rax, %%rdx;"
        "vmovups %%zmm11, (%%rdi, %%rdx, 4);"
        "add %%rax, %%rdx;"
        "vmovups %%zmm12, (%%rdi, %%rdx, 4);"

        "lea (%%rdi, %% rbx, 4), %%rdi;"
        "vmovups %%zmm13, (%%rdi);"
        "mov %%rax, %%rdx;"
        "vmovups %%zmm14, (%%rdi, %%rdx, 4);"
        "add %%rax, %%rdx;"
        "vmovups  %%zmm15, (%%rdi, %%rdx, 4);"

        :
        : [In] "m"(In),
          [Weight] "m"(Weight),
          [Out] "m"(Out),
          [HW] "m"(HW_s),
          [W] "m"(W_s),
          [C_in] "m"(C_in),
          [HRWS] "m"(HRWS_s),
          [WS] "m"(WS_s),
          [CRS] "m"(CRS_s),
          [RS] "m"(RS_s),
          [S] "m"(S_s)
        : "rax", "rbx", "rcx", "rdx", "rsi", "rdi", "r8", "r9", "r10", "r11", "r12", "r13", "zmm0", "zmm1", "zmm2", "zmm3", "zmm4", "zmm5", "zmm6", "zmm7", "zmm8", "zmm9", "zmm10", "zmm11", "zmm12", "zmm13", "zmm14", "zmm15");
    return 0;
}

extern "C" int conv_reset(float *Out, int KHW, int HW, int W,
                          int N_, int K_, int H_, int W_)
{
    for (int n = 0; n < N_; n++)
        for (int k = 0; k < K_; k++)
            for (int h = 0; h < H_; h++)
                for (int w = 0; w < W_; w++)
                {
                    Out[n * KHW + k * HW + h * W + w] = 0;
                }
    return 0;
}

    """
    from tvm.contrib import utils, clang
    temp = utils.tempdir()
    ll_path = temp.relpath("temp.ll")
    ll_code = clang.create_llvm(cc_code, output=ll_path, options=["-mavx", "-msse"])
    return ll_code

def test_intrin_no_tensorize(N, K, H, W, C, R, S):
    ins, outs = conv_notensorize(N,K,H,W,C,R,S)
    out = outs[0]
    inputs_np = [
        np.random.uniform(0, 1, [int(x) for x in y.shape]).astype(dtype)
        for y in ins
    ]
    outputs_np = [
        np.random.uniform(0, 1, [int(x) for x in y.shape]).astype(dtype)
        for y in outs
    ]
    outputs_gt_np = [
        np.random.uniform(0, 1, [int(x) for x in y.shape]).astype(dtype)
        for y in outs
    ]
    ctx = tvm.cpu()
    inputs = [tvm.nd.array(x, ctx) for x in inputs_np]
    outputs = [tvm.nd.array(x, ctx) for x in outputs_np]
    outputs_gt = [tvm.nd.array(x, ctx) for x in outputs_gt_np]

    n,k,h,w = out.op.axis
    sch = tvm.te.create_schedule(out.op)
    c,r,s = sch[out].op.reduce_axis
    no, ni = sch[out].split(n, NI)
    ko, ki = sch[out].split(k, KI)
    ho, hi = sch[out].split(h, HI)
    wo, wi = sch[out].split(w, WI)
    co, ci = sch[out].split(c, CI)
    
    sch[out].reorder(no,ko,ho,wo,co,ni,ki,hi,wi,ci,r,s)
    code = conv_impl()
    intrin = intrin_conv((NI,KI,HI,WI,CI,R,S), "float32", prefix="")
    sch[out].tensorize(ni,intrin)
    sch[out].pragma (no, "import_llvm", code)
    
    print(tvm.lower(sch, ins+outs,simple_mode=True))

    func = tvm.build(sch, ins + outs, name = "func")
    func(*inputs, *outputs)
    
    sch0 = tvm.te.create_schedule(out.op)
    func0 = tvm.build(sch0, ins + outs, name = "func_gt")
    func0(*inputs, *outputs_gt)
    print("schedule passed!")
    
    

    absolute_difference = np.max(outputs[0].numpy() - outputs_gt[0].numpy())
    tmp = outputs[0].numpy() - outputs_gt[0].numpy()
    with open("data.npy", 'wb') as f:
        np.save(f, tmp)
    print("atol:", absolute_difference)

    for a, b in zip(outputs, outputs_gt):
        tvm.testing.assert_allclose(
                a.numpy(), b.numpy(), atol=1, rtol=1e-6)
    
    print("test passed!")


def test_intrin(N, K, H, W, C, R, S, dtype):
    print("test intrin: ", N, K, H, W, C, R, S, dtype)
    ins, outs = conv(N, K, H, W, C, R, S)
    Out = outs[0]

    ctx = tvm.cpu()
    inputs_np = [
        np.random.uniform(0, 1, [int(x) for x in y.shape]).astype(dtype)
        for y in ins
    ]
    outputs_np = [
        np.random.uniform(0, 1, [int(x) for x in y.shape]).astype(dtype)
        for y in outs
    ]
    outputs_gt_np = [
        np.random.uniform(0, 1, [int(x) for x in y.shape]).astype(dtype)
        for y in outs
    ]
    inputs = [tvm.nd.array(x, ctx) for x in inputs_np]
    outputs = [tvm.nd.array(x, ctx) for x in outputs_np]
    outputs_gt = [tvm.nd.array(x, ctx) for x in outputs_gt_np]

    
    sch0 = tvm.te.create_schedule(Out.op)
    func0 = tvm.build(sch0, ins + outs, name = "func_gt")
    func0(*inputs, *outputs_gt)
    print("schedule passed!")
    
    sch = tvm.te.create_schedule(Out.op)
    no, ni, ko, ki, ho, hi, wo, wi = Out.op.axis
    rco, rci, rr, rs = Out.op.reduce_axis
    order = [no, ko, ho, wo, rco, ni, ki, hi, wi, rci,rr, rs]
    sch[Out].reorder(*order)
    packed, code = at.cpu_intrin("conv", (NI, KI, HI, WI, CI, R, S), dtype = "float32", isa ="avx512")
    print(tvm.lower(sch, ins+outs, simple_mode = True))
    sch[Out].tensorize(ni, packed.compute_intrinsic)
    sch[Out].pragma(no, "import_llvm", code)
    
    
    func = tvm.build(sch, ins + outs, name = "func")
    func(*inputs, *outputs)
    
    evaluator = func.time_evaluator(
            func.entry_name, ctx, min_repeat_ms=10, repeat=100
        )
    
    cost = evaluator(*inputs, *outputs)

    cost = np.mean(cost.results[10:])
    print(cost)

    absolute_difference = np.max(outputs[0].numpy() - outputs_gt[0].numpy())
    print("max atol: ", absolute_difference)
    print("cost: ", cost)
    
    for a, b in zip(outputs, outputs_gt):
        tvm.testing.assert_allclose(
                a.numpy(), b.numpy(), atol=1, rtol=1e-6)
    
    print("test passed!")



if __name__ == "__main__":
    l = sys.argv[1:]
    assert len(l) == 8
    l, dtype = l[:-1], l[-1] 
    l = [int(_) for _ in l]
    # test_intrin(*l, dtype)
    test_intrin_no_tensorize(*l)