from tvm.target.datatype import register

cuda_bf16 = "cuda_bf16"
cuda_tf32 = "cuda_tf32"

register(cuda_bf16, 131)
register(cuda_tf32, 132)