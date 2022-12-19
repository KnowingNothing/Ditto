/*******************************************************************************
* Copyright 2020 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>
#include <chrono>

#include "dnnl.hpp"
#include "example_utils.hpp"

using namespace dnnl;

using tag = memory::format_tag;
using dt = memory::data_type;

int B, M, N, K, L;
double PEAKGFLOPS;

void matmul_example(dnnl::engine::kind engine_kind) {

    // Create execution dnnl::engine.
    dnnl::engine engine(engine_kind, 0);

    // Create dnnl::stream.
    dnnl::stream engine_stream(engine);
    std::vector<std::unordered_map<int, memory>> net_args;
    std::vector<primitive> net;

    // Source (src), weights, bias, and destination (dst) tensors dimensions.
    memory::dims src_dims = {B, M, K};
    memory::dims weights1_dims = {B, K, L};
    memory::dims bias1_dims = {1, 1, L};
    memory::dims dst1_dims = {B, M, L};

    // Initialize src, weights, bias.
    

    // Create memory descriptors and memory objects for src, weights, bias, and
    // dst.
    auto src_md = memory::desc(src_dims, dt::f32, tag::abc);
    auto weights1_md = memory::desc(weights1_dims, dt::f32, tag::abc);
    auto bias1_md = memory::desc(bias1_dims, dt::f32, tag::abc);
    auto dst1_md = memory::desc(dst1_dims, dt::f32, tag::abc);

    auto src_mem = memory(src_md, engine);
    auto weights1_mem = memory(weights1_md, engine);
    auto bias1_mem = memory(bias1_md, engine);
    auto dst1_mem = memory(dst1_md, engine); 

    // Create operation descriptor
    // auto matmul1_d = matmul::desc(src_md, weights1_md, bias1_md, dst1_md);

    // Create primitive post-ops (Softmax).
    const float scale = 1.0f;
    const float alpha = 0.f;
    const float beta = 0.f;
    
    primitive_attr matmul1_attr;

    // Create primitive descriptor.
    auto matmul1_pd = matmul::primitive_desc(engine, src_md, weights1_md, dst1_md, matmul1_attr);

    // Create the primitive.
    auto matmul1_prim = matmul(matmul1_pd);

    // Primitive arguments.

    net_args.push_back(
            {{DNNL_ARG_SRC, src_mem}, {DNNL_ARG_WEIGHTS, weights1_mem},
                    {DNNL_ARG_BIAS, bias1_mem}, {DNNL_ARG_DST, dst1_mem}});

    net.push_back(matmul1_prim);


    // the softmax
    const int axis = 2;
    // auto logsoftmax_d =softmax_forward::desc(
    //     prop_kind::forward_inference, dst1_mem.get_desc(), axis);

    auto logsoftmax_pd = softmax_forward::primitive_desc(engine, 
        prop_kind::forward_inference, algorithm::softmax_accurate, dst1_mem.get_desc(), dst1_mem.get_desc(), axis);

    auto logsoftmax_prim = softmax_forward(logsoftmax_pd);

    net_args.push_back(
        {
            {DNNL_ARG_SRC, dst1_mem},
            {DNNL_ARG_DST, dst1_mem}
        }
    );

    net.push_back(logsoftmax_prim);

    // The second gemm
    memory::dims weights2_dims = {B, L, N};
    memory::dims bias2_dims = {1, 1, N};
    memory::dims dst2_dims = {B, M, N};

    auto weights2_md = memory::desc(weights2_dims, dt::f32, tag::abc);
    auto bias2_md = memory::desc(bias2_dims, dt::f32, tag::abc);
    auto dst2_md = memory::desc(dst2_dims, dt::f32, tag::abc);

    auto weights2_mem = memory(weights2_md, engine);
    auto bias2_mem = memory(bias2_md, engine);
    auto dst2_mem = memory(dst2_md, engine);

    // auto matmul2_d = matmul::desc(dst1_md, weights2_md, bias2_md, dst2_md);

    primitive_attr matmul2_attr;
    // Create primitive descriptor.
    auto matmul2_pd = matmul::primitive_desc(engine, dst1_md, weights2_md, dst2_md, matmul2_attr);

    auto matmul2_src_memory = dst1_mem;
    if (matmul2_pd.src_desc() != matmul2_src_memory.get_desc()) {
        matmul2_src_memory = memory(matmul2_pd.src_desc(), engine);
        net.push_back(reorder(dst1_mem, matmul2_src_memory));
        net_args.push_back(
                {{DNNL_ARG_FROM, dst1_mem}, {DNNL_ARG_TO, matmul2_src_memory}});
    }

    // Create the primitive.
    auto matmul2_prim = matmul(matmul2_pd);

    net.push_back(matmul2_prim);
    
    net_args.push_back({{DNNL_ARG_SRC, matmul2_src_memory}, {DNNL_ARG_WEIGHTS, weights2_mem},
                    {DNNL_ARG_BIAS, bias2_mem}, {DNNL_ARG_DST, dst2_mem}});

    std::chrono::high_resolution_clock::time_point t1_, t2_;

    int times = 1000;
    double time = 0;
    auto flush = [](const uint8_t* array, const int size){
        for (int i = 0; i < size; i += 64){
            _mm_clflush(&array[i]);
        }
    };
    for (int j = 0; j < times; ++j) {
        // Allocate buffers.
        std::vector<float> src_data(product(src_dims));
        std::vector<float> weights1_data(product(weights1_dims));
        std::vector<float> bias1_data(product(bias1_dims));
        std::vector<float> dst1_data(product(dst1_dims));
        std::vector<float> weights2_data(product(weights2_dims));
        std::vector<float> bias2_data(product(bias2_dims));
        std::vector<float> dst2_data(product(dst2_dims));
        std::generate(src_data.begin(), src_data.end(), []() {
            static int i = 0;
            return std::cos(i++ / 10.f);
        });
        std::generate(weights1_data.begin(), weights1_data.end(), []() {
            static int i = 0;
            return std::sin(i++ * 2.f);
        });
        std::generate(weights2_data.begin(), weights2_data.end(), []() {
            static int i = 0;
            return std::tanh(i++);
        });

        // Write data to memory object's handles.
        write_to_dnnl_memory(src_data.data(), src_mem);
        write_to_dnnl_memory(weights1_data.data(), weights1_mem);
        write_to_dnnl_memory(bias1_data.data(), bias1_mem);
        write_to_dnnl_memory(weights2_data.data(), weights2_mem);
        write_to_dnnl_memory(bias2_data.data(), bias2_mem);
        flush((uint8_t *)src_mem.get_data_handle(), src_mem.get_desc().get_size());
        flush((uint8_t *)weights1_mem.get_data_handle(), weights1_mem.get_desc().get_size());
        flush((uint8_t *)bias1_mem.get_data_handle(), bias1_mem.get_desc().get_size());
        flush((uint8_t *)weights2_mem.get_data_handle(), weights2_mem.get_desc().get_size());
        flush((uint8_t *)bias2_mem.get_data_handle(), bias2_mem.get_desc().get_size());
        assert(net.size() == net_args.size() && "something is missing");
        t1_ = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < net.size(); ++i)
            net.at(i).execute(engine_stream, net_args.at(i));
        engine_stream.wait();
        t2_ = std::chrono::high_resolution_clock::now();
        time += (double)(std::chrono::duration_cast<
                                 std::chrono::duration<double>>(t2_ - t1_))
                        .count();
    }
    time /= times;


    double workload = B * M * L * (K + N);
    double ratioToPeak = ((double)workload / time / 1e9) / PEAKGFLOPS;
    std::cout << "time(s): " << time << std::endl;
    std::cout << "ratioToPeak: " << ratioToPeak << std::endl;
}

int main(int argc, char **argv) {
    if (argc != 8){
        std::cout << "usage ./script cpu B M N K L gflops\n";
        return 0;
    }
    B = atoi (argv[2]);
    M = atoi (argv[3]);
    N = atoi (argv[4]);
    K = atoi (argv[5]);
    L = atoi (argv[6]);
    PEAKGFLOPS = atof(argv[7]);
    return handle_example_errors(matmul_example, parse_engine_kind(argc, argv, 6));
}