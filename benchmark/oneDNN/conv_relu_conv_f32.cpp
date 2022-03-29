#include <assert.h>

#include <chrono>
#include <vector>
#include <unordered_map>

#include "example_utils.hpp"
#include "oneapi/dnnl/dnnl.hpp"

using namespace dnnl;
int batch, C0, P0, Q0, C1, R1, S1, C2, R2, S2, PEAKGFLOPS, pad1, pad2, stride1,stride2;
int P0_pad, Q0_pad, P1, Q1, P1_pad, Q1_pad, P2, Q2;
double conv_relu_conv(engine::kind engine_kind, int times = 100) {
    using tag = memory::format_tag;
    using dt = memory::data_type;

    /// Initialize an engine and stream. The last parameter in the call represents
    /// the index of the engine.
    /// @snippet cnn_inference_f32.cpp Initialize engine and stream
    //[Initialize engine and stream]
    engine eng(engine_kind, 0);
    stream s(eng);
    //[Initialize engine and stream]

    /// Create a vector for the primitives and a vector to hold memory
    /// that will be used as arguments.
    /// @snippet cnn_inference_f32.cpp Create network
    //[Create network]
    std::vector<primitive> net;
    std::vector<std::unordered_map<int, memory>> net_args;
    //[Create network]

    const memory::dim batch = 1;

    // AlexNet: conv1
    // {batch, C0, P0, Q0} (x) {C1, C0, R1, S1} -> {batch, C1, P1, Q1}
    // strides: {stride1, stride1}
    memory::dims conv1_src_tz = {batch, C0, P0, Q0};
    memory::dims conv1_weights_tz = {C1, C0, R1, S1};
    memory::dims conv1_bias_tz = {C1};
    memory::dims conv1_dst_tz = {batch, C1, P1, Q1};
    memory::dims conv1_strides = {stride1, stride1};
    memory::dims conv1_padding = {pad1, pad1};

    /// Allocate buffers for input and output data, weights, and bias.
    /// @snippet cnn_inference_f32.cpp Allocate buffers
    //[Allocate buffers]
    std::vector<float> user_src(batch * C0 * P0 * Q0);
    // TODO: 1000?
    std::vector<float> user_dst(batch * 1000);
    std::vector<float> conv1_weights(product(conv1_weights_tz));
    std::vector<float> conv1_bias(product(conv1_bias_tz));
    //[Allocate buffers]

    /// Create memory that describes data layout in the buffers. This example uses
    /// tag::nchw (batch-channels-height-width) for input data and tag::oihw
    /// for weights.
    /// @snippet cnn_inference_f32.cpp Create user memory
    //[Create user memory]
    auto user_src_memory = memory({{conv1_src_tz}, dt::f32, tag::nchw}, eng);
    
    auto user_weights_memory
            = memory({{conv1_weights_tz}, dt::f32, tag::oihw}, eng);
    auto conv1_user_bias_memory
            = memory({{conv1_bias_tz}, dt::f32, tag::x}, eng);
    //[Create user memory]

    /// Create memory descriptors with layout tag::any. The `any` format enables
    /// the convolution primitive to choose the data format that will result in
    /// best performance based on its input parameters (convolution kernel
    /// sizes, strides, padding, and so on). If the resulting format is different
    /// from `nchw`, the user data must be transformed to the format required for
    /// the convolution (as explained below).
    /// @snippet cnn_inference_f32.cpp Create convolution memory descriptors
    //[Create convolution memory descriptors]
    auto conv1_src_md = memory::desc({conv1_src_tz}, dt::f32, tag::any);
    auto conv1_bias_md = memory::desc({conv1_bias_tz}, dt::f32, tag::any);
    auto conv1_weights_md = memory::desc({conv1_weights_tz}, dt::f32, tag::any);
    auto conv1_dst_md = memory::desc({conv1_dst_tz}, dt::f32, tag::any);
    //[Create convolution memory descriptors]

    /// Create a convolution descriptor by specifying propagation kind,
    /// [convolution algorithm](@ref dev_guide_convolution), shapes of input,
    /// weights, bias, output, convolution strides, padding, and kind of padding.
    /// Propagation kind is set to prop_kind::forward_inference to optimize for
    /// inference execution and omit computations that are necessary only for
    /// backward propagation.
    /// @snippet cnn_inference_f32.cpp Create convolution descriptor
    //[Create convolution descriptor]
    auto conv1_desc = convolution_forward::desc(prop_kind::forward_inference,
            algorithm::convolution_direct, conv1_src_md, conv1_weights_md,
            conv1_bias_md, conv1_dst_md, conv1_strides, conv1_padding,
            conv1_padding);
    //[Create convolution descriptor]

    /// Create a convolution primitive descriptor. Once created, this
    /// descriptor has specific formats instead of the `any` format specified
    /// in the convolution descriptor.
    /// @snippet cnn_inference_f32.cpp Create convolution primitive descriptor
    //[Create convolution primitive descriptor]
    auto conv1_prim_desc = convolution_forward::primitive_desc(conv1_desc, eng);
    //[Create convolution primitive descriptor]

    /// Check whether data and weights formats required by convolution is different
    /// from the user format. In case it is different change the layout using
    /// reorder primitive.
    /// @snippet cnn_inference_f32.cpp Reorder data and weights
    //[Reorder data and weights]
    auto conv1_src_memory = user_src_memory;
    if (conv1_prim_desc.src_desc() != user_src_memory.get_desc()) {
        conv1_src_memory = memory(conv1_prim_desc.src_desc(), eng);
        net.push_back(reorder(user_src_memory, conv1_src_memory));
        net_args.push_back({{DNNL_ARG_FROM, user_src_memory},
                {DNNL_ARG_TO, conv1_src_memory}});
    }

    auto conv1_weights_memory = user_weights_memory;
    if (conv1_prim_desc.weights_desc() != user_weights_memory.get_desc()) {
        conv1_weights_memory = memory(conv1_prim_desc.weights_desc(), eng);
        reorder(user_weights_memory, conv1_weights_memory)
                .execute(s, user_weights_memory, conv1_weights_memory);
    }
    //[Reorder data and weights]

    /// Create a memory primitive for output.
    /// @snippet cnn_inference_f32.cpp Create memory for output
    //[Create memory for output]
    auto conv1_dst_memory = memory(conv1_prim_desc.dst_desc(), eng);
    //[Create memory for output]

    /// Create a convolution primitive and add it to the net.
    /// @snippet cnn_inference_f32.cpp Create memory for output
    //[Create convolution primitive]
    net.push_back(convolution_forward(conv1_prim_desc));
    net_args.push_back({{DNNL_ARG_SRC, conv1_src_memory},
            {DNNL_ARG_WEIGHTS, conv1_weights_memory},
            {DNNL_ARG_BIAS, conv1_user_bias_memory},
            {DNNL_ARG_DST, conv1_dst_memory}});
    //[Create convolution primitive]

    // AlexNet: relu1
    // {batch, C1, P1, Q1} -> {batch, C1, P1, Q1}
    const float negative1_slope = 0.0f;

    /// Create the relu primitive. For better performance, keep the input data
    /// format for ReLU (as well as for other operation primitives until another
    /// convolution or inner product is encountered) the same as the one chosen
    /// for convolution. Also note that ReLU is done in-place by using conv1 memory.
    /// @snippet cnn_inference_f32.cpp Create relu primitive
    //[Create relu primitive]
    auto relu1_desc = eltwise_forward::desc(prop_kind::forward_inference,
            algorithm::eltwise_relu, conv1_dst_memory.get_desc(),
            negative1_slope);
    auto relu1_prim_desc = eltwise_forward::primitive_desc(relu1_desc, eng);

    net.push_back(eltwise_forward(relu1_prim_desc));
    net_args.push_back({{DNNL_ARG_SRC, conv1_dst_memory},
            {DNNL_ARG_DST, conv1_dst_memory}});
    //[Create relu primitive]

    // AlexNet: conv2
    // {batch, C1, P1, Q1} (x) {C2, C1, R2, S2} -> {batch, C2, P2, Q2}
    // strides: {stride2, stride2}
    memory::dims conv2_src_tz = {batch, C1, P1, Q1};
    memory::dims conv2_weights_tz = {C2, C1, R2, S2};
    memory::dims conv2_bias_tz = {C2};
    memory::dims conv2_dst_tz = {batch, C2, P2, Q2};
    memory::dims conv2_strides = {stride2, stride2};
    memory::dims conv2_padding = {pad2, pad2};

    std::vector<float> conv2_weights(product(conv2_weights_tz));
    std::vector<float> conv2_bias(product(conv2_bias_tz));

    // create memory for user data
    auto conv2_user_weights_memory
            = memory({{conv2_weights_tz}, dt::f32, tag::oihw}, eng);
    
    auto conv2_user_bias_memory
            = memory({{conv2_bias_tz}, dt::f32, tag::x}, eng);
    

    // create memory descriptors for convolution data w/ no specified format
    auto conv2_src_md = memory::desc({conv2_src_tz}, dt::f32, tag::any);
    auto conv2_bias_md = memory::desc({conv2_bias_tz}, dt::f32, tag::any);
    auto conv2_weights_md = memory::desc({conv2_weights_tz}, dt::f32, tag::any);
    auto conv2_dst_md = memory::desc({conv2_dst_tz}, dt::f32, tag::any);

    // create a convolution
    auto conv2_desc = convolution_forward::desc(prop_kind::forward_inference,
            algorithm::convolution_direct, conv2_src_md, conv2_weights_md,
            conv2_bias_md, conv2_dst_md, conv2_strides, conv2_padding,
            conv2_padding);
    auto conv2_prim_desc = convolution_forward::primitive_desc(conv2_desc, eng);

    auto conv2_src_memory = conv1_dst_memory;
    if (conv2_prim_desc.src_desc() != conv2_src_memory.get_desc()) {
        conv2_src_memory = memory(conv2_prim_desc.src_desc(), eng);
        net.push_back(reorder(conv1_dst_memory, conv2_src_memory));
        net_args.push_back({{DNNL_ARG_FROM, conv1_dst_memory},
                {DNNL_ARG_TO, conv2_src_memory}});
    }

    auto conv2_weights_memory = conv2_user_weights_memory;
    if (conv2_prim_desc.weights_desc()
            != conv2_user_weights_memory.get_desc()) {
        conv2_weights_memory = memory(conv2_prim_desc.weights_desc(), eng);
        reorder(conv2_user_weights_memory, conv2_weights_memory)
                .execute(s, conv2_user_weights_memory, conv2_weights_memory);
    }

    auto conv2_dst_memory = memory(conv2_prim_desc.dst_desc(), eng);

    // create convolution primitive and add it to net
    net.push_back(convolution_forward(conv2_prim_desc));
    net_args.push_back({{DNNL_ARG_SRC, conv2_src_memory},
            {DNNL_ARG_WEIGHTS, conv2_weights_memory},
            {DNNL_ARG_BIAS, conv2_user_bias_memory},
            {DNNL_ARG_DST, conv2_dst_memory}});
    
    //[Execute model]
    std::chrono::high_resolution_clock::time_point t1_, t2_;

    double time = 0;

    for (int j = 0; j < times; ++j) {
        for (int i = 0; i < user_src.size(); i++){
            user_src[i] = ((double)(rand()%999)) / 999;
        }
        for (int i = 0; i < conv1_bias.size(); i++){
            conv1_bias[i] = 0;
        }
        for (int i = 0; i < conv1_weights.size(); i++){
            conv1_weights[i] = ((double)(rand()%999)) / 999;
        }
        for (int i = 0; i < conv2_bias.size(); i++){
            conv2_bias[i] = 0;
        }
        for (int i = 0; i < conv2_weights.size(); i++){
            conv2_weights[i] = ((double)(rand()%999)) / 999;
        }
        write_to_dnnl_memory(user_src.data(), user_src_memory);
        write_to_dnnl_memory(conv1_weights.data(), user_weights_memory);
        write_to_dnnl_memory(conv1_bias.data(), conv1_user_bias_memory);
        write_to_dnnl_memory(conv2_weights.data(), conv2_user_weights_memory);
        write_to_dnnl_memory(conv2_bias.data(), conv2_user_bias_memory);
        assert(net.size() == net_args.size() && "something is missing");
        t1_ = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < net.size(); ++i)
            net.at(i).execute(s, net_args.at(i));
        s.wait();
        t2_ = std::chrono::high_resolution_clock::now();
        time += (double)(std::chrono::duration_cast<std::chrono::duration<double>>(t2_ - t1_)).count();
    }
    //[Execute model]
    return time / times / 1000;
}

void conv_relu_conv_f32(engine::kind engine_kind) {
    P0_pad = P0 + 2 * pad1;
    Q0_pad = Q0 + 2 * pad1;
    P1 = (P0_pad - R1) / stride1 + 1;
    Q1 = (Q0_pad - S1) / stride1 + 1;
    P1_pad = P1 + 2 * pad2;
    Q1_pad = Q1 + 2 * pad2;
    P2 = (P1_pad - R2)/ stride2 + 1;
    Q2 = (Q1_pad - S2)/ stride2 + 1;
    int times = 100;
    double time = conv_relu_conv(engine_kind, times);
    double workload = batch * (C0 * P0 * Q0 * C1 * R1 * S1 + C1 * P1 * Q1 * C2 * R2 * S2);
    double ratioToPeak = (workload / time / 1e9) / PEAKGFLOPS;
    std::cout << "time(s): " << time
              << " ratioToPeak: " << ratioToPeak << std::endl;
}
 
int main(int argc, char **argv) {
    if (argc != 13){
        std::cout << "Usage: ./script N, C0, P0, Q0, C1, R1, S1, C2, R2, S2 peak gflops" << std::endl;
        return 0;
    }
    batch = atoi(argv[2]);
    C0 = atoi(argv[3]);
    P0 = atoi(argv[4]);
    Q0 = atoi(argv[5]);
    C1 = atoi(argv[6]);
    R1 = atoi(argv[7]);
    S1 = atoi(argv[8]);
    C2 = atoi(argv[9]);
    R2 = atoi(argv[10]);
    S2 = atoi(argv[11]);
    PEAKGFLOPS = atoi(argv[12]);
    pad1 = pad2 = stride1 = stride2 = 1;
    return handle_example_errors(
            conv_relu_conv_f32, parse_engine_kind(argc, argv, 11));
}