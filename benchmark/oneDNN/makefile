IoneDNN = /home/gulang2022/intel/oneapi/dnnl/latest/cpu_iomp/examples -I/home/gulang2022/intel/oneapi/dnnl/latest/cpu_iomp/include
LoneDNN = /home/gulang2022/intel/oneapi/dnnl/latest/cpu_iomp/lib
LCOMPILER = /home/gulang2022/intel/oneapi/compiler/latest/linux/compiler/lib/intel64_lin/
CFLAGS =-Wl,-rpath=/home/gulang2022/intel/oneapi/dnnl/latest/cpu_iomp/lib  -ldnnl -fopenmp
CC=/home/gulang2022/intel/oneapi/compiler/latest/linux/bin/icpx

all: bmm_bmm_f32 bmm_softmax_bmm_f32 conv_relu_conv_f32 conv_conv_f32 bmm_relu_bmm_f32

conv_relu_conv_f32: conv_relu_conv_f32.cpp
	$(CC) -O3 conv_relu_conv_f32.cpp -o conv_relu_conv_f32 -I $(IoneDNN) -L $(LoneDNN) $(CFLAGS)

bmm_relu_bmm_f32: bmm_relu_bmm_f32.cpp
	$(CC) -O3 bmm_relu_bmm_f32.cpp -o bmm_relu_bmm_f32 -I $(IoneDNN) -L $(LoneDNN) $(CFLAGS)

bmm_softmax_bmm_f32: bmm_softmax_bmm_f32.cpp
	$(CC) -O3 bmm_softmax_bmm_f32.cpp -o bmm_softmax_bmm_f32 -I $(IoneDNN) -L $(LoneDNN) $(CFLAGS)

conv_conv_f32: conv_conv_f32.cpp
	$(CC) -O3 conv_conv_f32.cpp -o conv_conv_f32 -I $(IoneDNN) -L $(LoneDNN) $(CFLAGS)

bmm_bmm_f32: bmm_bmm_f32.cpp
	$(CC) -O3 bmm_bmm_f32.cpp -o bmm_bmm_f32 -I $(IoneDNN) -L $(LoneDNN) $(CFLAGS)

clean:
	rm bmm_bmm_f32 bmm_softmax_bmm_f32 conv_relu_conv_f32 conv_conv_f32 bmm_relu_bmm_f32

