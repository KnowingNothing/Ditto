# modify include path, library path and rpath below
IoneDNN = /opt/intel/oneapi/dnnl/latest/cpu_iomp/examples -I/opt/intel/oneapi/dnnl/latest/cpu_iomp/include
LoneDNN = /opt/intel/oneapi/dnnl/latest/cpu_iomp/lib
LCOMPILER = /opt/intel/compilers_and_libraries_2019.3.199/linux/compiler/lib/intel64_lin/
TEMP = /opt/intel/compilers_and_libraries_2019.3.199/linux/mkl/lib/intel64_lin/
CFLAGS =-Wl,-rpath=/opt/intel/oneapi/dnnl/latest/cpu_iomp/lib  -ldnnl -fopenmp

conv_relu_conv_f32: conv_relu_conv_f32.cpp
	icpc -O3 conv_relu_conv_f32.cpp -o conv_relu_conv_f32 -I $(IoneDNN) -L $(LoneDNN) $(CFLAGS)


clean:
	rm conv_relu_conv_f32