CXX ?= g++
CFLAGS = -std=c++11 -O3 -Wall 

gemm: gemm.cc
	$(CXX) $(CFLAGS) gemm.cc -o gemm 

gemm_prof: gemm.cc
	$(CXX) $(CFLAGS) -DPROFILE gemm.cc -o gemm_prof

genRunSh: genRunSh.cc
	$(CXX) $(CFLAGS) genRunSh.cc -o genRunSh -fopenmp -lpthread

clean: 
	rm gemm gemm_prof