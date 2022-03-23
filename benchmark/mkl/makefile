IMKL = /opt/intel/compilers_and_libraries_2019.3.199/linux/mkl/include
LMKL = /opt/intel/compilers_and_libraries_2019.3.199/linux/mkl/lib/intel64 -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core
LCOMPILER = /opt/intel/compilers_and_libraries_2019.3.199/linux/compiler/lib/intel64_lin/
TEMP = /opt/intel/compilers_and_libraries_2019.3.199/linux/mkl/lib/intel64_lin/
CFLAGS =-g  -Wl,--no-as-needed -liomp5 -lpthread -lm -ldl -fopenmp 

MKL2MM: MKL2MM.cc
	icpc -O3  MKL2MM.cc -o MKL2MM  -I $(IMKL) -L $(LMKL) -L $(LCOMPILER) $(CFLAGS) 

clean:
	rm MKL2MM
