cudaFlags = -arch=sm_70   --use_fast_math -Xptxas -v,-dlcm=ca,-O3 -Xcompiler -O3 

flags = -Ddfloat=double
flags += -DdfloatString='"double"'
flags += -I../../../core

BK1: BK1.cu makefile
	mpic++ $(flags) -c  -I../../../core/ ../../../core/meshBasis.cpp
	nvcc $(flags) -o BK1 $(cudaFlags) BK1.cu meshBasis.o -llapack -lblas

clean: 
	rm -f BK1 BK1.o meshBasis.o

