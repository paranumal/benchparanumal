ifndef OCCA_DIR
ERROR:
	@echo "Error, environment variable [OCCA_DIR] is not set"
endif

include ../../make.inp

include ${OCCA_DIR}/scripts/Makefile

# define variables
HDRDIR = ../../core
GSDIR  = ../../3rdParty/gslib/
OGSDIR = ../../libs/gatherScatter/

# set options for this machine
# specify which compilers to use for c, fortran and linking
cc	= mpicc
FC	= mpif77  
#-std=legacy -fdefault-real-8 -fdefault-double-8 
CC	= mpic++
LD	= mpic++

NVCC	= nvcc

compilerFlags += -DUSE_CUDA_NATIVE=0
compilerFlags += -Ddfloat=double
compilerFlags += -Ddlong=int
compilerFlags += -DdfloatString='"double"'
compilerFlags += -DdlongString='"int"'

nvccFlags = -Ddfloat=double -Ddlong=int -DdfloatString='"double"' -DdlongString='"int"'  -I../../core -I../../../../devParanumal/libs/gatherScatter/  -D DBP='"/home/tcew/Work/git/benchparanumal/CEED/BP/occa"'  -I./ -g 


# s-I$(ALMONDDIR)
# compiler flags to be used (set to compile with debugging on)
CFLAGS = -I. -DOCCA_VERSION_1_0 $(compilerFlags) $(flags) -I$(HDRDIR) -I$(OGSDIR)  -D DBP='"${CURDIR}"' $(LIBP_OPT_FLAGS) -I$(OGSDIR)/include 
#-I../../3rdParty/parmetis-4.0.3/metis/GKlib -I../../3rdParty/parmetis-4.0.3/include -I../../3rdParty/parmetis-4.0.3/metis/include -I../../3rdParty/parmetis-4.0.3/libparmetis 

# link flags to be used
LDFLAGS	= -DOCCA_VERSION_1_0 $(compilerFlags) $(LIBP_OPT_FLAGS) -llapack -lblas 

PARMETISDIR = ../../3rdParty/parmetis-4.0.3/build/Linux-x86_64/libparmetis/
METISDIR = ../../3rdParty/parmetis-4.0.3/build/Linux-x86_64/libmetis

# libraries to be linked in
LIBS	=  -L$(OGSDIR) -logs -L$(GSDIR)/lib -lgs -L$(OCCA_DIR)/lib -locca $(linkerFlags) -llapack -lblas  $(LIBP_EXT_LIBS) -lgfortran -fopenmp
#-lcuda -lcudart -fopenmp
#-lcudart -lcuda
#-lcudart -lcuda

#-L$(PARMETISDIR) -lparmetis -L$(METISDIR) -lmetis  -L$(ALMONDDIR) -lparAlmond

INCLUDES = BP.h BPPrecon.h
DEPS = $(INCLUDES) \
$(HDRDIR)/mesh.h \
$(HDRDIR)/mesh2D.h \
$(HDRDIR)/mesh3D.h \
$(OGSDIR)/ogs.hpp \

#$(ALMONDDIR)/parAlmond.hpp \

# types of files we are going to construct rules for
.SUFFIXES: .c .f .cu .cpp

# rule for .c files
.c.o: $(DEPS)
	$(CC) $(CFLAGS) -o $*.o -c $*.c $(paths)

.cpp.o: $(DEPS)
	$(CC) $(CFLAGS) -o $*.o -c $*.cpp $(paths)

.cu.o: $(DEPS)
	$(NVCC) $(nvccFlags) -o $*.o -c $*.cu $(paths)

.f.o: $(DEPS)
	$(FC) $(CFLAGS) -o $*.o -c $*.f $(paths)

# add this object for USE_STEFAN_MXM ./src/BPSerialLocalAxHex3D.o\

# ./src/BK5.o\

# list of objects to be compiled
AOBJS    = \
./src/BP.o \
./src/BPSetup.o \
./src/BPSolve.o\
./src/BPOperator.o\
./src/BPVectors.o \
./src/BPPlotVTU.o

# library objects
LOBJS = \
../../core/meshSetup.o \
../../core/meshBasis.o \
../../core/setupAide.o

# libogs
BP:$(AOBJS) $(LOBJS)
	$(LD)  $(LDFLAGS)  -o BP $(COBJS) $(AOBJS) $(LOBJS) $(paths) $(LIBS)

lib:$(AOBJS)
	ar -cr libBP.a $(AOBJS)

#libogs:
#	cd ../../libs/gatherScatter; make -j lib; cd ../../solvers/BP

#libblas:
#	cd ../../3rdParty/BlasLapack; make -j lib; cd ../../solvers/BP

#libparAlmond:
#	cd ../../libs/parAlmond; make -j lib; cd ../../solvers/BP

all: lib BP

# what to do if user types "make clean"
clean:
	rm src/*.o BP ../../core/*.o 

realclean:
	rm src/*.o BP ../../core/*.o 

