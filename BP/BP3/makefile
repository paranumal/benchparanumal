#####################################################################################
#
#The MIT License (MIT)
#
#Copyright (c) 2017-2022 Tim Warburton, Noel Chalmers, Jesse Chan, Ali Karakus
#
#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files (the "Software"), to deal
#in the Software without restriction, including without limitation the rights
#to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:
#
#The above copyright notice and this permission notice shall be included in all
#copies or substantial portions of the Software.
#
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#SOFTWARE.
#
#####################################################################################

define BP3_HELP_MSG

Bakeoff Problem 3 makefile targets:

   make BP3 (default)
   make clean
   make clean-libs
   make clean-kernels
   make realclean
   make info
   make help

Usage:

make BP3
   Build BP3 executable.
make clean
   Clean the BP3 executable and object files.
make clean-libs
   In addition to "make clean", also clean the mesh, ogs, and core libraries.
make clean-kernels
   In addition to "make clean-libs", also cleans the cached OCCA kernels.
make realclean
   In addition to "make clean-kernels", also clean 3rd party libraries.
make info
   List directories and compiler flags in use.
make help
   Display this help message.

Can use "make verbose=true" for verbose output.

endef

ifeq (,$(filter BP3 clean clean-libs clean-kernels \
                realclean info help,$(MAKECMDGOALS)))
ifneq (,$(MAKECMDGOALS))
$(error ${BP3_HELP_MSG})
endif
endif

ifndef LIBP_MAKETOP_LOADED
ifeq (,$(wildcard ../../make.top))
$(error cannot locate ${PWD}/../../make.top)
else
include ../../make.top
endif
endif

#libraries
CORE_LIBS=mesh ogs core

#includes
INCLUDES=${LIBP_INCLUDES} \
		 -I${LIBP_LIBS_DIR}/include \
         -I.

#defines
DEFINES =${LIBP_DEFINES} \
         -DLIBP_DIR='"${LIBP_DIR}"'

#.cpp compilation flags
BP3_CXXFLAGS=${LIBP_CXXFLAGS} ${DEFINES} ${INCLUDES}

#link libraries
LIBS=-L${LIBP_LIBS_DIR} $(addprefix -l,$(CORE_LIBS)) \
     ${LIBP_LIBS}

#link flags
LFLAGS=${BP3_CXXFLAGS} ${LIBS}

#object dependancies
DEPS=$(wildcard *.hpp) \
     $(wildcard $(LIBP_INCLUDE_DIR)/*.h) \
     $(wildcard $(LIBP_INCLUDE_DIR)/*.hpp)

SRC =$(wildcard src/*.cpp)

OBJS=$(SRC:.cpp=.o)

.PHONY: all libp_libs clean clean-libs \
		clean-kernels realclean help info BP3

all: BP3

libp_libs:
ifneq (,${verbose})
	${MAKE} -C ${LIBP_LIBS_DIR} $(CORE_LIBS) verbose=${verbose}
else
	@${MAKE} -C ${LIBP_LIBS_DIR} $(CORE_LIBS) --no-print-directory
endif

BP3:$(OBJS) bp3Main.o | libp_libs
ifneq (,${verbose})
	$(LIBP_LD) -o BP3 bp3Main.o $(OBJS) $(LFLAGS)
else
	@printf "%b" "$(EXE_COLOR)Linking $(@F)$(NO_COLOR)\n";
	@$(LIBP_LD) -o BP3 bp3Main.o $(OBJS) $(LFLAGS)
endif

# rule for .cpp files
%.o: %.cpp $(DEPS) | libp_libs
ifneq (,${verbose})
	$(LIBP_CXX) -o $*.o -c $*.cpp $(BP3_CXXFLAGS)
else
	@printf "%b" "$(OBJ_COLOR)Compiling $(@F)$(NO_COLOR)\n";
	@$(LIBP_CXX) -o $*.o -c $*.cpp $(BP3_CXXFLAGS)
endif

#cleanup
clean:
	rm -f src/*.o *.o BP3

clean-libs: clean
	${MAKE} -C ${LIBP_LIBS_DIR} clean

clean-kernels: clean-libs
	rm -rf ${LIBP_DIR}/.occa/

realclean: clean-libs
	${MAKE} -C ${OCCA_DIR} clean
	rm -rf ${LIBP_DIR}/.occa/

help:
	$(info $(value BP3_HELP_MSG))
	@true

info:
	$(info LIBP_DIR  = $(LIBP_DIR))
	$(info LIBP_ARCH = $(LIBP_ARCH))
	$(info CXXFLAGS  = $(BP3_CXXFLAGS))
	$(info LIBS      = $(LIBS))
	@true
