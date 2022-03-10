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

define LIBP_HELP_MSG

LIBP Bakeoff Kernels makefile targets:

	 make BK (default)
	 make BK{#}
	 make clean
	 make clean-libs
	 make realclean
	 make info
	 make help

Usage:

make BK
	 Builds each Bakeoff Kernel executable.
make BK{#}
	 Builds a Bakeoff Kernel executable,
	 # can be 1-6.
make clean
	 Cleans all BK executables and object files.
make clean-BK{#}
	 Cleans a BK executable and object files.
make clean-libs
	 In addition to "make clean", also clean the core, mesh, and ogs libraries.
make clean-kernels
   In addition to "make clean-libs", also cleans the cached OCCA kernels.
make realclean
	 In addition to "make clean-libs", also clean 3rd party libraries.
make info
	 List directories and compiler flags in use.
make help
	 Display this help message.

Can use "make verbose=true" for verbose output.

endef

ifeq (,$(filter BK \
								BK0 BK1 BK2 BK3 BK4 BK5 BK6 \
								clean-BK0 clean-BK1 clean-BK2 clean-BK3 clean-BK4 clean-BK5 clean-BK6 \
								clean clean-libs clean-kernels \
								realclean info help,$(MAKECMDGOALS)))
ifneq (,$(MAKECMDGOALS))
$(error ${LIBP_HELP_MSG})
endif
endif

ifndef LIBP_MAKETOP_LOADED
ifeq (,$(wildcard ../make.top))
$(error cannot locate ${PWD}/../make.top)
else
include ../make.top
endif
endif

#libraries
CORE_LIBS=mesh ogs core

.PHONY: all BK \
				BK1 BK2 BK3 BK4 BK5 BK6 \
				clean-BK1 clean-BK2 clean-BK3 clean-BK4 clean-BK5 clean-BK6 \
				clean clean-libs clean-kernels \
				realclean info help

all: BK

BK: BK1 BK2 BK3 BK4 BK5 BK6

libp_libs:
ifneq (,${verbose})
	${MAKE} -C ${LIBP_LIBS_DIR} $(CORE_LIBS) verbose=${verbose}
else
	@${MAKE} -C ${LIBP_LIBS_DIR} $(CORE_LIBS) --no-print-directory
endif

BK1: libp_libs
ifneq (,${verbose})
	${MAKE} -C $(@F) verbose=${verbose}
else
	@printf "%b" "$(SOL_COLOR)Building $(@F) benchmark $(NO_COLOR)\n";
	@${MAKE} -C $(@F) --no-print-directory
endif

BK2: libp_libs
ifneq (,${verbose})
	${MAKE} -C $(@F) verbose=${verbose}
else
	@printf "%b" "$(SOL_COLOR)Building $(@F) benchmark $(NO_COLOR)\n";
	@${MAKE} -C $(@F) --no-print-directory
endif

BK3: libp_libs
ifneq (,${verbose})
	${MAKE} -C $(@F) verbose=${verbose}
else
	@printf "%b" "$(SOL_COLOR)Building $(@F) benchmark $(NO_COLOR)\n";
	@${MAKE} -C $(@F) --no-print-directory
endif

BK4: libp_libs
ifneq (,${verbose})
	${MAKE} -C $(@F) verbose=${verbose}
else
	@printf "%b" "$(SOL_COLOR)Building $(@F) benchmark $(NO_COLOR)\n";
	@${MAKE} -C $(@F) --no-print-directory
endif

BK5: libp_libs
ifneq (,${verbose})
	${MAKE} -C $(@F) verbose=${verbose}
else
	@printf "%b" "$(SOL_COLOR)Building $(@F) benchmark $(NO_COLOR)\n";
	@${MAKE} -C $(@F) --no-print-directory
endif

BK6: libp_libs
ifneq (,${verbose})
	${MAKE} -C $(@F) verbose=${verbose}
else
	@printf "%b" "$(SOL_COLOR)Building $(@F) benchmark $(NO_COLOR)\n";
	@${MAKE} -C $(@F) --no-print-directory
endif

#cleanup
clean: clean-BK1 clean-BK2 clean-BK3 clean-BK4 clean-BK5 clean-BK6

clean-BK1:
	${MAKE} -C BK1 clean

clean-BK2:
	${MAKE} -C BK2 clean

clean-BK3:
	${MAKE} -C BK3 clean

clean-BK4:
	${MAKE} -C BK4 clean

clean-BK5:
	${MAKE} -C BK5 clean

clean-BK6:
	${MAKE} -C BK6 clean

clean-libs: clean
	${MAKE} -C ${LIBP_LIBS_DIR} clean

clean-kernels: clean-libs
	rm -rf ${LIBP_DIR}/.occa/

realclean: clean-libs
	${MAKE} -C ${OCCA_DIR} clean
	rm -rf ${LIBP_DIR}/.occa/

help:
	$(info $(value LIBP_HELP_MSG))
	@true

info:
	$(info LIBP_DIR  = $(LIBP_DIR))
	$(info LIBP_ARCH = $(LIBP_ARCH))
	$(info CXXFLAGS  = $(CXXFLAGS))
	@true
