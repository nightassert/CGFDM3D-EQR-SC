#================================================================
#   ESS, Southern University of Science and Technology
#
#   File Name: Makefile
#   Discription: Unified build for CGFDM and WENO (GPU or CPU only)
#
#   Both numerical schemes are compiled into one executable.  Select the
#   scheme at run time with "scheme": "CGFDM", "WENO-JS" or "WENO-Z" in
#   params.json.
#
#   Targets:
#       make CUDA_ARCH=sm_80        # A100 (CUDA 13)
#       make CUDA_ARCH=sm_86        # RTX 3060 / RTX 30-series
#       make CUDA_ARCH=sm_89        # RTX 40-series
#       make CUDA_ARCH=sm_90        # H100
#       make GPU_CUDA=OFF           # CPU-only build for MPI clusters
#
#   The CPU build drops all CUDA code and uses the plain C++ loops of
#   every kernel; both schemes work (the WENO kernels have full CPU
#   paths).  It needs only a C++ compiler plus MPI and PROJ.
#
#   Override CUDA_HOME, MPI_HOME or PROJ_HOME if they are installed
#   somewhere other than the default locations.
#================================================================

GPU_CUDA ?= ON

# Thread block layout.  Exactly one of XFAST / ZFAST must be enabled.
XFAST := ON
ZFAST :=

FREE_SURFACE := ON
PML := ON

SOLVE_PGA :=
SOLVE_DISPLACEMENT := ON

Terrain_Smooth := ON
DealWithFirstLayer := ON
LayeredStructureTerrain := ON
StructureTerrain := ON
SET_BASIN :=

#----------------------------------------------------------------
# Toolchain locations
#----------------------------------------------------------------
CUDA_HOME ?= /usr/local/cuda
MPI_HOME  ?= /usr/lib/x86_64-linux-gnu/openmpi
PROJ_HOME ?= /usr

CC  ?= gcc
CXX ?= g++

# Target architecture.  CUDA 13 no longer supports sm_70 or older.
# The fat binary embeds both SASS and PTX so it also runs on newer GPUs.
CUDA_ARCH ?= sm_80
CUDA_ARCH_NUM := $(patsubst sm_%,%,$(CUDA_ARCH))
GENCODE := -gencode arch=compute_$(CUDA_ARCH_NUM),code=[sm_$(CUDA_ARCH_NUM),compute_$(CUDA_ARCH_NUM)]

ifeq ($(XFAST),ON)
ifeq ($(ZFAST),ON)
$(error Enable only one of XFAST and ZFAST)
endif
endif

#----------------------------------------------------------------
# Sources and objects
#----------------------------------------------------------------
SRCDIR := ./src
OBJDIR := ./obj
BINDIR := ./bin

DFLAGS_LIST := GPU_CUDA XFAST ZFAST FREE_SURFACE PML SOLVE_PGA \
               SOLVE_DISPLACEMENT Terrain_Smooth DealWithFirstLayer \
               LayeredStructureTerrain StructureTerrain SET_BASIN

DFLAGS := $(foreach flag,$(DFLAGS_LIST),$(if $($(flag)),-D$(flag)))

INCS := -I$(MPI_HOME)/include \
        -I$(PROJ_HOME)/include

# GPU mode: compile and link with nvcc (relocatable device code).
# CPU mode: plain C++ compiler, no CUDA headers or libraries.
ifeq ($(GPU_CUDA),ON)
CXX           := $(CUDA_HOME)/bin/nvcc
CXX_FLAGS     := -c -O3 -std=c++17 -x cu -rdc=true $(GENCODE) $(INCS) -I$(CUDA_HOME)/include
CXX_LINK      := -O3 -rdc=true $(GENCODE)
CXX_LIBS      := -L$(CUDA_HOME)/lib64 -lcudart
else
CXX_FLAGS     := -c -O3 -std=c++17 $(INCS)
CXX_LINK      := -O3
CXX_LIBS      :=
endif

LIBS := $(CXX_LIBS) \
        -L$(MPI_HOME)/lib -lmpi \
        -L$(PROJ_HOME)/lib -lproj -lm

CFLAGS := -c -O3 -std=c99

OBJS := cjson.o init_gpu.o init_grid.o init_MPI.o main.o getParams.o create_dir.o \
        run.o printInfo.o modelChecking.o cpu_Malloc.o MPI_send_recv.o data_io.o \
        coord.o terrain.o medium.o dealMedium.o crustMedium.o calc_CFL.o CJM.o \
        contravariant.o MPI_send_recv_FLOAT.o multiSource.o wave_deriv.o wave_rk.o \
        propagate.o freeSurface.o singleSource.o station.o PGV.o addMoment.o \
        init_pml_para.o pml_deriv.o pml_rk.o pml_freeSurface.o \
        weno_pml.o alternative_flux_FD.o exp_decay_absorb.o solve_displacement.o wave_rk_tvd.o

OBJS := $(addprefix $(OBJDIR)/,$(OBJS))

#----------------------------------------------------------------
# Build rules
#----------------------------------------------------------------
.PHONY: all clean

all: $(BINDIR)/main

$(BINDIR)/main: $(OBJS) | $(BINDIR)
	$(CXX) $(CXX_LINK) $(LIBS) $^ -o $@

# Headers are not tracked individually; make every object depend on all of
# them so a change to struct.h (e.g. a new PARAMS field) cannot leave stale
# objects linked against a different layout.
$(OBJS): $(wildcard $(SRCDIR)/*.h)

$(OBJDIR)/%.o: $(SRCDIR)/%.cpp | $(OBJDIR)
	$(CXX) $(CXX_FLAGS) $(DFLAGS) $< -o $@

$(OBJDIR)/%.o: $(SRCDIR)/%.c | $(OBJDIR)
	$(CC) $(CFLAGS) $< -o $@

$(OBJDIR):
	mkdir -p $(OBJDIR)

$(BINDIR):
	mkdir -p $(BINDIR)

clean:
	rm -rf $(OBJDIR) $(BINDIR) output
