#================================================================
#   Copyright (C) 2021 Sangfor Ltd. All rights reserved.
#   
#   File Name：Makefile
#   Author: Wenqiang Wang
#   Created Time:2021-10-30
#   Discription:
#
#================================================================

GPU_CUDA := ON

XFAST := ON
ZFAST := # ON

# ! For alternative flux finite difference by Tianhong Xu
# * Only for GPU now
SCFDM := ON
# ! Shock Capturing Methods
WENO := ON
MP :=  # ! Not correct
# ! Riemann Solvers
LF := ON

DFLAGS_LIST := SCFDM LF WENO MP

FREE_SURFACE := ON
PML :=
SOLVE_DISPLACEMENT := ON
Terrain_Smooth := ON 
DealWithFirstLayer := ON

LayeredStructureTerrain := ON
StructureTerrain := ON


SET_BASIN := 

FLOAT16 := 
SRCDIR := ./src


CCHOME := /usr
CUDAHOME := /public/software/cuda-10.0
MPIHOME := /public/software/openmpi-4.1.1-cuda.10
PROJHOME := /public/software/proj-8.1.0



CC := $(CCHOME)/bin/gcc

#General Compiler
ifdef GPU_CUDA
#GC := $(CUDAHOME)/bin/nvcc -rdc=true -maxrregcount=127 -arch=sm_70 -Xptxas=-v 
GC := $(CUDAHOME)/bin/nvcc -rdc=true -maxrregcount=64 -arch=sm_70 #-Xptxas=-v 
#GC := $(CUDAHOME)/bin/nvcc -rdc=true  -arch=sm_70 #-Xptxas=-v 
else
GC := $(CCHOME)/bin/g++ 
endif


LIBS := -L$(CUDAHOME)/lib64 -lcudart -lcublas
INCS := -I$(CUDAHOME)/include 

LIBS += -L$(MPIHOME)/lib -lmpi
INCS += -I$(MPIHOME)/include 


LIBS += -L$(PROJHOME)/lib -lproj
INCS += -I$(PROJHOME)/include  



OBJDIR := ./obj
BINDIR := ./bin


CFLAGS := -c -O3 
LFLAGS := -O3

GCFLAGS := 

ifdef GPU_CUDA
#LFLAGS += -Xptxas=-v 

#LFLAGS += -arch=sm_70 -rdc=true -Xptxas=-v 
#GCFLAGS += --fmad=false 
GCFLAGS += -x cu
endif

vpath

vpath % $(SRCDIR)
vpath % $(OBJDIR)
vpath % $(BINDIR)


DFLAGS_LIST += XFAST ZFAST GPU_CUDA FLOAT16 FREE_SURFACE PML SOLVE_DISPLACEMENT \
			   Terrain_Smooth DealWithFirstLayer SET_BASIN LayeredStructureTerrain StructureTerrain

DFLAGS := $(foreach flag,$(DFLAGS_LIST),$(if $($(flag)),-D$(flag)))


OBJS := cjson.o init_gpu.o init_grid.o init_MPI.o main.o getParams.o create_dir.o \
		run.o printInfo.o modelChecking.o  cpu_Malloc.o MPI_send_recv.o data_io.o \
		coord.o terrain.o  medium.o dealMedium.o crustMedium.o calc_CFL.o CJM.o \
		contravariant.o MPI_send_recv.o MPI_send_recv_FLOAT.o multiSource.o wave_deriv.o wave_rk.o \
		propagate.o freeSurface.o singleSource.o station.o PGV.o addMoment.o \
		init_pml_para.o pml_deriv.o pml_rk.o pml_freeSurface.o \
		alternative_flux_FD.o

OBJS := $(addprefix $(OBJDIR)/,$(OBJS))


$(BINDIR)/main: $(OBJS)
	$(GC) $(LFLAGS) $(LIBS) $^ -o $@


$(OBJDIR)/%.o : $(SRCDIR)/%.cpp
	$(GC) $(CFLAGS) $(DFLAGS) $(GCFLAGS) $(INCS)  $^ -o $@


$(OBJDIR)/%.o : $(SRCDIR)/%.c
	$(CC) $(CFLAGS) $^ -o $@

clean:
	-rm $(OBJDIR)/* -rf
	-rm $(BINDIR)/* -rf
	-rm output -rf

