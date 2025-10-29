######################################################################
# Compiler settings
CC = gcc
NVCC = nvcc

######################################################################
# Flags
FLAG1 = -DNDEBUG
CFLAGS = $(FLAG1) -pg
NVCCFLAGS = -std=c++11 -I. -arch=sm_86

######################################################################
# CUDA path
CUDA_LIB_PATH = /opt/nvidia/hpc_sdk/Linux_x86_64/25.7/cuda/12.9/targets/x86_64-linux/lib
CUDA_INC_PATH = /opt/nvidia/hpc_sdk/Linux_x86_64/25.7/cuda/12.9/targets/x86_64-linux/include

######################################################################
# Source files
ARCH_CPU = convolve.c error.c pnmio.c pyramid.c selectGoodFeatures.c \
           storeFeatures.c trackFeatures.c klt.c klt_util.c writeFeatures.c

ARCH_GPU = error.c pnmio.c pyramidGPU.c selectGoodFeaturesGPU.c \
           storeFeaturesGPU.c kltGPU.c klt_utilGPU.c writeFeaturesGPU.c

CU_SOURCES = trackFeaturesGPU.cu convolveGPU.cu
EXAMPLES = example3.c

EXAMPLE3_GPU = example3GPU.c

######################################################################
# Libraries
LIB = -L/usr/local/lib -L/usr/lib -L$(CUDA_LIB_PATH) -lcudart -lcuda -lm

######################################################################
# Suffix rules
.SUFFIXES: .c .o .cu

######################################################################
# Default target
all: cpu

######################################################################
# Compile rules
.c.o:
	$(CC) -c $(CFLAGS) $< -o $@

.cu.o:
	$(NVCC) -c $(NVCCFLAGS) $< -o $@

######################################################################
# CPU target
cpu: $(ARCH_CPU:.c=.o)
	ar ruv libklt.a $(ARCH_CPU:.c=.o)
	$(CC) $(CFLAGS) -o example3_cpu example3.c -L. -lklt $(LIB)
	@echo "Built CPU version: example3_cpu"

######################################################################
# GPU target
gpu: $(ARCH_GPU:.c=.o) $(CU_SOURCES:.cu=.o)
	ar ruv libklt_gpu.a $(ARCH_GPU:.c=.o) $(CU_SOURCES:.cu=.o)
	$(CC) $(CFLAGS) -o example3_gpu $(EXAMPLE3_GPU) -L. -lklt_gpu $(LIB)
	@echo "Built GPU version: example3_gpu using $(EXAMPLE3_GPU)"

######################################################################
# Run targets
run_cpu: cpu
	./example3_cpu

run_gpu: gpu
	LD_LIBRARY_PATH=$(CUDA_LIB_PATH):$$LD_LIBRARY_PATH ./example3_gpu

######################################################################
# Clean targets
clean:
	rm -f *.o *.a example3_cpu example3_gpu $(EXAMPLES:.c=) \
	      feat/feat*.ppm feat/features.ft feat/features.txt feat/features2.ft feat/features2.txt gmon.out *_analysis.* *.png *.pdf *.dot

######################################################################
# Profiling and Visualization Rules

GPROF2DOT = ./gprof2dot.py
DOT = dot

EXAMPLE ?= example3_cpu
PROFILE_TXT = $(EXAMPLE)_analysis.txt
PROFILE_DOT = $(EXAMPLE)_analysis.dot
PROFILE_PNG = $(EXAMPLE)_analysis.png
PROFILE_PDF = $(EXAMPLE)_analysis.pdf

######################################################################
# Download gprof2dot if not available
$(GPROF2DOT):
	wget -q https://raw.githubusercontent.com/jrfonseca/gprof2dot/master/gprof2dot.py
	chmod +x gprof2dot.py

######################################################################
# Profiling - FIXED: removed comment from execution line
gprof: $(EXAMPLE)
	@echo "Running gprof for $(EXAMPLE)..."
	./$(EXAMPLE)
	gprof ./$(EXAMPLE) gmon.out > $(PROFILE_TXT)
	@echo "Profiling data saved to $(PROFILE_TXT)"

dot: $(PROFILE_TXT) $(GPROF2DOT)
	$(GPROF2DOT) -f prof $(PROFILE_TXT) -o $(PROFILE_DOT)
	@echo "Generated $(PROFILE_DOT)"

png: dot
	$(DOT) -Tpng $(PROFILE_DOT) -o $(PROFILE_PNG)
	@echo "Generated $(PROFILE_PNG)"

pdf: dot
	$(DOT) -Tpdf $(PROFILE_DOT) -o $(PROFILE_PDF)
	@echo "Generated $(PROFILE_PDF)"

######################################################################
.PHONY: all cpu gpu clean gprof dot png pdf run_cpu run_gpu