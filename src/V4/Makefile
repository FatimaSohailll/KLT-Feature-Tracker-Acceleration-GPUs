######################################################################
# Compiler: use NVIDIA HPC SDK (OpenACC)
CC = nvc

######################################################################
# Flags
# -acc            → enable OpenACC
# -Minfo=accel    → show accelerator optimization info
# -fast           → aggressive optimization
# -DNDEBUG        → disable asserts (optional)
CFLAGS = -acc -Minfo=accel -fast -DNDEBUG
CPU_CFLAGS = -pg -fast -DNDEBUG
GPU_CFLAGS = -acc -Minfo=accel -fast -DNDEBUG

######################################################################
# Source files
EXAMPLES = example3.c
ARCHCPU = convolve.c error.c pnmio.c pyramid.c selectGoodFeatures.c \
       storeFeatures.c trackFeatures.c klt.c klt_util.c writeFeatures.c
ARCHGPU = convolveGPU.c error.c pnmio.c pyramid.c selectGoodFeatures.c \
       storeFeatures.c trackFeatures.c klt.c klt_util.c writeFeatures.c

LIB = -L/usr/local/lib -L/usr/lib

######################################################################
# Runtime parameters with defaults
DATASET ?= images_provided
FEATURES ?= 150
FRAMES ?= 10

# Handle command line parameters for run targets
ifeq ($(words $(MAKECMDGOALS)),4)
DATASET := $(word 2, $(MAKECMDGOALS))
FEATURES := $(word 3, $(MAKECMDGOALS))
FRAMES := $(word 4, $(MAKECMDGOALS))
# prevent make from treating them as targets
$(eval $(DATASET):;@:)
$(eval $(FEATURES):;@:)
$(eval $(FRAMES):;@:)
endif

.SUFFIXES: .c .o

######################################################################
# Build everything
all: klt_cpu klt_gpu

######################################################################
# Compile .c → .o for CPU
cpu_%.o: %.c
	$(CC) -c $(CPU_CFLAGS) -o $@ $<

######################################################################
# Compile .c → .o for GPU  
gpu_%.o: %.c
	$(CC) -c $(GPU_CFLAGS) -o $@ $<

######################################################################
# Build CPU version
klt_cpu: $(addprefix cpu_, $(ARCHCPU:.c=.o))
	rm -f libklt_cpu.a
	ar ruv libklt_cpu.a $^
	$(CC) $(CPU_CFLAGS) -o example3_cpu example3.c -L. -lklt_cpu $(LIB) -lm

######################################################################
# Build GPU version
klt_gpu: $(addprefix gpu_, $(ARCHGPU:.c=.o))
	rm -f libklt_gpu.a
	ar ruv libklt_gpu.a $^
	$(CC) $(GPU_CFLAGS) -o example3_gpu example3.c -L. -lklt_gpu $(LIB) -lm

######################################################################
# Run targets with parameters
run_cpu: klt_cpu
	@echo "Running CPU version with dataset=$(DATASET), features=$(FEATURES), frames=$(FRAMES)"
	./example3_cpu $(DATASET) $(FEATURES) $(FRAMES)

run_gpu: klt_gpu
	@echo "Running GPU version with dataset=$(DATASET), features=$(FEATURES), frames=$(FRAMES)"
	./example3_gpu $(DATASET) $(FEATURES) $(FRAMES)

######################################################################
# Individual object file rules to handle special cases
cpu_convolve.o: convolve.c
	$(CC) -c $(CPU_CFLAGS) -o $@ $<

gpu_convolveGPU.o: convolveGPU.c
	$(CC) -c $(GPU_CFLAGS) -o $@ $<

######################################################################
# Clean rule
clean:
	rm -f *.o *.a example3_cpu example3_gpu $(EXAMPLES:.c=) *.tar *.tar.gz \
	      libklt_cpu.a libklt_gpu.a cpu_*.o gpu_*.o feat*.ppm features.ft features.txt

nsys:
	# Example with actual arguments
	nsys profile --trace=openacc,nvtx,osrt --stats=true ./example3_gpu $(DATASET) $(FEATURES) $(FRAMES)
######################################################################
# Debug targets to see what's being built
debug:
	@echo "CPU objects: $(addprefix cpu_, $(ARCHCPU:.c=.o))"
	@echo "GPU objects: $(addprefix gpu_, $(ARCHGPU:.c=.o))"
	@echo "Default parameters: DATASET=$(DATASET), FEATURES=$(FEATURES), FRAMES=$(FRAMES)"

######################################################################
# Help
help:
	nsys profile --trace=openacc,nvtx,osrt --stats=true ./example3_gpu 
	@echo "Available targets:"
	@echo "  all       - Build both CPU and GPU versions"
	@echo "  klt_cpu   - Build CPU version only"
	@echo "  klt_gpu   - Build GPU version only"
	@echo "  run_cpu   - Build and run CPU version with default parameters"
	@echo "  run_gpu   - Build and run GPU version with default parameters"
	@echo "  clean     - Clean all built files"
	@echo "  debug     - Show build configuration"
	@echo ""
	@echo "Usage with custom parameters:"
	@echo "  make run_cpu <dataset> <features> <frames>"
	@echo "  make run_gpu <dataset> <features> <frames>"
	@echo ""
	@echo "Examples:"
	@echo "  make run_cpu images_provided 100 5"
	@echo "  make run_gpu my_dataset 200 15"
	@echo ""
	@echo "Current defaults: DATASET=$(DATASET), FEATURES=$(FEATURES), FRAMES=$(FRAMES)"

GPROF2DOT = ./gprof2dot.py
DOT = dot

EXAMPLE ?= example3_cpu
PROFILE_TXT = $(EXAMPLE)_analysis.txt
PROFILE_DOT = $(EXAMPLE)_analysis.dot
PROFILE_PNG = $(EXAMPLE)_analysis.png
PROFILE_PDF = $(EXAMPLE)_analysis.pdf

gprof: ./$(EXAMPLE)
	gprof ./$(EXAMPLE) gmon.out > $(PROFILE_TXT)

# Convert gprof text to DOT
dot: $(PROFILE_TXT) $(GPROF2DOT)
	$(GPROF2DOT) -f prof $(PROFILE_TXT) -o $(PROFILE_DOT)

# Convert DOT to PNG
png: dot
	$(DOT) -Tpng $(PROFILE_DOT) -o $(PROFILE_PNG)

# Convert DOT to PDF
pdf: dot
	$(DOT) -Tpdf $(PROFILE_DOT) -o $(PROFILE_PDF)
