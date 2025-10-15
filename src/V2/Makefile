######################################################################
# Choose your favorite C compiler
CC = gcc
CXX = g++
NVCC = nvcc

FLAG1 = -DNDEBUG

######################################################################
# Add your favorite C flags here.
CFLAGS = $(FLAG1) $(FLAG2) -pg -Wall -Wextra -std=c99
CXXFLAGS = $(FLAG1) -pg -Wall -Wextra -std=c++11
NVCCFLAGS = -O2 -std=c++11

######################################################################
# CUDA paths
CUDA_LIB_PATH = /usr/local/cuda/lib64
CUDA_INC_PATH = /usr/local/cuda/include

# Source files
C_SOURCES = convolve.c error.c pnmio.c pyramid.c selectGoodFeatures.c \
           storeFeatures.c klt.c klt_util.c writeFeatures.c
CU_SOURCES = trackFeatures.cu
CPP_SOURCES = main.cpp
EXAMPLES = example3.c

# Object files
C_OBJECTS = $(C_SOURCES:.c=.o)
CU_OBJECTS = $(CU_SOURCES:.cu=.o)
CPP_OBJECTS = $(CPP_SOURCES:.cpp=.o)
EXAMPLE_OBJECTS = $(EXAMPLES:.c=.o)

# Headers
HEADERS = base.h convolve.h cuda_utils.h error.h klt.h klt_util.h \
         pnmio.h pyramid.h

# Libraries
LIB = -L/usr/local/lib -L/usr/lib -L$(CUDA_LIB_PATH) -lcudart -lcuda -lm

.SUFFIXES:  .c .o .cpp .cu

# Default target - now builds example3 instead of main
all:  example3

# Compile C files
.c.o:
	$(CC) -c $(CFLAGS) $< -o $@

# Compile C++ files
.cpp.o:
	$(CXX) -c $(CXXFLAGS) $< -o $@

# Compile CUDA files
.cu.o:
	$(NVCC) -c $(NVCCFLAGS) -I. $< -o $@

# Main executable (for main.cpp) - uses CUDA
main: $(C_OBJECTS) $(CU_OBJECTS) main.o
	$(CXX) -pg -o $@ $^ $(LIB)

# Library
lib: $(C_OBJECTS) $(CU_OBJECTS)
	rm -f libklt.a
	ar ruv libklt.a $^
	ranlib libklt.a

# Example 3 - now includes CUDA objects
example3: $(C_OBJECTS) $(CU_OBJECTS) example3.o
	$(CXX) -pg -o $@ $^ $(LIB)

# Dependencies
depend:
	makedepend $(C_SOURCES) $(EXAMPLES) main.cpp

clean:
	rm -f *.o *.a main example3 $(EXAMPLES:.c=) *.tar *.tar.gz libklt.a \
	      feat*.ppm features.ft features.txt gmon.out

distclean: clean
	rm -f *.png *.pdf *.dot *_analysis.txt

######################################################################
# Profiling and Visualization Rules

# External tools
GPROF2DOT = ./gprof2dot.py
DOT = dot

# Select example (default = example3)
EXAMPLE ?= example3
PROFILE_TXT = $(EXAMPLE)_analysis.txt
PROFILE_DOT = $(EXAMPLE)_analysis.dot
PROFILE_PNG = $(EXAMPLE)_analysis.png
PROFILE_PDF = $(EXAMPLE)_analysis.pdf

# Download gprof2dot if missing
$(GPROF2DOT):
	wget https://raw.githubusercontent.com/jrfonseca/gprof2dot/master/gprof2dot.py
	chmod +x gprof2dot.py

# Run example to generate gmon.out and gprof text
gprof: $(EXAMPLE)
	./$(EXAMPLE)
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

.PHONY: all lib clean distclean depend gprof dot png pdf