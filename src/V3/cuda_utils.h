#include <cuda_runtime.h>
#include <stdlib.h>		/* malloc() */
#include <stdio.h>		/* fflush() */
#include <cuda.h>

#ifndef _CUDA_UTILS_H_
#define _CUDA_UTILS_H_

#ifdef __cplusplus
extern "C" {
#endif

// MACRO for error checking
#define CUDA_CHECK(call) \
do { \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
        fprintf(stderr, "CUDA Error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

#ifdef __cplusplus
}
#endif

#endif