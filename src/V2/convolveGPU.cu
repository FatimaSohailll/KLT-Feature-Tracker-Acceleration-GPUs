/*********************************************************************
 * convolveGPU.cu
 *********************************************************************/

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <assert.h>
#include "convolveGPU.h"
#include "convolve.h"
#include "error.h"  // For KLTError if needed

#define MAX_KERNEL_WIDTH 71

// ===============================
// Device kernels
// ===============================

__global__ void convolveHorizKernel(const float* imgin, float* imgout,
                                    int ncols, int nrows,
                                    const float* kernel, int kWidth) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int radius = kWidth / 2;

  if (x >= ncols || y >= nrows) return;

  float sum = 0.0f;

  if (x >= radius && x < ncols - radius) {
    for (int k = -radius; k <= radius; k++) {
      sum += imgin[y * ncols + (x + k)] * kernel[radius - k];
    }
    imgout[y * ncols + x] = sum;
  } else {
    imgout[y * ncols + x] = 0.0f;
  }
}


__global__ void convolveVertKernel(const float* imgin, float* imgout,
                                   int ncols, int nrows,
                                   const float* kernel, int kWidth) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int radius = kWidth / 2;

  if (x >= ncols || y >= nrows) return;

  float sum = 0.0f;

  if (y >= radius && y < nrows - radius) {
    for (int k = -radius; k <= radius; k++) {
      sum += imgin[(y + k) * ncols + x] * kernel[radius - k];
    }
    imgout[y * ncols + x] = sum;
  } else {
    imgout[y * ncols + x] = 0.0f;
  }
}


// ===============================
// Host wrappers
// ===============================

void convolveImageHorizGPU(
  _KLT_FloatImage imgin,
  ConvolutionKernel kernel,
  _KLT_FloatImage imgout)
{
  int ncols = imgin->ncols, nrows = imgin->nrows;
  int imgSize = ncols * nrows * sizeof(float);
  int kSize = kernel.width * sizeof(float);

  float *d_in, *d_out, *d_kernel;

  cudaMalloc(&d_in, imgSize);
  cudaMalloc(&d_out, imgSize);
  cudaMalloc(&d_kernel, kSize);

  cudaMemcpy(d_in, imgin->data, imgSize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_kernel, kernel.data, kSize, cudaMemcpyHostToDevice);

  dim3 block(16, 16);
  dim3 grid((ncols + block.x - 1) / block.x,
            (nrows + block.y - 1) / block.y);

  convolveHorizKernel<<<grid, block>>>(d_in, d_out, ncols, nrows, d_kernel, kernel.width);
  cudaDeviceSynchronize();

  cudaMemcpy(imgout->data, d_out, imgSize, cudaMemcpyDeviceToHost);

  cudaFree(d_in);
  cudaFree(d_out);
  cudaFree(d_kernel);
}


void convolveImageVertGPU(
  _KLT_FloatImage imgin,
  ConvolutionKernel kernel,
  _KLT_FloatImage imgout)
{
  int ncols = imgin->ncols, nrows = imgin->nrows;
  int imgSize = ncols * nrows * sizeof(float);
  int kSize = kernel.width * sizeof(float);

  float *d_in, *d_out, *d_kernel;

  cudaMalloc(&d_in, imgSize);
  cudaMalloc(&d_out, imgSize);
  cudaMalloc(&d_kernel, kSize);

  cudaMemcpy(d_in, imgin->data, imgSize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_kernel, kernel.data, kSize, cudaMemcpyHostToDevice);

  dim3 block(16, 16);
  dim3 grid((ncols + block.x - 1) / block.x,
            (nrows + block.y - 1) / block.y);

  convolveVertKernel<<<grid, block>>>(d_in, d_out, ncols, nrows, d_kernel, kernel.width);
  cudaDeviceSynchronize();

  cudaMemcpy(imgout->data, d_out, imgSize, cudaMemcpyDeviceToHost);

  cudaFree(d_in);
  cudaFree(d_out);
  cudaFree(d_kernel);
}


void convolveSeparateGPU(
  _KLT_FloatImage imgin,
  ConvolutionKernel horiz_kernel,
  ConvolutionKernel vert_kernel,
  _KLT_FloatImage imgout)
{
  _KLT_FloatImage tmp = _KLTCreateFloatImage(imgin->ncols, imgin->nrows);

  convolveImageHorizGPU(imgin, horiz_kernel, tmp);
  convolveImageVertGPU(tmp, vert_kernel, imgout);

  _KLTFreeFloatImage(tmp);
}
