/*********************************************************************
 * convolveGPU.cu
 *********************************************************************/

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <assert.h>
#include "convolveGPU.h"
#include "error.h"  // For KLTError if needed

#define MAX_KERNEL_WIDTH 71

/* Kernels */
static ConvolutionKernel gauss_kernel;
static ConvolutionKernel gaussderiv_kernel;
static float sigma_last = -10.0;


// ===============================
// Device kernels
// ===============================



void _KLTToFloatImage(
  KLT_PixelType *img,
  int ncols, int nrows,
  _KLT_FloatImage floatimg)
{
  KLT_PixelType *ptrend = img + ncols*nrows;
  float *ptrout = floatimg->data;

  /* Output image must be large enough to hold result */
  assert(floatimg->ncols >= ncols);
  assert(floatimg->nrows >= nrows);

  floatimg->ncols = ncols;
  floatimg->nrows = nrows;

  while (img < ptrend)  *ptrout++ = (float) *img++;
}

static void computeKernelsGPU(
  float sigma,
  ConvolutionKernel *gauss,
  ConvolutionKernel *gaussderiv)
{
  const float factor = 0.01f;   /* for truncating tail */
  int i;

  assert(MAX_KERNEL_WIDTH % 2 == 1);
  assert(sigma >= 0.0);

  /* Compute kernels, and automatically determine widths */
  {
    const int hw = MAX_KERNEL_WIDTH / 2;
    float max_gauss = 1.0f, max_gaussderiv = (float) (sigma*exp(-0.5f));
	
    /* Compute gauss and deriv */
    for (i = -hw ; i <= hw ; i++)  {
      gauss->data[i+hw]      = (float) exp(-i*i / (2*sigma*sigma));
      gaussderiv->data[i+hw] = -i * gauss->data[i+hw];
    }

    /* Compute widths */
    gauss->width = MAX_KERNEL_WIDTH;
    for (i = -hw ; fabs(gauss->data[i+hw] / max_gauss) < factor ; 
         i++, gauss->width -= 2);
    gaussderiv->width = MAX_KERNEL_WIDTH;
    for (i = -hw ; fabs(gaussderiv->data[i+hw] / max_gaussderiv) < factor ; 
         i++, gaussderiv->width -= 2);
    if (gauss->width == MAX_KERNEL_WIDTH || 
        gaussderiv->width == MAX_KERNEL_WIDTH)
      KLTError("(_computeKernels) MAX_KERNEL_WIDTH %d is too small for "
               "a sigma of %f", MAX_KERNEL_WIDTH, sigma);
  }

  /* Shift if width less than MAX_KERNEL_WIDTH */
  for (i = 0 ; i < gauss->width ; i++)
    gauss->data[i] = gauss->data[i+(MAX_KERNEL_WIDTH-gauss->width)/2];
  for (i = 0 ; i < gaussderiv->width ; i++)
    gaussderiv->data[i] = gaussderiv->data[i+(MAX_KERNEL_WIDTH-gaussderiv->width)/2];
  /* Normalize gauss and deriv */
  {
    const int hw = gaussderiv->width / 2;
    float den;
			
    den = 0.0;
    for (i = 0 ; i < gauss->width ; i++)  den += gauss->data[i];
    for (i = 0 ; i < gauss->width ; i++)  gauss->data[i] /= den;
    den = 0.0;
    for (i = -hw ; i <= hw ; i++)  den -= i*gaussderiv->data[i+hw];
    for (i = -hw ; i <= hw ; i++)  gaussderiv->data[i+hw] /= den;
  }

  sigma_last = sigma;
}

void _KLTGetKernelWidths(
  float sigma,
  int *gauss_width,
  int *gaussderiv_width)
{
  computeKernelsGPU(sigma, &gauss_kernel, &gaussderiv_kernel);

  printf("[DEBUG] sigma = %.4f | gauss_width = %d | gaussderiv_width = %d\n",
         sigma, gauss_kernel.width, gaussderiv_kernel.width);

  *gauss_width = gauss_kernel.width;
  *gaussderiv_width = gaussderiv_kernel.width;
}

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

void _KLTComputeGradients(
  _KLT_FloatImage img,
  float sigma,
  _KLT_FloatImage gradx,
  _KLT_FloatImage grady)
{
				
  /* Output images must be large enough to hold result */
  assert(gradx->ncols >= img->ncols);
  assert(gradx->nrows >= img->nrows);
  assert(grady->ncols >= img->ncols);
  assert(grady->nrows >= img->nrows);

  /* Compute kernels, if necessary */
  if (fabs(sigma - sigma_last) > 0.05)
    computeKernelsGPU(sigma, &gauss_kernel, &gaussderiv_kernel);
	
  convolveSeparateGPU(img, gaussderiv_kernel, gauss_kernel, gradx);
  convolveSeparateGPU(img, gauss_kernel, gaussderiv_kernel, grady);

}

/*********************************************************************
 * _KLTComputeSmoothedImage
 */

void _KLTComputeSmoothedImage(
  _KLT_FloatImage img,
  float sigma,
  _KLT_FloatImage smooth)
{
  /* Output image must be large enough to hold result */
  assert(smooth->ncols >= img->ncols);
  assert(smooth->nrows >= img->nrows);

  /* Compute kernel, if necessary; gauss_deriv is not used */
  if (fabs(sigma - sigma_last) > 0.05)
    computeKernelsGPU(sigma, &gauss_kernel, &gaussderiv_kernel);

  convolveSeparateGPU(img, gauss_kernel, gauss_kernel, smooth);
}