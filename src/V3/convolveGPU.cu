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
// device constant memory for kernels
__constant__ float d_kernel[MAX_KERNEL_WIDTH];
__constant__ float d_kernel_deriv[MAX_KERNEL_WIDTH];

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
  *gauss_width = gauss_kernel.width;
  *gaussderiv_width = gaussderiv_kernel.width;
}

__global__ void convolveHorizShared(
    const float* __restrict__ imgin,
    float* __restrict__ imgout,
    int ncols, int nrows,
    int kWidth,
    int use_deriv)
{
    extern __shared__ float tile[];  // dynamic shared memory

    int R = kWidth / 2;
    int BLOCK_W = blockDim.x;
    int BLOCK_H = blockDim.y;

    // Shared tile dimensions (with halo on both sides)
    int TILE_W = BLOCK_W + 2 * R;
    int TILE_H = BLOCK_H;

    // Global coordinates of the FIRST pixel this block outputs
    int out_x = blockIdx.x * BLOCK_W + threadIdx.x;
    int out_y = blockIdx.y * BLOCK_H + threadIdx.y;

    // To cooperatively load tile, we use a 2D loop over TILE_W x TILE_H
    // with all 256 threads helping until tile is filled.
    for (int dy = threadIdx.y; dy < TILE_H; dy += BLOCK_H) {
        for (int dx = threadIdx.x; dx < TILE_W; dx += BLOCK_W) {

            // Compute the global coordinates of this tile element
            int gx = blockIdx.x * BLOCK_W + dx - R;
            int gy = blockIdx.y * BLOCK_H + dy;  // same y

            // Boundary clamp in X
            gx = max(0, min(gx, ncols - 1));

            // Valid because this is a horizontal convolution tile (only width expands)
            gy = max(0, min(gy, nrows - 1));

            // Load into shared memory
            tile[dy * TILE_W + dx] = imgin[gy * ncols + gx];
        }
    }

    __syncthreads();

    // If output coordinate is outside image grid → nothing to write
    if (out_x >= ncols || out_y >= nrows) return;

    float sum = 0.0f;
    int tile_y = threadIdx.y;
    int tile_x = threadIdx.x + R;     // offset: skip left halo

    // Horizontal convolution entirely from shared memory
    for (int k = -R; k <= R; k++) {
    	float coeff = use_deriv ? d_kernel_deriv[R - k] : d_kernel[R - k];
        sum += tile[tile_y * TILE_W + (tile_x + k)] * coeff;
    }
        
    imgout[out_y * ncols + out_x] = sum;
}



__global__ void convolveVertShared(
    const float* __restrict__ imgin,
    float* __restrict__ imgout,
    int ncols, int nrows,
    int kWidth,
    int use_deriv)
{
    extern __shared__ float tile[];  // dynamic shared memory

    int R = kWidth / 2;
    int BLOCK_W = blockDim.x;
    int BLOCK_H = blockDim.y;

    // For vertical convolution: halo extends in Y direction
    int TILE_W = BLOCK_W;
    int TILE_H = BLOCK_H + 2 * R;

    // Global output coords
    int out_x = blockIdx.x * BLOCK_W + threadIdx.x;
    int out_y = blockIdx.y * BLOCK_H + threadIdx.y;

    // Cooperative load into shared memory
    for (int dy = threadIdx.y; dy < TILE_H; dy += BLOCK_H) {
        for (int dx = threadIdx.x; dx < TILE_W; dx += BLOCK_W) {

            // Map tile coordinates -> global image coords
            int gx = blockIdx.x * BLOCK_W + dx;
            int gy = blockIdx.y * BLOCK_H + dy - R;

            // Clamp in vertical direction
            gx = max(0, min(gx, ncols - 1));
            gy = max(0, min(gy, nrows - 1));

            tile[dy * TILE_W + dx] = imgin[gy * ncols + gx];
        }
    }

    __syncthreads();

    // Boundary guard
    if (out_x >= ncols || out_y >= nrows) return;

    float sum = 0.0f;

    int tile_y = threadIdx.y + R;  // skip top halo
    int tile_x = threadIdx.x;

    // Vertical convolution from shared memory
    for (int k = -R; k <= R; k++) {
    	float coeff = use_deriv ? d_kernel_deriv[R - k] : d_kernel[R - k];
        sum += tile[(tile_y + k) * TILE_W + tile_x] * coeff;
    }

    imgout[out_y * ncols + out_x] = sum;
}


// ===============================
// Host wrappers
// ===============================

void convolveImageHorizGPU(
  _KLT_FloatImage imgin,
  ConvolutionKernel kernel,
  _KLT_FloatImage imgout,
  int use_deriv)
{
  int ncols = imgin->ncols, nrows = imgin->nrows;
  int imgSize = ncols * nrows * sizeof(float);
  int kSize = kernel.width * sizeof(float);

  float *d_in, *d_out;

  cudaMalloc(&d_in, imgSize);
  cudaMalloc(&d_out, imgSize);
  cudaMemcpy(d_in, imgin->data, imgSize, cudaMemcpyHostToDevice);
  
  // upload correct kernel into constant memory (derivative or gaussian)
  if (use_deriv) {
      cudaMemcpyToSymbol(d_kernel_deriv, kernel.data, kSize);
  } else {
      cudaMemcpyToSymbol(d_kernel, kernel.data, kSize);
  }
  
  dim3 block(32, 8);
  dim3 grid((ncols + block.x - 1) / block.x,
            (nrows + block.y - 1) / block.y);

  int R = kernel.width / 2;
  int TILE_W = block.x + 2*R;
  int TILE_H = block.y;
  int sharedBytes = TILE_W * TILE_H * sizeof(float);

convolveHorizShared<<<grid, block, sharedBytes>>>(
    d_in, d_out,
    ncols, nrows,
    kernel.width,
    use_deriv
);

  cudaMemcpy(imgout->data, d_out, imgSize, cudaMemcpyDeviceToHost);

  cudaFree(d_in);
  cudaFree(d_out);
}


void convolveImageVertGPU(
  _KLT_FloatImage imgin,
  ConvolutionKernel kernel,
  _KLT_FloatImage imgout,
  int use_deriv)
{
  int ncols = imgin->ncols, nrows = imgin->nrows;
  int imgSize = ncols * nrows * sizeof(float);
  int kSize = kernel.width * sizeof(float);

  float *d_in, *d_out;

  cudaMalloc(&d_in, imgSize);
  cudaMalloc(&d_out, imgSize);
  cudaMemcpy(d_in, imgin->data, imgSize, cudaMemcpyHostToDevice);
  
  // upload correct kernel into constant memory (derivative or gaussian)
  if (use_deriv) {
      cudaMemcpyToSymbol(d_kernel_deriv, kernel.data, kSize);
  } else {
      cudaMemcpyToSymbol(d_kernel, kernel.data, kSize);
  }
  
  dim3 block(16, 16);
  dim3 grid((ncols + block.x - 1) / block.x,
            (nrows + block.y - 1) / block.y);

 int R = kernel.width / 2;
int TILE_W = block.x + 2*R;
int TILE_H = block.y;
int sharedBytes = TILE_W * TILE_H * sizeof(float);

convolveVertShared<<<grid, block, sharedBytes>>>(
    d_in, d_out,
    ncols, nrows,
    kernel.width,
    use_deriv
);

  cudaMemcpy(imgout->data, d_out, imgSize, cudaMemcpyDeviceToHost);

  cudaFree(d_in);
  cudaFree(d_out);
}


void convolveSeparateGPU(
  _KLT_FloatImage imgin,
  ConvolutionKernel horiz_kernel,
  ConvolutionKernel vert_kernel,
  _KLT_FloatImage imgout)
{
    // allocate one tmp float-image for device-only intermediate
    int ncols = imgin->ncols, nrows = imgin->nrows;
    size_t imgSize = (size_t)ncols * (size_t)nrows * sizeof(float);

    // device buffers: d_in (source), d_tmp (after horiz), d_out (final)
    float *d_in = nullptr, *d_tmp = nullptr, *d_out = nullptr;

    cudaError_t cerr;

    // allocate device buffers once
    cerr = cudaMalloc(&d_in, imgSize);
    if (cerr != cudaSuccess) {
        fprintf(stderr, "cudaMalloc d_in failed: %s\n", cudaGetErrorString(cerr));
        return;
    }
    cerr = cudaMalloc(&d_tmp, imgSize);
    if (cerr != cudaSuccess) {
        fprintf(stderr, "cudaMalloc d_tmp failed: %s\n", cudaGetErrorString(cerr));
        cudaFree(d_in);
        return;
    }
    cerr = cudaMalloc(&d_out, imgSize);
    if (cerr != cudaSuccess) {
        fprintf(stderr, "cudaMalloc d_out failed: %s\n", cudaGetErrorString(cerr));
        cudaFree(d_in);
        cudaFree(d_tmp);
        return;
    }

    // single upload of the input image
    cerr = cudaMemcpy(d_in, imgin->data, imgSize, cudaMemcpyHostToDevice);
    if (cerr != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy H2D failed: %s\n", cudaGetErrorString(cerr));
        cudaFree(d_in); cudaFree(d_tmp); cudaFree(d_out);
        return;
    }

    // Decide whether horiz/vert are derivative kernels
    int horiz_deriv = (horiz_kernel.width == gaussderiv_kernel.width) ? 1 : 0;
    int vert_deriv  = (vert_kernel.width  == gaussderiv_kernel.width) ? 1 : 0;

    // ---------- HORIZONTAL pass ----------
    {
        // upload correct kernel to constant memory
        int kSize = horiz_kernel.width * sizeof(float);
        if (horiz_deriv)
            cudaMemcpyToSymbol(d_kernel_deriv, horiz_kernel.data, kSize);
        else
            cudaMemcpyToSymbol(d_kernel, horiz_kernel.data, kSize);

        // launch params (match your horiz wrapper)
        dim3 block(32, 8);
        dim3 grid((ncols + block.x - 1) / block.x,
                  (nrows + block.y - 1) / block.y);

        int R = horiz_kernel.width / 2;
        int TILE_W = block.x + 2 * R;
        int TILE_H = block.y;
        size_t sharedBytes = (size_t)TILE_W * (size_t)TILE_H * sizeof(float);

        // d_in -> d_tmp
        convolveHorizShared<<<grid, block, sharedBytes>>>(
            d_in, d_tmp,
            ncols, nrows,
            horiz_kernel.width,
            horiz_deriv
        );

        cerr = cudaGetLastError();
        if (cerr != cudaSuccess) {
            fprintf(stderr, "convolveHorizShared launch failed: %s\n", cudaGetErrorString(cerr));
            cudaFree(d_in); cudaFree(d_tmp); cudaFree(d_out);
            return;
        }
        // no cudaDeviceSynchronize() here — the next kernel will wait implicitly in the same stream
    }

    // ---------- VERTICAL pass ----------
    {
        // upload correct kernel to constant memory
        int kSize = vert_kernel.width * sizeof(float);
        if (vert_deriv)
            cudaMemcpyToSymbol(d_kernel_deriv, vert_kernel.data, kSize);
        else
            cudaMemcpyToSymbol(d_kernel, vert_kernel.data, kSize);

        // launch params (match your vert wrapper)
        dim3 blockV(16, 16);
        dim3 gridV((ncols + blockV.x - 1) / blockV.x,
                   (nrows + blockV.y - 1) / blockV.y);

        int Rv = vert_kernel.width / 2;
        int TILE_Wv = blockV.x;
        int TILE_Hv = blockV.y + 2 * Rv;
        size_t sharedBytesV = (size_t)TILE_Wv * (size_t)TILE_Hv * sizeof(float);

        // d_tmp -> d_out
        convolveVertShared<<<gridV, blockV, sharedBytesV>>>(
            d_tmp, d_out,
            ncols, nrows,
            vert_kernel.width,
            vert_deriv
        );

        cerr = cudaGetLastError();
        if (cerr != cudaSuccess) {
            fprintf(stderr, "convolveVertShared launch failed: %s\n", cudaGetErrorString(cerr));
            cudaFree(d_in); cudaFree(d_tmp); cudaFree(d_out);
            return;
        }
    }

    // Download result once
    cerr = cudaMemcpy(imgout->data, d_out, imgSize, cudaMemcpyDeviceToHost);
    if (cerr != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy D2H failed: %s\n", cudaGetErrorString(cerr));
    }

    // free device buffers
    cudaFree(d_in);
    cudaFree(d_tmp);
    cudaFree(d_out);
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
