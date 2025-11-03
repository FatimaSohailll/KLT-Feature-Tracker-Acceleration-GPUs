/*********************************************************************
 * klt_util.c
 *********************************************************************/

/* Standard includes */
#include <assert.h>
#include <stdlib.h>  /* malloc() */
#include <math.h>		/* fabs() */

/* Our includes */
#include "base.h"
#include "error.h"
#include "pnmio.h"
#include "kltGPU.h"
#include "klt_utilGPU.h"


/*********************************************************************/

float _KLTComputeSmoothSigma(
  KLT_TrackingContext tc)
{
  return (tc->smooth_sigma_fact * max(tc->window_width, tc->window_height));
}


/*********************************************************************
 * _KLTCreateFloatImage
 */

_KLT_FloatImage _KLTCreateFloatImage(
  int ncols,
  int nrows)
{
  _KLT_FloatImage floatimg;
  int nbytes = sizeof(_KLT_FloatImageRec) +
    ncols * nrows * sizeof(float);

  floatimg = (_KLT_FloatImage)  malloc(nbytes);
  if (floatimg == NULL)
    KLTError("(_KLTCreateFloatImage)  Out of memory");
  floatimg->ncols = ncols;
  floatimg->nrows = nrows;
  floatimg->data = (float *)  (floatimg + 1);

  return(floatimg);
}

/*********************************************************************
 * _KLTFreeFloatImage
 */

void _KLTFreeFloatImage(
  _KLT_FloatImage floatimg)
{
  free(floatimg);
}


/*********************************************************************
 * _KLTPrintSubFloatImage
 */

void _KLTPrintSubFloatImage(
  _KLT_FloatImage floatimg,
  int x0, int y0,
  int width, int height)
{
  int ncols = floatimg->ncols;
  int offset;
  int i, j;

  assert(x0 >= 0);
  assert(y0 >= 0);
  assert(x0 + width <= ncols);
  assert(y0 + height <= floatimg->nrows);

  fprintf(stderr, "\n");
  for (j = 0 ; j < height ; j++)  {
    for (i = 0 ; i < width ; i++)  {
      offset = (j+y0)*ncols + (i+x0);
      fprintf(stderr, "%6.2f ", *(floatimg->data + offset));
    }
    fprintf(stderr, "\n");
  }
  fprintf(stderr, "\n");
}
	

/*********************************************************************
 * _KLTWriteFloatImageToPGM
 */

void _KLTWriteFloatImageToPGM(
  _KLT_FloatImage img,
  char *filename)
{
  int npixs = img->ncols * img->nrows;
  float mmax = -999999.9f, mmin = 999999.9f;
  float fact;
  float *ptr;
  uchar *byteimg, *ptrout;
  int i;

  /* Calculate minimum and maximum values of float image */
  ptr = img->data;
  for (i = 0 ; i < npixs ; i++)  {
    mmax = max(mmax, *ptr);
    mmin = min(mmin, *ptr);
    ptr++;
  }
	
  /* Allocate memory to hold converted image */
  byteimg = (uchar *) malloc(npixs * sizeof(uchar));

  /* Convert image from float to uchar */
  fact = 255.0f / (mmax-mmin);
  ptr = img->data;
  ptrout = byteimg;
  for (i = 0 ; i < npixs ; i++)  {
    *ptrout++ = (uchar) ((*ptr++ - mmin) * fact);
  }

  /* Write uchar image to PGM */
  pgmWriteFile(filename, byteimg, img->ncols, img->nrows);

  /* Free memory */
  free(byteimg);
}

/*********************************************************************
 * _KLTWriteFloatImageToPGM
 */

void _KLTWriteAbsFloatImageToPGM(
  _KLT_FloatImage img,
  char *filename,float scale)
{
  int npixs = img->ncols * img->nrows;
  float fact;
  float *ptr;
  uchar *byteimg, *ptrout;
  int i;
  float tmp;
	
  /* Allocate memory to hold converted image */
  byteimg = (uchar *) malloc(npixs * sizeof(uchar));

  /* Convert image from float to uchar */
  fact = 255.0f / scale;
  ptr = img->data;
  ptrout = byteimg;
  for (i = 0 ; i < npixs ; i++)  {
    tmp = (float) (fabs(*ptr++) * fact);
    if(tmp > 255.0) tmp = 255.0;
    *ptrout++ =  (uchar) tmp;
  }

  /* Write uchar image to PGM */
  pgmWriteFile(filename, byteimg, img->ncols, img->nrows);

  /* Free memory */
  free(byteimg);
}


_KLT_FloatImage _KLTCreateFloatImageGPU(int ncols, int nrows)
{
  _KLT_FloatImage floatimg;
  size_t nbytes = sizeof(_KLT_FloatImageRec) + ncols * nrows * sizeof(float);
  
  /* Allocate memory for the structure on host */
  floatimg = (_KLT_FloatImage) malloc(nbytes);
  if (floatimg == NULL)
    KLTError("(_KLTCreateFloatImageGPU)  Out of memory");
  
  /* Set parameters */
  floatimg->ncols = ncols;
  floatimg->nrows = nrows;
  
  /* Allocate GPU memory for image data */
  cudaError_t err = cudaMalloc(&floatimg->data, ncols * nrows * sizeof(float));
  if (err != cudaSuccess) {
    free(floatimg);
    KLTError("(_KLTCreateFloatImageGPU)  CUDA memory allocation failed: %s", 
             cudaGetErrorString(err));
  }
  
  return floatimg;
}

void _KLTFreeFloatImageGPU(_KLT_FloatImage img)
{
  if (img != NULL) {
    if (img->data != NULL) {
      cudaFree(img->data);
    }
    free(img);
  }
}

__global__ void subsampleKernel(
    const float* __restrict__ src,
    float* __restrict__ dst,
    int src_ncols,
    int dst_ncols,
    int dst_nrows,
    int subsampling,
    int subhalf)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < dst_ncols && y < dst_nrows) {
        int src_x = subsampling * x + subhalf;
        int src_y = subsampling * y + subhalf;
        dst[y * dst_ncols + x] = src[src_y * src_ncols + src_x];
    }
}

void _KLTSubsampleImageGPU(
    _KLT_FloatImage src, 
    _KLT_FloatImage dst,
    int src_ncols, 
    int dst_ncols, 
    int dst_nrows,
    int subsampling, 
    int subhalf)
{
    dim3 block(16, 16);
    dim3 grid((dst_ncols + block.x - 1) / block.x,
              (dst_nrows + block.y - 1) / block.y);
    
    subsampleKernel<<<grid, block>>>(
        src->data, dst->data,
        src_ncols, dst_ncols, dst_nrows,
        subsampling, subhalf
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        KLTError("(_KLTSubsampleImageGPU) Kernel launch failed: %s",
                 cudaGetErrorString(err));
    }
}
