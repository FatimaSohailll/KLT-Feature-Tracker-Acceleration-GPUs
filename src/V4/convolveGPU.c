/* Standard includes */
#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>

/* Our includes */
#include "base.h"
#include "error.h"
#include "convolve.h"
#include "klt_util.h"

#define MAX_KERNEL_WIDTH 71

typedef struct
{
  int width;
  float data[MAX_KERNEL_WIDTH];
} ConvolutionKernel;

/* Kernels (cached) */
static ConvolutionKernel gauss_kernel;
static ConvolutionKernel gaussderiv_kernel;
static float sigma_last = -10.0f;

/*********************************************************************

 * _KLTToFloatImage
 */
void _KLTToFloatImage(
    KLT_PixelType *img,
    int ncols, int nrows,
    _KLT_FloatImage floatimg)
{
  KLT_PixelType *ptrend = img + ncols * nrows;
  float *ptrout = floatimg->data;

  assert(floatimg->ncols >= ncols);
  assert(floatimg->nrows >= nrows);

  floatimg->ncols = ncols;
  floatimg->nrows = nrows;

  while (img < ptrend)
    *ptrout++ = (float)*img++;
}

/*********************************************************************
 * _computeKernels
 */
static void _computeKernels(
    float sigma,
    ConvolutionKernel *gauss,
    ConvolutionKernel *gaussderiv)
{
  const float factor = 0.01f;
  int i;

  assert(MAX_KERNEL_WIDTH % 2 == 1);
  assert(sigma >= 0.0f);

  {
    const int hw = MAX_KERNEL_WIDTH / 2;
    float max_gauss = 1.0f, max_gaussderiv = (float)(sigma * exp(-0.5f));

    for (i = -hw; i <= hw; i++)
    {
      gauss->data[i + hw] = (float)exp(-i * i / (2 * sigma * sigma));
      gaussderiv->data[i + hw] = -i * gauss->data[i + hw];
    }

    gauss->width = MAX_KERNEL_WIDTH;
    for (i = -hw; fabs(gauss->data[i + hw] / max_gauss) < factor;
         i++, gauss->width -= 2)
      ;
    gaussderiv->width = MAX_KERNEL_WIDTH;
    for (i = -hw; fabs(gaussderiv->data[i + hw] / max_gaussderiv) < factor;
         i++, gaussderiv->width -= 2)
      ;

    if (gauss->width == MAX_KERNEL_WIDTH || gaussderiv->width == MAX_KERNEL_WIDTH)
      KLTError("(_computeKernels) MAX_KERNEL_WIDTH %d is too small for sigma of %f",
               MAX_KERNEL_WIDTH, sigma);
  }

  for (i = 0; i < gauss->width; i++)
    gauss->data[i] = gauss->data[i + (MAX_KERNEL_WIDTH - gauss->width) / 2];
  for (i = 0; i < gaussderiv->width; i++)
    gaussderiv->data[i] = gaussderiv->data[i + (MAX_KERNEL_WIDTH - gaussderiv->width) / 2];

  {
    const int hw = gaussderiv->width / 2;
    float den;

    den = 0.0f;
    for (i = 0; i < gauss->width; i++)
      den += gauss->data[i];
    for (i = 0; i < gauss->width; i++)
      gauss->data[i] /= den;

    den = 0.0f;
    for (i = -hw; i <= hw; i++)
      den -= i * gaussderiv->data[i + hw];
    for (i = -hw; i <= hw; i++)
      gaussderiv->data[i + hw] /= den;
  }

  sigma_last = sigma;
}

/*********************************************************************
 * _KLTGetKernelWidths
 */
void _KLTGetKernelWidths(
    float sigma,
    int *gauss_width,
    int *gaussderiv_width)
{
  _computeKernels(sigma, &gauss_kernel, &gaussderiv_kernel);
  *gauss_width = gauss_kernel.width;
  *gaussderiv_width = gaussderiv_kernel.width;
}

static void convolve_horiz(
    float *in_data,
    int ncols, int nrows,
    float *kernel_data,
    int kw,
    float *out_data)
{
  int radius = kw / 2;
  int total = ncols * nrows;

  assert(kw % 2 == 1);
#pragma acc parallel loop present(                                \
    in_data[0 : total], kernel_data[0 : kw], out_data[0 : total]) \
    collapse(2)
  for (int j = 0; j < nrows; j++)
  {
    for (int i = 0; i < ncols; i++)
    {
      if (i < radius || i >= ncols - radius)
      {
        out_data[j * ncols + i] = 0.0f;
        continue;
      }

      float sum = 0.0f;
      for (int k = 0; k < kw; k++)
      {
        int src_idx = j * ncols + (i + k - radius);
        sum += in_data[src_idx] * kernel_data[k];
      }
      out_data[j * ncols + i] = sum;
    }
  }
}

static void convolve_vert(
    float *in_data,
    int ncols, int nrows,
    float *kernel_data,
    int kw,
    float *out_data)
{
  int radius = kw / 2;
  int total = ncols * nrows;

  assert(kw % 2 == 1);
#pragma acc parallel loop present(                                \
    in_data[0 : total], kernel_data[0 : kw], out_data[0 : total]) \
    collapse(2)
  for (int j = 0; j < nrows; j++)
  {
    for (int i = 0; i < ncols; i++)
    {
      if (j < radius || j >= nrows - radius)
      {
        out_data[j * ncols + i] = 0.0f;
        continue;
      }

      float sum = 0.0f;
      for (int k = 0; k < kw; k++)
      {
        int src_idx = (j + k - radius) * ncols + i;
        sum += in_data[src_idx] * kernel_data[k];
      }
      out_data[j * ncols + i] = sum;
    }
  }
}

/*********************************************************************
 * _convolveSeparate
 *********************************************************************/
static void _convolveSeparate(
    _KLT_FloatImage imgin,
    ConvolutionKernel horiz_kernel,
    ConvolutionKernel vert_kernel,
    _KLT_FloatImage imgout)
{
  /* Create temporary image */
  _KLT_FloatImage tmpimg;
  tmpimg = _KLTCreateFloatImage(imgin->ncols, imgin->nrows);

  int total = imgin->ncols * imgin->nrows;
  int kw_h = horiz_kernel.width;
  int kw_v = vert_kernel.width;
/* Horizontal then vertical. These functions manage their own OpenACC
   data regions and accept raw pointers so that mapping is correct. */
#pragma acc data copyin(imgin->data[0 : total],      \
                        horiz_kernel.data[0 : kw_h], \
                        vert_kernel.data[0 : kw_v])  \
    create(tmpimg -> data[0 : total])                \
        copyout(imgout->data[0 : total])
  {
    convolve_horiz(imgin->data, imgin->ncols, imgin->nrows,
                   horiz_kernel.data, horiz_kernel.width,
                   tmpimg->data);

    convolve_vert(tmpimg->data, imgin->ncols, imgin->nrows,
                  vert_kernel.data, vert_kernel.width,
                  imgout->data);
  }

  _KLTFreeFloatImage(tmpimg);
}

/*********************************************************************
 * _KLTComputeGradients
 */
void _KLTComputeGradients(
    _KLT_FloatImage img,
    float sigma,
    _KLT_FloatImage gradx,
    _KLT_FloatImage grady)
{
  assert(gradx->ncols >= img->ncols);
  assert(gradx->nrows >= img->nrows);
  assert(grady->ncols >= img->ncols);
  assert(grady->nrows >= img->nrows);

  if (fabsf(sigma - sigma_last) > 0.05f)
    _computeKernels(sigma, &gauss_kernel, &gaussderiv_kernel);

  _convolveSeparate(img, gaussderiv_kernel, gauss_kernel, gradx);
  _convolveSeparate(img, gauss_kernel, gaussderiv_kernel, grady);
}

/*********************************************************************
 * _KLTComputeSmoothedImage
 */
void _KLTComputeSmoothedImage(
    _KLT_FloatImage img,
    float sigma,
    _KLT_FloatImage smooth)
{
  assert(smooth->ncols >= img->ncols);
  assert(smooth->nrows >= img->nrows);

  if (fabsf(sigma - sigma_last) > 0.05f)
    _computeKernels(sigma, &gauss_kernel, &gaussderiv_kernel);

  _convolveSeparate(img, gauss_kernel, gauss_kernel, smooth);
}
