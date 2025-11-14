/*********************************************************************
 * convolve.c
 *********************************************************************/

/* Standard includes */
#include <assert.h>
#include <math.h>
#include <stdlib.h>   /* malloc(), realloc() */

/* Our includes */
#include "base.h"
#include "error.h"
#include "convolve.h"
#include "klt_util.h"   /* printing */

#define MAX_KERNEL_WIDTH 71

typedef struct {
    int width;
    float data[MAX_KERNEL_WIDTH];
} ConvolutionKernel;

/* Kernels */
static ConvolutionKernel gauss_kernel;
static ConvolutionKernel gaussderiv_kernel;
static float sigma_last = -10.0;


/*********************************************************************
 * _KLTToFloatImage
 *********************************************************************/
void _KLTToFloatImage(
    KLT_PixelType *img,
    int ncols, int nrows,
    _KLT_FloatImage floatimg)
{
    KLT_PixelType *ptrend = img + ncols*nrows;
    float *ptrout = floatimg->data;

    assert(floatimg->ncols >= ncols);
    assert(floatimg->nrows >= nrows);

    floatimg->ncols = ncols;
    floatimg->nrows = nrows;

    while (img < ptrend)
        *ptrout++ = (float) *img++;
}


/*********************************************************************
 * _computeKernels
 *********************************************************************/
static void _computeKernels(
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
        float max_gauss = 1.0f, max_gaussderiv = (float)(sigma*exp(-0.5f));

        for (i = -hw; i <= hw; i++) {
            gauss->data[i+hw] = (float) exp(-i*i / (2*sigma*sigma));
            gaussderiv->data[i+hw] = -i * gauss->data[i+hw];
        }

        gauss->width = MAX_KERNEL_WIDTH;
        for (i = -hw; fabs(gauss->data[i+hw] / max_gauss) < factor; i++, gauss->width -= 2);
        gaussderiv->width = MAX_KERNEL_WIDTH;
        for (i = -hw; fabs(gaussderiv->data[i+hw] / max_gaussderiv) < factor; i++, gaussderiv->width -= 2);

        if (gauss->width == MAX_KERNEL_WIDTH || gaussderiv->width == MAX_KERNEL_WIDTH)
            KLTError("(_computeKernels) MAX_KERNEL_WIDTH %d is too small for sigma %f", MAX_KERNEL_WIDTH, sigma);
    }

    /* Shift and normalize */
    for (i = 0; i < gauss->width; i++)
        gauss->data[i] = gauss->data[i + (MAX_KERNEL_WIDTH - gauss->width)/2];
    for (i = 0; i < gaussderiv->width; i++)
        gaussderiv->data[i] = gaussderiv->data[i + (MAX_KERNEL_WIDTH - gaussderiv->width)/2];

    {
        int hw = gaussderiv->width / 2;
        float den = 0.0f;
        for (i = 0; i < gauss->width; i++) den += gauss->data[i];
        for (i = 0; i < gauss->width; i++) gauss->data[i] /= den;
        den = 0.0f;
        for (i = -hw; i <= hw; i++) den -= i * gaussderiv->data[i+hw];
        for (i = -hw; i <= hw; i++) gaussderiv->data[i+hw] /= den;
    }

    sigma_last = sigma;
}


/*********************************************************************
 * _KLTGetKernelWidths
 *********************************************************************/
void _KLTGetKernelWidths(
    float sigma,
    int *gauss_width,
    int *gaussderiv_width)
{
    _computeKernels(sigma, &gauss_kernel, &gaussderiv_kernel);
    *gauss_width = gauss_kernel.width;
    *gaussderiv_width = gaussderiv_kernel.width;
}


/*********************************************************************
 * _convolveImageHoriz (OpenACC)
 *********************************************************************/
static void _convolveImageHoriz(
    _KLT_FloatImage imgin,
    ConvolutionKernel kernel,
    _KLT_FloatImage imgout)
{
    int ncols = imgin->ncols;
    int nrows = imgin->nrows;
    int radius = kernel.width / 2;
    int kw = kernel.width;
    int total_pixels = ncols * nrows;

    float *in = imgin->data;
    float *out = imgout->data;
    float *k = kernel.data;

    assert(kernel.width % 2 == 1);
    assert(imgin != imgout);
    assert(imgout->ncols >= imgin->ncols);
    assert(imgout->nrows >= imgin->nrows);

    #pragma acc parallel loop collapse(2) copyin(kernel.data[0:kw]) present(in[0:total_pixels], out[0:total_pixels])
    for (int j = 0; j < nrows; j++) {
        for (int i = 0; i < ncols; i++) {
            if (i < radius || i >= ncols - radius) {
                out[j*ncols + i] = 0.0f;
                continue;
            }
            float sum = 0.0f;
            for (int kk = 0; kk < kw; kk++) {
                sum += in[j*ncols + (i + kk - radius)] * k[kk];
            }
            out[j*ncols + i] = sum;
        }
    }
}


/*********************************************************************
 * _convolveImageVert (OpenACC)
 *********************************************************************/
static void _convolveImageVert(
    _KLT_FloatImage imgin,
    ConvolutionKernel kernel,
    _KLT_FloatImage imgout)
{
    int ncols = imgin->ncols;
    int nrows = imgin->nrows;
    int radius = kernel.width / 2;
    int kw = kernel.width;
    int total_pixels = ncols * nrows;

    float *in = imgin->data;
    float *out = imgout->data;
    float *k = kernel.data;

    assert(kernel.width % 2 == 1);
    assert(imgin != imgout);
    assert(imgout->ncols >= imgin->ncols);
    assert(imgout->nrows >= imgin->nrows);

    #pragma acc parallel loop collapse(2) copyin(kernel.data[0:kw]) present(in[0:total_pixels], out[0:total_pixels])
    for (int j = 0; j < nrows; j++) {
        for (int i = 0; i < ncols; i++) {
            if (j < radius || j >= nrows - radius) {
                out[j*ncols + i] = 0.0f;
                continue;
            }
            float sum = 0.0f;
            for (int kk = 0; kk < kw; kk++) {
                sum += in[(j + kk - radius)*ncols + i] * k[kk];
            }
            out[j*ncols + i] = sum;
        }
    }
}


/*********************************************************************
 * _convolveSeparate (OpenACC)
 *********************************************************************/
static void _convolveSeparate(
    _KLT_FloatImage imgin,
    ConvolutionKernel horiz_kernel,
    ConvolutionKernel vert_kernel,
    _KLT_FloatImage imgout)
{
    int n = imgin->ncols * imgin->nrows;
    int kw_h = horiz_kernel.width;
    int kw_v = vert_kernel.width;

    _KLT_FloatImage tmpimg = _KLTCreateFloatImage(imgin->ncols, imgin->nrows);

    #pragma acc data \
        copyin(imgin->data[0:n]) \
        create(tmpimg->data[0:n]) \
        copyout(imgout->data[0:n])
    {
        _convolveImageHoriz(imgin, horiz_kernel, tmpimg);
        _convolveImageVert(tmpimg, vert_kernel, imgout);
    }

    _KLTFreeFloatImage(tmpimg);
}


/*********************************************************************
 * _KLTComputeGradients
 *********************************************************************/
void _KLTComputeGradients(
    _KLT_FloatImage img,
    float sigma,
    _KLT_FloatImage gradx,
    _KLT_FloatImage grady)
{
    assert(gradx->ncols >= img->ncols && gradx->nrows >= img->nrows);
    assert(grady->ncols >= img->ncols && grady->nrows >= img->nrows);

    if (fabs(sigma - sigma_last) > 0.05)
        _computeKernels(sigma, &gauss_kernel, &gaussderiv_kernel);

    _convolveSeparate(img, gaussderiv_kernel, gauss_kernel, gradx);
    _convolveSeparate(img, gauss_kernel, gaussderiv_kernel, grady);
}


/*********************************************************************
 * _KLTComputeSmoothedImage
 *********************************************************************/
void _KLTComputeSmoothedImage(
    _KLT_FloatImage img,
    float sigma,
    _KLT_FloatImage smooth)
{
    assert(smooth->ncols >= img->ncols && smooth->nrows >= img->nrows);

    if (fabs(sigma - sigma_last) > 0.05)
        _computeKernels(sigma, &gauss_kernel, &gaussderiv_kernel);

    _convolveSeparate(img, gauss_kernel, gauss_kernel, smooth);
}

