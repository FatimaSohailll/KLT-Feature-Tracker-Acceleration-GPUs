/*********************************************************************
 * convolveGPU.h
 *********************************************************************/

#ifndef _CONVOLVEGPU_H_
#define _CONVOLVEGPU_H_

#include "kltGPU.h"
#include "klt_utilGPU.h"

#ifdef __cplusplus
extern "C" {
#endif


// Struct for kernel (same as CPU version)
typedef struct {
  int width;
  float data[71];
} ConvolutionKernel;

// GPU wrappers
extern void convolveImageHorizGPU(
  _KLT_FloatImage imgin,
  ConvolutionKernel kernel,
  _KLT_FloatImage imgout);

extern void convolveImageVertGPU(
  _KLT_FloatImage imgin,
  ConvolutionKernel kernel,
  _KLT_FloatImage imgout);

extern void convolveSeparateGPU(
  _KLT_FloatImage imgin,
  ConvolutionKernel horiz_kernel,
  ConvolutionKernel vert_kernel,
  _KLT_FloatImage imgout);

extern void _KLTToFloatImage(
  KLT_PixelType *img,
  int ncols, int nrows,
  _KLT_FloatImage floatimg);

extern void _KLTToFloatImageGPU(
    KLT_PixelType *img,
    int ncols, int nrows,
    _KLT_FloatImage floatimg);
    
extern void _KLTComputeGradients(
  _KLT_FloatImage img,
  float sigma,
  _KLT_FloatImage gradx,
  _KLT_FloatImage grady);

extern void _KLTGetKernelWidths(
  float sigma,
  int *gauss_width,
  int *gaussderiv_width);

extern void _KLTComputeSmoothedImage(
  _KLT_FloatImage img,
  float sigma,
  _KLT_FloatImage smooth);
extern void _KLTComputeSmoothedImageGPU(
  _KLT_FloatImage img,
  float sigma,
  _KLT_FloatImage smooth);

extern void convolveSeparateGPUDevicePyramids(
    float* d_imgin,
    float* d_imgout,
    int ncols, int nrows,
    ConvolutionKernel horiz_kernel,
    ConvolutionKernel vert_kernel,
    int horiz_deriv,
    int vert_deriv);

extern void _KLTComputeGradientsGPU(
  _KLT_FloatImage img,
  float sigma,
  _KLT_FloatImage gradx,
  _KLT_FloatImage grady);

#ifdef __cplusplus
}
#endif

#endif
