/*********************************************************************
 * convolveGPU.h
 *********************************************************************/

#ifndef _CONVOLVEGPU_H_
#define _CONVOLVEGPU_H_

#include "klt.h"
#include "klt_util.h"

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

#ifdef __cplusplus
}
#endif

#endif
