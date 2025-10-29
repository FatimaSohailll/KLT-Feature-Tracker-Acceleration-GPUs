/*********************************************************************
 * convolveGPU.h
 *********************************************************************/

#ifndef _CONVOLVEGPU_H_
#define _CONVOLVEGPU_H_

#include "klt.h"
#include "klt_util.h"

// Struct for kernel (same as CPU version)
typedef struct {
  int width;
  float data[71];
} ConvolutionKernel;

// GPU wrappers
void convolveImageHorizGPU(
  _KLT_FloatImage imgin,
  ConvolutionKernel kernel,
  _KLT_FloatImage imgout);

void convolveImageVertGPU(
  _KLT_FloatImage imgin,
  ConvolutionKernel kernel,
  _KLT_FloatImage imgout);

void convolveSeparateGPU(
  _KLT_FloatImage imgin,
  ConvolutionKernel horiz_kernel,
  ConvolutionKernel vert_kernel,
  _KLT_FloatImage imgout);

#endif
