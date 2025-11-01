/*********************************************************************
 * pyramid.h
 *********************************************************************/

#ifndef _PYRAMID_H_
#define _PYRAMID_H_

#ifdef __cplusplus
extern "C" {
#endif

#include "klt_utilGPU.h"

typedef struct  {
  int subsampling;
  int nLevels;
  _KLT_FloatImage *img;
  int *ncols, *nrows;
}  _KLT_PyramidRec, *_KLT_Pyramid;


extern _KLT_Pyramid _KLTCreatePyramid(
  int ncols,
  int nrows,
  int subsampling,
  int nlevels);

extern void _KLTComputePyramid(
  _KLT_FloatImage floatimg, 
  _KLT_Pyramid pyramid,
  float sigma_fact);

extern void _KLTFreePyramid(
  _KLT_Pyramid pyramid);

extern _KLT_Pyramid _KLTCreatePyramidGPU(
  int ncols,
  int nrows,
  int subsampling,
  int nlevels);
extern void _KLTComputePyramidGPU(
  _KLT_FloatImage floatimg, 
  _KLT_Pyramid pyramid,
  float sigma_fact);

extern void _KLTFreePyramidGPU(
  _KLT_Pyramid pyramid);

#ifdef __cplusplus
}
#endif

#endif
