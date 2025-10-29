/*********************************************************************
 * klt_util.h
 *********************************************************************/

#ifndef _KLT_UTILGPU_H_
#define _KLT_UTILGPU_H_

#ifdef __cplusplus
extern "C" {
#endif

typedef struct  {
  int ncols;
  int nrows;
  float *data;
}  _KLT_FloatImageRec, *_KLT_FloatImage;

extern _KLT_FloatImage _KLTCreateFloatImage(
  int ncols, 
  int nrows);

extern void _KLTFreeFloatImage(
  _KLT_FloatImage);
	
extern void _KLTPrintSubFloatImage(
  _KLT_FloatImage floatimg,
  int x0, int y0,
  int width, int height);

extern void _KLTWriteFloatImageToPGM(
  _KLT_FloatImage img,
  char *filename);

/* for affine mapping */
extern void _KLTWriteAbsFloatImageToPGM(
  _KLT_FloatImage img,
  char *filename,float scale);

#ifdef __cplusplus
}
#endif

#endif
