/*********************************************************************
 * klt_utilGPU.h
 *********************************************************************/

#ifndef _KLT_UTILGPU_H_
#define _KLT_UTILGPU_H_

#ifdef __cplusplus
extern "C"
{
#endif

  typedef struct
  {
    int ncols;
    int nrows;
    float *data;          // CPU data pointer
    float *gpu_data;      // GPU data pointer
    int is_gpu_allocated; // if gpu memory was allocated
  } _KLT_FloatImageRec, *_KLT_FloatImage;

  extern _KLT_FloatImage _KLTCreateFloatImage(
      int ncols,
      int nrows);

  extern _KLT_FloatImage _KLTCreateFloatImageGPU(
      int ncols,
      int nrows);

  extern void _KLTFreeFloatImage(
      _KLT_FloatImage);

  extern void _KLTFreeFloatImageGPU(
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
      char *filename, float scale);

  /* GPU memory management */
  extern void _KLTCopyFloatImageToGPU(
      _KLT_FloatImage floatimg);

  extern void _KLTCopyFloatImageToCPU(
      _KLT_FloatImage floatimg);

  extern _KLT_FloatImage _KLTCreateFloatImageGPU(int ncols, int nrows);
  extern void _KLTFreeFloatImageGPU(_KLT_FloatImage img);
  extern void _KLTSubsampleImageGPU(_KLT_FloatImage src, _KLT_FloatImage dst,
                                    int src_ncols, int dst_ncols, int dst_nrows,
                                    int subsampling, int subhalf);

#ifdef __cplusplus
}
#endif

#endif
