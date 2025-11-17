/*********************************************************************
 * pyramid.c
 * OpenACC-accelerated pyramid functions
 *********************************************************************/

/* Standard includes */
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <openacc.h>

/* Our includes */
#include "base.h"
#include "error.h"
#include "convolve.h"
#include "pyramid.h"

/*********************************************************************
 * _KLTCreatePyramid
 *********************************************************************/
_KLT_Pyramid _KLTCreatePyramid(
  int ncols,
  int nrows,
  int subsampling,
  int nlevels)
{
  _KLT_Pyramid pyramid;
  int nbytes = sizeof(_KLT_PyramidRec) +	
    nlevels * sizeof(_KLT_FloatImage *) +
    nlevels * sizeof(int) +
    nlevels * sizeof(int);
  int i;

  if (subsampling != 2 && subsampling != 4 && 
      subsampling != 8 && subsampling != 16 && subsampling != 32)
    KLTError("(_KLTCreatePyramid)  Pyramid's subsampling must "
             "be either 2, 4, 8, 16, or 32");

  pyramid = (_KLT_Pyramid)malloc(nbytes);
  if (pyramid == NULL)
    KLTError("(_KLTCreatePyramid)  Out of memory");

  pyramid->subsampling = subsampling;
  pyramid->nLevels = nlevels;
  pyramid->img = (_KLT_FloatImage *)(pyramid + 1);
  pyramid->ncols = (int *)(pyramid->img + nlevels);
  pyramid->nrows = (int *)(pyramid->ncols + nlevels);

  for (i = 0 ; i < nlevels ; i++)  {
    pyramid->img[i] = _KLTCreateFloatImage(ncols, nrows);
    pyramid->ncols[i] = ncols;
    pyramid->nrows[i] = nrows;
    ncols /= subsampling;
    nrows /= subsampling;
  }

  return pyramid;
}

/*********************************************************************
 * _KLTCreatePyramidGPU
 *********************************************************************/
_KLT_Pyramid _KLTCreatePyramidGPU(
  int ncols,
  int nrows,
  int subsampling,
  int nlevels)
{
  _KLT_Pyramid pyramid;
  int nbytes = sizeof(_KLT_PyramidRec) +	
    nlevels * sizeof(_KLT_FloatImage *) +
    nlevels * sizeof(int) +
    nlevels * sizeof(int);
  int i;

  if (subsampling != 2 && subsampling != 4 && 
      subsampling != 8 && subsampling != 16 && subsampling != 32)
    KLTError("(_KLTCreatePyramidGPU)  Pyramid's subsampling must "
             "be either 2, 4, 8, 16, or 32");

  pyramid = (_KLT_Pyramid)malloc(nbytes);
  if (pyramid == NULL)
    KLTError("(_KLTCreatePyramidGPU)  Out of memory");

  pyramid->subsampling = subsampling;
  pyramid->nLevels = nlevels;
  pyramid->img = (_KLT_FloatImage *)(pyramid + 1);
  pyramid->ncols = (int *)(pyramid->img + nlevels);
  pyramid->nrows = (int *)(pyramid->ncols + nlevels);

  for (i = 0 ; i < nlevels ; i++)  {
    pyramid->img[i] = _KLTCreateFloatImage(ncols, nrows);
    pyramid->ncols[i] = ncols;
    pyramid->nrows[i] = nrows;
    
    /* Allocate device memory for this level */
    int level_size = ncols * nrows;
    #pragma acc enter data create(pyramid->img[i]->data[0:level_size])
    
    ncols /= subsampling;
    nrows /= subsampling;
  }

  return pyramid;
}

/*********************************************************************
 * _KLTFreePyramid
 *********************************************************************/
void _KLTFreePyramid(
  _KLT_Pyramid pyramid)
{
  int i;
  for (i = 0 ; i < pyramid->nLevels ; i++)
    _KLTFreeFloatImage(pyramid->img[i]);
  free(pyramid);
}

/*********************************************************************
 * _KLTFreePyramidGPU
 *********************************************************************/
void _KLTFreePyramidGPU(
  _KLT_Pyramid pyramid)
{
  int i;
  for (i = 0 ; i < pyramid->nLevels ; i++) {
    int level_size = pyramid->ncols[i] * pyramid->nrows[i];
    #pragma acc exit data delete(pyramid->img[i]->data[0:level_size])
    _KLTFreeFloatImage(pyramid->img[i]);
  }
  free(pyramid);
}

static void _KLTSubsampleImage(
  _KLT_FloatImage imgin,
  _KLT_FloatImage imgout,
  int oldncols,
  int newncols,
  int newnrows,
  int subsampling,
  int subhalf)
{
  int input_size = oldncols * imgin->nrows;  // Use actual row count
  int output_size = newncols * newnrows;
  
  #pragma acc data present(imgin->data[0:input_size], imgout->data[0:output_size])
  {
    #pragma acc parallel loop collapse(2)
    for (int y = 0 ; y < newnrows ; y++) {
      for (int x = 0 ; x < newncols ; x++) {
        int src_y = subsampling * y + subhalf;
        int src_x = subsampling * x + subhalf;
        // Add bounds checking
        if (src_x < oldncols && src_y < imgin->nrows) {
          imgout->data[y*newncols + x] = imgin->data[src_y * oldncols + src_x];
        } else {
          imgout->data[y*newncols + x] = 0.0f;
        }
      }
    }
  }
}

/*********************************************************************
 * _KLTComputePyramidGPU (CORRECTED OpenACC version)
 *********************************************************************/
void _KLTComputePyramidGPU(
  _KLT_FloatImage img, 
  _KLT_Pyramid pyramid,
  float sigma_fact)
{
  int ncols = img->ncols, nrows = img->nrows;
  int subsampling = pyramid->subsampling;
  int subhalf = subsampling / 2;
  float sigma = subsampling * sigma_fact;
  int i;

  if (subsampling != 2 && subsampling != 4 && 
      subsampling != 8 && subsampling != 16 && subsampling != 32)
    KLTError("(_KLTComputePyramidGPU)  Pyramid's subsampling must "
             "be either 2, 4, 8, 16, or 32");

  assert(pyramid->ncols[0] == img->ncols);
  assert(pyramid->nrows[0] == img->nrows);

  /* Copy input image to device and initialize level 0 */
  int size_level0 = ncols * nrows;
  #pragma acc enter data copyin(img->data[0:size_level0])
  #pragma acc enter data copyin(pyramid->img[0]->data[0:size_level0])
  
  #pragma acc parallel loop present(img->data, pyramid->img[0]->data)
  for (int i = 0; i < size_level0; i++) {
    pyramid->img[0]->data[i] = img->data[i];
  }

  /* Process pyramid levels */
  for (i = 1 ; i < pyramid->nLevels ; i++)  {
    _KLT_FloatImage tmpimg = _KLTCreateFloatImage(ncols, nrows);
    int tmp_size = ncols * nrows;
    int new_size = (ncols / subsampling) * (nrows / subsampling);
    
    /* Allocate temporary image on device */
    #pragma acc enter data create(tmpimg->data[0:tmp_size])
    
    /* Compute smoothed image - convolution functions should handle device data */
    _KLTComputeSmoothedImage(pyramid->img[i-1], sigma, tmpimg);
    
    /* Update dimensions for next level */
    int oldncols = ncols;
    ncols /= subsampling;
    nrows /= subsampling;
    
    /* Ensure output level is allocated on device */
    #pragma acc enter data create(pyramid->img[i]->data[0:new_size])
    
    /* Subsample */
    _KLTSubsampleImage(tmpimg, pyramid->img[i], 
                      oldncols, ncols, nrows, 
                      subsampling, subhalf);

    /* Cleanup temporary image */
    #pragma acc exit data delete(tmpimg->data[0:tmp_size])
    _KLTFreeFloatImage(tmpimg);
  }
  
  /* Cleanup input image */
  #pragma acc exit data delete(img->data[0:size_level0])
}
