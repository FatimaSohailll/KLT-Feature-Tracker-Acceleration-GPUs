/*********************************************************************
 * pyramid.c
 *
 *********************************************************************/

/* Standard includes */
#include <assert.h>
#include <stdlib.h>		/* malloc() ? */
#include <string.h>		/* memset() ? */
#include <math.h>		/* */

/* Our includes */
#include "base.h"
#include "error.h"
#include "convolve.h"	/* for computing pyramid */
#include "pyramid.h"


/*********************************************************************
 *
 */

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

     
  /* Allocate memory for structure and set parameters */
  pyramid = (_KLT_Pyramid)  malloc(nbytes);
  if (pyramid == NULL)
    KLTError("(_KLTCreatePyramid)  Out of memory");
     
  /* Set parameters */
  pyramid->subsampling = subsampling;
  pyramid->nLevels = nlevels;
  pyramid->img = (_KLT_FloatImage *) (pyramid + 1);
  pyramid->ncols = (int *) (pyramid->img + nlevels);
  pyramid->nrows = (int *) (pyramid->ncols + nlevels);

  /* Allocate memory for each level of pyramid and assign pointers */
  for (i = 0 ; i < nlevels ; i++)  {
    pyramid->img[i] =  _KLTCreateFloatImage(ncols, nrows);
    pyramid->ncols[i] = ncols;  pyramid->nrows[i] = nrows;
    ncols /= subsampling;  nrows /= subsampling;
  }

  return pyramid;
}


/*********************************************************************
 *
 */

void _KLTFreePyramid(
  _KLT_Pyramid pyramid)
{
  int i;

  /* Free images */
  for (i = 0 ; i < pyramid->nLevels ; i++) {
    #pragma acc exit data delete(pyramid->img[i]->data[0:pyramid->ncols[i]*pyramid->nrows[i]])
    _KLTFreeFloatImage(pyramid->img[i]);
  }

  /* Free structure */
  free(pyramid);
}


/*********************************************************************
 * _KLTSubsampleImage (OpenACC version)
 * Subsample image using OpenACC
 *********************************************************************/
static void _KLTSubsampleImage(
  _KLT_FloatImage imgin,
  _KLT_FloatImage imgout,
  int oldncols,
  int newncols,
  int newnrows,
  int subsampling,
  int subhalf)
{
  int total_pixels = newncols * newnrows;
  
  #pragma acc data present(imgin->data[0:oldncols*oldncols], imgout->data[0:total_pixels])
  {
    #pragma acc parallel loop collapse(2) gang vector
    for (int y = 0 ; y < newnrows ; y++) {
      for (int x = 0 ; x < newncols ; x++) {
        imgout->data[y*newncols + x] = 
          imgin->data[(subsampling*y + subhalf) * oldncols + (subsampling*x + subhalf)];
      }
    }
  }
}


/*********************************************************************
 *
 */

void _KLTComputePyramid(
  _KLT_FloatImage img, 
  _KLT_Pyramid pyramid,
  float sigma_fact)
{
  _KLT_FloatImage currimg, tmpimg;
  int ncols = img->ncols, nrows = img->nrows;
  int subsampling = pyramid->subsampling;
  int subhalf = subsampling / 2;
  float sigma = subsampling * sigma_fact;  /* empirically determined */
  int oldncols;
  int i, x, y;
	
  if (subsampling != 2 && subsampling != 4 && 
      subsampling != 8 && subsampling != 16 && subsampling != 32)
    KLTError("(_KLTComputePyramid)  Pyramid's subsampling must "
             "be either 2, 4, 8, 16, or 32");

  assert(pyramid->ncols[0] == img->ncols);
  assert(pyramid->nrows[0] == img->nrows);

  /* Copy original image to level 0 of pyramid */
  int size_level0 = ncols * nrows;
  #pragma acc parallel loop present(pyramid->img[0]->data[0:size_level0]) copyin(img->data[0:size_level0])
  for (i = 0; i < size_level0; i++) {
    pyramid->img[0]->data[i] = img->data[i];
  }

  /* Copy level 0 to device */
  #pragma acc enter data copyin(pyramid->img[0]->data[0:size_level0])

  currimg = pyramid->img[0];
  for (i = 1 ; i < pyramid->nLevels ; i++)  {
    tmpimg = _KLTCreateFloatImage(ncols, nrows);
    
    /* Copy temporary image to device */
    int tmp_size = ncols * nrows;
    #pragma acc enter data create(tmpimg->data[0:tmp_size])
    
    /* Compute smoothed image on GPU */
    _KLTComputeSmoothedImage(currimg, sigma, tmpimg);

    /* Subsample on GPU */
    oldncols = ncols;
    ncols /= subsampling;  nrows /= subsampling;
    
    int new_size = ncols * nrows;
    #pragma acc enter data create(pyramid->img[i]->data[0:new_size])
    
    _KLTSubsampleImage(tmpimg, pyramid->img[i], 
                      oldncols, ncols, nrows, 
                      subsampling, subhalf);

    /* Reassign current image */
    currimg = pyramid->img[i];
    
    /* Free temporary image from device and host */
    #pragma acc exit data delete(tmpimg->data[0:tmp_size])
    _KLTFreeFloatImage(tmpimg);
  }
}

/*********************************************************************
 * _KLTComputePyramidGPU (OpenACC version)
 * GPU-accelerated pyramid computation using OpenACC
 *********************************************************************/
void _KLTComputePyramidGPU(
  _KLT_FloatImage img, 
  _KLT_Pyramid pyramid,
  float sigma_fact)
{
  _KLT_FloatImage d_currimg, d_tmpimg;
  int ncols = img->ncols, nrows = img->nrows;
  int subsampling = pyramid->subsampling;
  int subhalf = subsampling / 2;
  float sigma = subsampling * sigma_fact;  /* empirically determined */
  int oldncols;
  int i;
	
  if (subsampling != 2 && subsampling != 4 && 
      subsampling != 8 && subsampling != 16 && subsampling != 32)
    KLTError("(_KLTComputePyramidGPU)  Pyramid's subsampling must "
             "be either 2, 4, 8, 16, or 32");

  assert(pyramid->ncols[0] == img->ncols);
  assert(pyramid->nrows[0] == img->nrows);

  /* Copy original image to level 0 of pyramid on GPU */
  int size_level0 = ncols * nrows;
  #pragma acc enter data copyin(pyramid->img[0]->data[0:size_level0])
  #pragma acc parallel loop present(img->data[0:size_level0], pyramid->img[0]->data[0:size_level0])
  for (i = 0; i < size_level0; i++) {
    pyramid->img[0]->data[i] = img->data[i];
  }

  d_currimg = pyramid->img[0];
  for (i = 1 ; i < pyramid->nLevels ; i++)  {
    /* Create temporary image */
    d_tmpimg = _KLTCreateFloatImage(ncols, nrows);
    int tmp_size = ncols * nrows;
    
    /* Copy temporary image to GPU */
    #pragma acc enter data create(d_tmpimg->data[0:tmp_size])
    
    /* Compute smoothed image directly on GPU */
    _KLTComputeSmoothedImage(d_currimg, sigma, d_tmpimg);

    /* Subsample on GPU */
    oldncols = ncols;
    ncols /= subsampling;  
    nrows /= subsampling;
    
    int new_size = ncols * nrows;
    #pragma acc enter data create(pyramid->img[i]->data[0:new_size])
    
    /* Perform subsampling on GPU */
    _KLTSubsampleImage(d_tmpimg, pyramid->img[i], 
                      oldncols, ncols, nrows, 
                      subsampling, subhalf);

    /* Reassign current image */
    d_currimg = pyramid->img[i];
    
    /* Free temporary GPU image */
    #pragma acc exit data delete(d_tmpimg->data[0:tmp_size])
    _KLTFreeFloatImage(d_tmpimg);
  }
}

/*********************************************************************
 * _KLTCreatePyramidGPU (OpenACC version)
 * Create pyramid with GPU memory allocation
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

     
  /* Allocate memory for structure and set parameters */
  pyramid = (_KLT_Pyramid)  malloc(nbytes);
  if (pyramid == NULL)
    KLTError("(_KLTCreatePyramidGPU)  Out of memory");
     
  /* Set parameters */
  pyramid->subsampling = subsampling;
  pyramid->nLevels = nlevels;
  pyramid->img = (_KLT_FloatImage *) (pyramid + 1);
  pyramid->ncols = (int *) (pyramid->img + nlevels);
  pyramid->nrows = (int *) (pyramid->ncols + nlevels);

  /* Allocate GPU memory for each level of pyramid and assign pointers */
  for (i = 0 ; i < nlevels ; i++)  {
    pyramid->img[i] =  _KLTCreateFloatImage(ncols, nrows);
    pyramid->ncols[i] = ncols;  pyramid->nrows[i] = nrows;
    
    /* Allocate GPU memory for this pyramid level */
    int level_size = ncols * nrows;
    #pragma acc enter data create(pyramid->img[i]->data[0:level_size])
    
    ncols /= subsampling;  nrows /= subsampling;
  }

  return pyramid;
}

/*********************************************************************
 * _KLTFreePyramidGPU (OpenACC version)
 * Free pyramid with GPU memory deallocation
 *********************************************************************/
void _KLTFreePyramidGPU(
  _KLT_Pyramid pyramid)
{
  int i;

  /* Free GPU images */
  for (i = 0 ; i < pyramid->nLevels ; i++) {
    int level_size = pyramid->ncols[i] * pyramid->nrows[i];
    #pragma acc exit data delete(pyramid->img[i]->data[0:level_size])
    _KLTFreeFloatImage(pyramid->img[i]);
  }

  /* Free structure */
  free(pyramid);
}
