/*********************************************************************
 * trackFeatures.c
 * OpenACC-accelerated feature tracking - Parallel version
 *********************************************************************/

/* Standard includes */
#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <openacc.h>

/* Our includes */
#include "base.h"
#include "error.h"
#include "convolve.h"
#include "klt.h"
#include "klt_util.h"
#include "pyramid.h"

extern int KLT_verbose;

typedef float *_FloatWindow;

/*********************************************************************
 * _interpolate (OpenACC)
 * GPU-accelerated bilinear interpolation
 *********************************************************************/
#pragma acc routine seq
static float _interpolate(
  float x,
  float y,
  float *img_data,
  int ncols,
  int nrows)
{
  int xt = (int) x;
  int yt = (int) y;
  float ax = x - xt;
  float ay = y - yt;
  
  // Boundary check
  if (xt < 0 || yt < 0 || xt >= ncols - 1 || yt >= nrows - 1)
    return 0.0f;

  float *ptr = img_data + (ncols * yt) + xt;

  return ((1-ax) * (1-ay) * ptr[0] +
          ax    * (1-ay) * ptr[1] +
          (1-ax) * ay    * ptr[ncols] +
          ax    * ay    * ptr[ncols + 1]);
}

/*********************************************************************
 * _computeIntensityDifference
 *********************************************************************/
#pragma acc routine seq
static void _computeIntensityDifference(
  float *img1_data, int img1_ncols, int img1_nrows,
  float *img2_data, int img2_ncols, int img2_nrows,
  float x1, float y1,
  float x2, float y2,
  int width, int height,
  _FloatWindow imgdiff)
{
  int hw = width/2, hh = height/2;
  int total_pixels = width * height;

  for (int j = -hh; j <= hh; j++) {
    for (int i = -hw; i <= hw; i++) {
      int idx = (j+hh) * width + (i+hw);
      float g1 = _interpolate(x1+i, y1+j, img1_data, img1_ncols, img1_nrows);
      float g2 = _interpolate(x2+i, y2+j, img2_data, img2_ncols, img2_nrows);
      imgdiff[idx] = g1 - g2;
    }
  }
}

/*********************************************************************
 * _computeGradientSum
 *********************************************************************/
#pragma acc routine seq
static void _computeGradientSum(
  float *gradx1_data, int gradx1_ncols, int gradx1_nrows,
  float *grady1_data, int grady1_ncols, int grady1_nrows,
  float *gradx2_data, int gradx2_ncols, int gradx2_nrows,
  float *grady2_data, int grady2_ncols, int grady2_nrows,
  float x1, float y1,
  float x2, float y2,
  int width, int height,
  _FloatWindow gradx,
  _FloatWindow grady)
{
  int hw = width/2, hh = height/2;
  int total_pixels = width * height;

  for (int j = -hh; j <= hh; j++) {
    for (int i = -hw; i <= hw; i++) {
      int idx = (j+hh) * width + (i+hw);
      gradx[idx] = _interpolate(x1+i, y1+j, gradx1_data, gradx1_ncols, gradx1_nrows) + 
                   _interpolate(x2+i, y2+j, gradx2_data, gradx2_ncols, gradx2_nrows);
      grady[idx] = _interpolate(x1+i, y1+j, grady1_data, grady1_ncols, grady1_nrows) + 
                   _interpolate(x2+i, y2+j, grady2_data, grady2_ncols, grady2_nrows);
    }
  }
}

/*********************************************************************
 * _compute2by2GradientMatrix
 *********************************************************************/
#pragma acc routine seq
static void _compute2by2GradientMatrix(
  _FloatWindow gradx,
  _FloatWindow grady,
  int width, int height,
  float *gxx, float *gxy, float *gyy)
{
  int total_pixels = width * height;
  float l_gxx = 0.0f, l_gxy = 0.0f, l_gyy = 0.0f;

  for (int i = 0; i < total_pixels; i++) {
    float gx = gradx[i];
    float gy = grady[i];
    l_gxx += gx * gx;
    l_gxy += gx * gy;
    l_gyy += gy * gy;
  }

  *gxx = l_gxx;
  *gxy = l_gxy;
  *gyy = l_gyy;
}

/*********************************************************************
 * _compute2by1ErrorVector
 *********************************************************************/
#pragma acc routine seq
static void _compute2by1ErrorVector(
  _FloatWindow imgdiff,
  _FloatWindow gradx,
  _FloatWindow grady,
  int width, int height,
  float step_factor,
  float *ex, float *ey)
{
  int total_pixels = width * height;
  float l_ex = 0.0f, l_ey = 0.0f;

  for (int i = 0; i < total_pixels; i++) {
    float diff = imgdiff[i];
    l_ex += diff * gradx[i];
    l_ey += diff * grady[i];
  }

  *ex = l_ex * step_factor;
  *ey = l_ey * step_factor;
}

/*********************************************************************
 * _solveEquation
 *********************************************************************/
static int _solveEquation(
  float gxx, float gxy, float gyy,
  float ex, float ey,
  float small,
  float *dx, float *dy)
{
  float det = gxx * gyy - gxy * gxy;

  if (det < small) return KLT_SMALL_DET;

  *dx = (gyy * ex - gxy * ey) / det;
  *dy = (gxx * ey - gxy * ex) / det;
  return KLT_TRACKED;
}

/*********************************************************************
 * _sumAbsFloatWindow
 *********************************************************************/
#pragma acc routine seq
static float _sumAbsFloatWindow(_FloatWindow fw, int width, int height)
{
  int size = width * height;
  float sum = 0.0f;

  for (int i = 0; i < size; i++) {
    sum += fabsf(fw[i]);
  }
  return sum;
}

/*********************************************************************
 * _trackFeatureSingleLevel
 * GPU-accelerated feature tracking at single pyramid level
 *********************************************************************/
#pragma acc routine seq
static int _trackFeatureSingleLevel(
  float x1, float y1, float *x2, float *y2,
  float *img1_data, int img1_ncols, int img1_nrows,
  float *gradx1_data, int gradx1_ncols, int gradx1_nrows,
  float *grady1_data, int grady1_ncols, int grady1_nrows,
  float *img2_data, int img2_ncols, int img2_nrows,
  float *gradx2_data, int gradx2_ncols, int gradx2_nrows,
  float *grady2_data, int grady2_ncols, int grady2_nrows,
  int width, int height, 
  float step_factor, int max_iterations,
  float small, float th, float max_residue, 
  int lighting_insensitive,
  _FloatWindow imgdiff, _FloatWindow gradx, _FloatWindow grady)
{
  float gxx, gxy, gyy, ex, ey, dx, dy;
  int iteration = 0;
  int status = KLT_TRACKED;
  int hw = width/2;
  int hh = height/2;
  float one_plus_eps = 1.001f;
  int total_pixels = width * height;
  
  do {
    // Boundary checks
    if (x1-hw < 0.0f || img1_ncols-(x1+hw) < one_plus_eps ||
        *x2-hw < 0.0f || img2_ncols-(*x2+hw) < one_plus_eps ||
        y1-hh < 0.0f || img1_nrows-(y1+hh) < one_plus_eps ||
        *y2-hh < 0.0f || img2_nrows-(*y2+hh) < one_plus_eps) {
      status = KLT_OOB;
      break;
    }

    // Compute windows on GPU
    _computeIntensityDifference(img1_data, img1_ncols, img1_nrows,
                               img2_data, img2_ncols, img2_nrows,
                               x1, y1, *x2, *y2, width, height, imgdiff);
    _computeGradientSum(gradx1_data, gradx1_ncols, gradx1_nrows,
                       grady1_data, grady1_ncols, grady1_nrows,
                       gradx2_data, gradx2_ncols, gradx2_nrows,
                       grady2_data, grady2_ncols, grady2_nrows,
                       x1, y1, *x2, *y2, width, height, gradx, grady);

    // Compute matrices
    _compute2by2GradientMatrix(gradx, grady, width, height, &gxx, &gxy, &gyy);
    _compute2by1ErrorVector(imgdiff, gradx, grady, width, height, step_factor, &ex, &ey);

    // Solve equation
    status = _solveEquation(gxx, gxy, gyy, ex, ey, small, &dx, &dy);
    if (status == KLT_SMALL_DET) break;

    *x2 += dx;
    *y2 += dy;
    iteration++;

  } while ((fabsf(dx) >= th || fabsf(dy) >= th) && iteration < max_iterations);

  // Final boundary check
  if (*x2-hw < 0.0f || img2_ncols-(*x2+hw) < one_plus_eps ||
      *y2-hh < 0.0f || img2_nrows-(*y2+hh) < one_plus_eps) {
    status = KLT_OOB;
  }

  // Check residue
  if (status == KLT_TRACKED) {
    _computeIntensityDifference(img1_data, img1_ncols, img1_nrows,
                               img2_data, img2_ncols, img2_nrows,
                               x1, y1, *x2, *y2, width, height, imgdiff);
    float residue = _sumAbsFloatWindow(imgdiff, width, height) / (width * height);
    if (residue > max_residue)
      status = KLT_LARGE_RESIDUE;
  }

  if (status == KLT_SMALL_DET)  return KLT_SMALL_DET;
  else if (status == KLT_OOB)  return KLT_OOB;
  else if (status == KLT_LARGE_RESIDUE)  return KLT_LARGE_RESIDUE;
  else if (iteration >= max_iterations)  return KLT_MAX_ITERATIONS;
  else  return KLT_TRACKED;
}

/*********************************************************************
 * KLTTrackFeatures (OpenACC)
 * Main feature tracking function with parallel GPU acceleration
 *********************************************************************/
void KLTTrackFeatures(
  KLT_TrackingContext tc,
  KLT_PixelType *img1,
  KLT_PixelType *img2,
  int ncols,
  int nrows,
  KLT_FeatureList featurelist)
{
  // DEBUG: Add this at the VERY beginning of the function
  static int call_count = 0;
  call_count++;
  fprintf(stderr, "=== KLTTrackFeatures call %d ===\n", call_count);

  _KLT_FloatImage tmpimg, floatimg1, floatimg2;
  _KLT_Pyramid pyramid1, pyramid1_gradx, pyramid1_grady;
  _KLT_Pyramid pyramid2, pyramid2_gradx, pyramid2_grady;
  float subsampling = (float)tc->subsampling;
  KLT_BOOL floatimg1_created = FALSE;
  int i;

  if (KLT_verbose >= 1) {
    fprintf(stderr, "(KLT) Tracking %d features in a %d by %d image...  ",
            KLTCountRemainingFeatures(featurelist), ncols, nrows);
    fflush(stderr);
  }

  // Window size validation
  if (tc->window_width % 2 != 1) {
    tc->window_width = tc->window_width+1;
    KLTWarning("Tracking context's window width must be odd. Changing to %d.\n", tc->window_width);
  }
  if (tc->window_height % 2 != 1) {
    tc->window_height = tc->window_height+1;
    KLTWarning("Tracking context's window height must be odd. Changing to %d.\n", tc->window_height);
  }
  if (tc->window_width < 3) tc->window_width = 3;
  if (tc->window_height < 3) tc->window_height = 3;

  // Create temporary image
  tmpimg = _KLTCreateFloatImage(ncols, nrows);

  // Process first image pyramid
  if (tc->sequentialMode && tc->pyramid_last != NULL) {
    pyramid1 = (_KLT_Pyramid)tc->pyramid_last;
    pyramid1_gradx = (_KLT_Pyramid)tc->pyramid_last_gradx;
    pyramid1_grady = (_KLT_Pyramid)tc->pyramid_last_grady;
  } else {
    floatimg1_created = TRUE;
    floatimg1 = _KLTCreateFloatImage(ncols, nrows);
    _KLTToFloatImage(img1, ncols, nrows, tmpimg);
    _KLTComputeSmoothedImage(tmpimg, _KLTComputeSmoothSigma(tc), floatimg1);
    
    // Copy to GPU
    #pragma acc enter data copyin(floatimg1->data[0:ncols*nrows])
    
    pyramid1 = _KLTCreatePyramidGPU(ncols, nrows, (int)subsampling, tc->nPyramidLevels);
    _KLTComputePyramidGPU(floatimg1, pyramid1, tc->pyramid_sigma_fact);
    
    pyramid1_gradx = _KLTCreatePyramidGPU(ncols, nrows, (int)subsampling, tc->nPyramidLevels);
    pyramid1_grady = _KLTCreatePyramidGPU(ncols, nrows, (int)subsampling, tc->nPyramidLevels);
    
    for (i = 0; i < tc->nPyramidLevels; i++) {
      _KLTComputeGradients(pyramid1->img[i], tc->grad_sigma, 
                          pyramid1_gradx->img[i], pyramid1_grady->img[i]);
    }
  }

  // Process second image pyramid
  floatimg2 = _KLTCreateFloatImage(ncols, nrows);
  _KLTToFloatImage(img2, ncols, nrows, tmpimg);
  _KLTComputeSmoothedImage(tmpimg, _KLTComputeSmoothSigma(tc), floatimg2);
  
  // Copy to GPU
  #pragma acc enter data copyin(floatimg2->data[0:ncols*nrows])
  
  pyramid2 = _KLTCreatePyramidGPU(ncols, nrows, (int)subsampling, tc->nPyramidLevels);
  _KLTComputePyramidGPU(floatimg2, pyramid2, tc->pyramid_sigma_fact);
  
  pyramid2_gradx = _KLTCreatePyramidGPU(ncols, nrows, (int)subsampling, tc->nPyramidLevels);
  pyramid2_grady = _KLTCreatePyramidGPU(ncols, nrows, (int)subsampling, tc->nPyramidLevels);
  
  for (i = 0; i < tc->nPyramidLevels; i++) {
    _KLTComputeGradients(pyramid2->img[i], tc->grad_sigma,
                        pyramid2_gradx->img[i], pyramid2_grady->img[i]);
  }

  // Prepare data structures for parallel tracking
  int nFeatures = featurelist->nFeatures;
  int window_size = tc->window_width * tc->window_height;
  
  // Allocate feature arrays
  float *feature_x = (float *)malloc(nFeatures * sizeof(float));
  float *feature_y = (float *)malloc(nFeatures * sizeof(float));
  int *feature_val = (int *)malloc(nFeatures * sizeof(int));
  
  // Initialize feature arrays
  for (int indx = 0; indx < nFeatures; indx++) {
    feature_x[indx] = featurelist->feature[indx]->x;
    feature_y[indx] = featurelist->feature[indx]->y;
    feature_val[indx] = featurelist->feature[indx]->val;
  }
  
  // DEBUG: Print initial positions
  fprintf(stderr, "DEBUG: First feature initial position: (%f, %f)\n", feature_x[0], feature_y[0]);
  
  // Prepare pyramid data arrays
  float **pyramid1_img_ptrs = (float **)malloc(tc->nPyramidLevels * sizeof(float *));
  float **pyramid1_gradx_ptrs = (float **)malloc(tc->nPyramidLevels * sizeof(float *));
  float **pyramid1_grady_ptrs = (float **)malloc(tc->nPyramidLevels * sizeof(float *));
  float **pyramid2_img_ptrs = (float **)malloc(tc->nPyramidLevels * sizeof(float *));
  float **pyramid2_gradx_ptrs = (float **)malloc(tc->nPyramidLevels * sizeof(float *));
  float **pyramid2_grady_ptrs = (float **)malloc(tc->nPyramidLevels * sizeof(float *));
  int *pyramid_ncols = (int *)malloc(tc->nPyramidLevels * sizeof(int));
  int *pyramid_nrows = (int *)malloc(tc->nPyramidLevels * sizeof(int));
  
  for (i = 0; i < tc->nPyramidLevels; i++) {
    pyramid1_img_ptrs[i] = pyramid1->img[i]->data;
    pyramid1_gradx_ptrs[i] = pyramid1_gradx->img[i]->data;
    pyramid1_grady_ptrs[i] = pyramid1_grady->img[i]->data;
    pyramid2_img_ptrs[i] = pyramid2->img[i]->data;
    pyramid2_gradx_ptrs[i] = pyramid2_gradx->img[i]->data;
    pyramid2_grady_ptrs[i] = pyramid2_grady->img[i]->data;
    pyramid_ncols[i] = pyramid1->ncols[i];
    pyramid_nrows[i] = pyramid1->nrows[i];
  }

  // Copy data to GPU
  #pragma acc enter data copyin( \
    feature_x[0:nFeatures], feature_y[0:nFeatures], feature_val[0:nFeatures], \
    pyramid1_img_ptrs[0:tc->nPyramidLevels], pyramid1_gradx_ptrs[0:tc->nPyramidLevels], \
    pyramid1_grady_ptrs[0:tc->nPyramidLevels], pyramid2_img_ptrs[0:tc->nPyramidLevels], \
    pyramid2_gradx_ptrs[0:tc->nPyramidLevels], pyramid2_grady_ptrs[0:tc->nPyramidLevels], \
    pyramid_ncols[0:tc->nPyramidLevels], pyramid_nrows[0:tc->nPyramidLevels])
  
  // Allocate device memory for temporary windows using acc_malloc
  float *d_imgdiff = (float *)acc_malloc(window_size * nFeatures * sizeof(float));
  float *d_gradx = (float *)acc_malloc(window_size * nFeatures * sizeof(float));
  float *d_grady = (float *)acc_malloc(window_size * nFeatures * sizeof(float));
  
  if (d_imgdiff == NULL || d_gradx == NULL || d_grady == NULL) {
    KLTError("Failed to allocate device memory for temporary windows");
  }
  
  // Parallel feature tracking on GPU
  #pragma acc parallel loop gang vector \
    present(feature_x[0:nFeatures], feature_y[0:nFeatures], feature_val[0:nFeatures], \
            pyramid1_img_ptrs[0:tc->nPyramidLevels], pyramid1_gradx_ptrs[0:tc->nPyramidLevels], \
            pyramid1_grady_ptrs[0:tc->nPyramidLevels], pyramid2_img_ptrs[0:tc->nPyramidLevels], \
            pyramid2_gradx_ptrs[0:tc->nPyramidLevels], pyramid2_grady_ptrs[0:tc->nPyramidLevels], \
            pyramid_ncols[0:tc->nPyramidLevels], pyramid_nrows[0:tc->nPyramidLevels], \
            d_imgdiff[0:window_size * nFeatures], d_gradx[0:window_size * nFeatures], d_grady[0:window_size * nFeatures])
  for (int indx = 0; indx < nFeatures; indx++) {
    // Only track features that are not lost
    if (feature_val[indx] < 0) continue;
    
    // Calculate offsets for this feature's temporary arrays
    int offset = indx * window_size;
    float *imgdiff = &d_imgdiff[offset];
    float *gradx = &d_gradx[offset];
    float *grady = &d_grady[offset];
    
    float xloc = feature_x[indx];
    float yloc = feature_y[indx];
    float xlocout = xloc;
    float ylocout = yloc;
    
    int val = KLT_TRACKED;
    
    // Transform location to coarsest resolution
    for (int r = tc->nPyramidLevels - 1; r >= 0; r--) {
      xloc /= subsampling;
      yloc /= subsampling;
    }
    xlocout = xloc;
    ylocout = yloc;
    
    // Track through pyramid levels from coarsest to finest
    for (int r = tc->nPyramidLevels - 1; r >= 0; r--) {
      xloc *= subsampling;
      yloc *= subsampling;
      xlocout *= subsampling;
      ylocout *= subsampling;
      
      float *img1 = pyramid1_img_ptrs[r];
      float *gradx1 = pyramid1_gradx_ptrs[r];
      float *grady1 = pyramid1_grady_ptrs[r];
      float *img2 = pyramid2_img_ptrs[r];
      float *gradx2 = pyramid2_gradx_ptrs[r];
      float *grady2 = pyramid2_grady_ptrs[r];
      int level_ncols = pyramid_ncols[r];
      int level_nrows = pyramid_nrows[r];
      
      val = _trackFeatureSingleLevel(
        xloc, yloc, &xlocout, &ylocout,
        img1, level_ncols, level_nrows,
        gradx1, level_ncols, level_nrows,
        grady1, level_ncols, level_nrows,
        img2, level_ncols, level_nrows,
        gradx2, level_ncols, level_nrows,
        grady2, level_ncols, level_nrows,
        tc->window_width, tc->window_height,
        tc->step_factor, tc->max_iterations,
        tc->min_determinant, tc->min_displacement,
        tc->max_residue, tc->lighting_insensitive,
        imgdiff, gradx, grady);
      
      if (val == KLT_SMALL_DET || val == KLT_OOB)
        break;
    }
    
    // Check out of bounds with border
    if (xlocout < tc->borderx || xlocout > ncols - 1 - tc->borderx || 
        ylocout < tc->bordery || ylocout > nrows - 1 - tc->bordery) {
      val = KLT_OOB;
    }
    
    // Update feature
    if (val != KLT_TRACKED) {
      feature_x[indx] = -1.0f;
      feature_y[indx] = -1.0f;
      feature_val[indx] = val;
    } else {
      feature_x[indx] = xlocout;
      feature_y[indx] = ylocout;
      feature_val[indx] = KLT_TRACKED;
    }
  }
  
  // Copy results back from GPU
  #pragma acc update self(feature_x[0:nFeatures], feature_y[0:nFeatures], feature_val[0:nFeatures])
  
  // DEBUG: Print final positions
  fprintf(stderr, "DEBUG: First feature final position: (%f, %f) status: %d\n", 
          feature_x[0], feature_y[0], feature_val[0]);
  
  // Update feature list with results
  for (int indx = 0; indx < nFeatures; indx++) {
    featurelist->feature[indx]->x = feature_x[indx];
    featurelist->feature[indx]->y = feature_y[indx];
    featurelist->feature[indx]->val = feature_val[indx];
  }
  
  // Free device memory for temporary windows
  acc_free(d_imgdiff);
  acc_free(d_gradx);
  acc_free(d_grady);

  // Cleanup GPU memory
  #pragma acc exit data delete( \
    feature_x[0:nFeatures], feature_y[0:nFeatures], feature_val[0:nFeatures], \
    pyramid1_img_ptrs[0:tc->nPyramidLevels], pyramid1_gradx_ptrs[0:tc->nPyramidLevels], \
    pyramid1_grady_ptrs[0:tc->nPyramidLevels], pyramid2_img_ptrs[0:tc->nPyramidLevels], \
    pyramid2_gradx_ptrs[0:tc->nPyramidLevels], pyramid2_grady_ptrs[0:tc->nPyramidLevels], \
    pyramid_ncols[0:tc->nPyramidLevels], pyramid_nrows[0:tc->nPyramidLevels])
  
  // Free host memory
  free(feature_x);
  free(feature_y);
  free(feature_val);
  free(pyramid1_img_ptrs);
  free(pyramid1_gradx_ptrs);
  free(pyramid1_grady_ptrs);
  free(pyramid2_img_ptrs);
  free(pyramid2_gradx_ptrs);
  free(pyramid2_grady_ptrs);
  free(pyramid_ncols);
  free(pyramid_nrows);

  // Cleanup image data
  if (floatimg1_created) {
    #pragma acc exit data delete(floatimg1->data[0:ncols*nrows])
    _KLTFreeFloatImage(floatimg1);
  }
  
  #pragma acc exit data delete(floatimg2->data[0:ncols*nrows])
  _KLTFreeFloatImage(floatimg2);

  // Cleanup pyramid structures
  if (tc->sequentialMode) {
    tc->pyramid_last = pyramid2;
    tc->pyramid_last_gradx = pyramid2_gradx;
    tc->pyramid_last_grady = pyramid2_grady;
  } else {
    _KLTFreePyramidGPU(pyramid2);
    _KLTFreePyramidGPU(pyramid2_gradx);
    _KLTFreePyramidGPU(pyramid2_grady);
    if (floatimg1_created) {
      _KLTFreePyramidGPU(pyramid1);
      _KLTFreePyramidGPU(pyramid1_gradx);
      _KLTFreePyramidGPU(pyramid1_grady);
    }
  }

  _KLTFreeFloatImage(tmpimg);

  if (KLT_verbose >= 1) {
    fprintf(stderr, "\n\t%d features successfully tracked.\n", KLTCountRemainingFeatures(featurelist));
    fflush(stderr);
  }
}
