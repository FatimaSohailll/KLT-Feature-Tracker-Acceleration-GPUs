/*********************************************************************
 * trackFeatures.cu
 *
 *********************************************************************/

/* Standard includes */
#include <assert.h>
#include <math.h>		/* fabs() */
#include <stdlib.h>		/* malloc() */
#include <stdio.h>		/* fflush() */
#include <cuda.h>

/* Our includes */
#include "base.h"
#include "error.h"
#include "convolve.h"	/* for computing pyramid */
#include "klt.h"
#include "klt_util.h"	/* _KLT_FloatImage */
#include "pyramid.h"	/* _KLT_Pyramid */

extern int KLT_verbose;

typedef float *_FloatWindow;

 /*******************************************************************
  *************************** GPU FUNCTIONS *************************
  *******************************************************************/

/*********************************************************************
 * _interpolate
 * 
 * Given a point (x,y) in an image, computes the bilinear interpolated 
 * gray-level value of the point in the image.  
 */

__device__ float gpu_interpolate(float x, float y, const float* img, int cols, int rows) {
    int xt = (int)x; // coordinates of top left corner
    int yt = (int)y;
    float ax = x - xt;
    float ay = y - yt;

    // check boundary / range
    if (xt < 0) xt = 0; // clamp to 0
    if (yt < 0) yt = 0;
    if (xt >= cols - 1) xt = cols - 2;
    if (yt >= rows - 1) yt = rows - 2;

    const float* ptr = img + (cols * yt) + xt;

    float val = (1 - ax) * (1 - ay) * ptr[0] +
                ax * (1 - ay) * ptr[1] +
                (1 - ax) * ay * ptr[cols] +
                ax * ay * ptr[cols + 1];

    return val;
}

/*********************************************************************
 * _computeGradientSum
 *
 * Given two gradients and the window center in both images,
 * aligns the gradients wrt the window and computes the sum of the two 
 * overlaid gradients.
 */

__device__ void gpu_computeGradientSum(
    const float* gradx1, const float* grady1, 
    const float* gradx2, const float* grady2,
    float* out_gradx, float* out_grady, 
    int cols, int rows, 
    float x1, float y1, float x2, float y2, 
    int width, int height) {
    
    int w = width / 2;
    int h = height / 2;

    for (int j = -h; j <= h; j++) {
        for (int i = -w; i <= w; i++) {
            int idx = (j + h) * width + (i + w);

            // absolute coordinates of both images
            float x1t = x1 + i;
            float y1t = y1 + j;
            float x2t = x2 + i;
            float y2t = y2 + j;

            float g1x = gpu_interpolate(x1t, y1t, gradx1, cols, rows);
            float g2x = gpu_interpolate(x2t, y2t, gradx2, cols, rows);
            out_gradx[idx] = g1x + g2x;

            float g1y = gpu_interpolate(x1t, y1t, grady1, cols, rows);
            float g2y = gpu_interpolate(x2t, y2t, grady2, cols, rows);
            out_grady[idx] = g1y + g2y;
        }
    }
}

/*********************************************************************
 * _computeIntensityDifference
 *
 * Given two images and the window center in both images,
 * aligns the images wrt the window and computes the difference 
 * between the two overlaid images.
 */

__device__ void gpu_computeIntensityDifference(
    const float* img1, const float* img2, 
    int cols, int rows,
    float x1, float y1, float x2, float y2, 
    int width, int height, 
    float* imgDiff) {
    
    int w = width / 2;
    int h = height / 2;

    for (int j = -h; j <= h; j++) {
        for (int i = -w; i <= w; i++) {
            int idx = (j + h) * width + (i + w);

            // calculate interpolated values
            float g1 = gpu_interpolate(x1 + i, y1 + j, img1, cols, rows);
            float g2 = gpu_interpolate(x2 + i, y2 + j, img2, cols, rows);
            
            imgDiff[idx] = g1 - g2;
        }
    }
}

/*********************************************************************
 * _computeIntensityDifferenceLightingInsensitive
 *
 * Given two images and the window center in both images,
 * aligns the images wrt the window and computes the difference 
 * between the two overlaid images; normalizes for overall gain and bias.
 */
__device__ void gpu_computeIntensityDifferenceLightingInsensitive(
    const float* img1, const float* img2, 
    float x1, float y1, float x2, float y2,
    int width, int height, 
    int cols, int rows, 
    float* imgDiff) {
    
    int w = width / 2;
    int h = height / 2;
    float sum1 = 0.0f, sum2 = 0.0f;
    float sum1_sq = 0.0f, sum2_sq = 0.0f;
    int total = width * height;

    // compute sums and squared sums
    for (int j = -h; j <= h; j++) {
        for (int i = -w; i <= w; i++) {
            float g1 = gpu_interpolate(x1 + i, y1 + j, img1, cols, rows);
            float g2 = gpu_interpolate(x2 + i, y2 + j, img2, cols, rows);
            sum1 += g1;
            sum2 += g2;
            sum1_sq += g1 * g1;
            sum2_sq += g2 * g2;
        }
    }

    float mean1_sq = sum1_sq / total;
    float mean2_sq = sum2_sq / total;
    float alpha = sqrtf(mean1_sq / (mean2_sq + 1e-6f));  // avoid div-by-zero

    float mean1 = sum1 / total;
    float mean2 = sum2 / total;
    float beta = mean1 - alpha * mean2;

    // compute adjusted differences
    int idx = 0;
    for (int j = -h; j <= h; j++) {
        for (int i = -w; i <= w; i++) {
            float g1 = gpu_interpolate(x1 + i, y1 + j, img1, cols, rows);
            float g2 = gpu_interpolate(x2 + i, y2 + j, img2, cols, rows);
            imgDiff[idx++] = g1 - (alpha * g2 + beta);
        }
    }
}

/*********************************************************************
 * _computeGradientSumLightingInsensitive
 *
 * Given two gradients and the window center in both images,
 * aligns the gradients wrt the window and computes the sum of the two 
 * overlaid gradients; normalizes for overall gain and bias.
 */

__device__ void gpu_computeGradientSumLightingInsensitive(
    const float* gradx1, const float* grady1,
    const float* gradx2, const float* grady2,
    const float* img1, const float* img2,
    float x1, float y1, float x2, float y2,
    int width, int height,
    int cols, int rows,
    float* out_gradx, float* out_grady) {
    
    int w = width / 2;
    int h = height / 2;

    float sum1 = 0.0f, sum2 = 0.0f;
    int total = width * height;

    for (int j = -h; j <= h; j++) {
        for (int i = -w; i <= w; i++) {
            float g1 = gpu_interpolate(x1 + i, y1 + j, img1, cols, rows);
            float g2 = gpu_interpolate(x2 + i, y2 + j, img2, cols, rows);
            sum1 += g1;
            sum2 += g2;
        }
    }

    float mean1 = sum1 / total;
    float mean2 = sum2 / total;

    // Avoid division by zero
    float alpha = sqrtf(mean1 / (mean2 + 1e-6f));

    // compute adjusted gradient sums
    int idx = 0;
    for (int j = -h; j <= h; j++) {
        for (int i = -w; i <= w; i++) {
            float g1x = gpu_interpolate(x1 + i, y1 + j, gradx1, cols, rows);
            float g2x = gpu_interpolate(x2 + i, y2 + j, gradx2, cols, rows);
            out_gradx[idx] = g1x + g2x * alpha;

            float g1y = gpu_interpolate(x1 + i, y1 + j, grady1, cols, rows);
            float g2y = gpu_interpolate(x2 + i, y2 + j, grady2, cols, rows);
            out_grady[idx] = g1y + g2y * alpha;

            idx++;
        }
    }
}

/*********************************************************************
 * _compute2by2GradientMatrix
 *
 */

__device__ void gpu_compute2by2GradientMatrix(
    const float* gradx, const float* grady, 
    int width, int height,
    float* gxx, float* gxy, float* gyy) {
    
    float sum_xx = 0.0f;
    float sum_xy = 0.0f;
    float sum_yy = 0.0f;

    const int total = width * height;

    for (int i = 0; i < total; i++) {
        float gx = gradx[i];
        float gy = grady[i];
        sum_xx += gx * gx;
        sum_xy += gx * gy;
        sum_yy += gy * gy;
    }

    *gxx = sum_xx;
    *gxy = sum_xy;
    *gyy = sum_yy;
}

/*********************************************************************
 * _compute2by1ErrorVector
 *
 */

__device__ void gpu_compute2by1ErrorVector(
    const float* imgDiff, const float* gradx, const float* grady,
    int width, int height, 
    float step_factor, 
    float* ex, float* ey) {
    
    float sum_ex = 0.0f;
    float sum_ey = 0.0f;

    int total = width * height;

    for (int i = 0; i < total; i++) {
        float diff = imgDiff[i];
        sum_ex += diff * gradx[i];
        sum_ey += diff * grady[i];
    }

    // apply step factor
    *ex = sum_ex * step_factor;
    *ey = sum_ey * step_factor;
}

/*********************************************************************
 * _solveEquation
 *
 * Solves the 2x2 matrix equation
 *         [gxx gxy] [dx] = [ex]
 *         [gxy gyy] [dy] = [ey]
 * for dx and dy.
 *
 * Returns KLT_TRACKED on success and KLT_SMALL_DET on failure
 */

__device__ int gpu_solveEquation(
    float gxx, float gxy, float gyy, 
    float ex, float ey,
    float small, 
    float *dx, float *dy) {
    
    float det = gxx * gyy - gxy * gxy;

    if (det < small) return KLT_SMALL_DET;

    *dx = (gyy * ex - gxy * ey) / det;
    *dy = (gxx * ey - gxy * ex) / det;
    return KLT_TRACKED;
}

/*********************************************************************
 * _sumAbsFloatWindow
 */

__device__ float gpu_sumAbsFloatWindow(const float* fw, int width, int height) {
    float sum = 0.0f;
    int total = width * height;

    for (int i = 0; i < total; i++)
        sum += fabsf(fw[i]);
    return sum;
}

/*********************************************************************
 * _trackFeature
 *
 * Tracks a feature point from one image to the next.
 *
 * RETURNS
 * KLT_SMALL_DET if feature is lost,
 * KLT_MAX_ITERATIONS if tracking stopped because iterations timed out,
 * KLT_TRACKED otherwise.
 */
__device__ int gpu_trackFeature(
    float x1, float y1,                    // location in first image
    float* x2, float* y2,                  // location in second image (updated)
    const float* img1, const float* gradx1, const float* grady1,
    const float* img2, const float* gradx2, const float* grady2,
    int nc, int nr,                        // image dimensions
    int width, int height,                 // window size
    float step_factor,
    int max_iterations,
    float small,
    float th,
    float max_residue,
    int lighting_insensitive,
    float* imgdiff,                        // pre-allocated buffers
    float* gradx,
    float* grady) {
    
    float gxx, gxy, gyy, ex, ey, dx, dy;
    int iteration = 0;
    int status = KLT_TRACKED;
    int hw = width / 2;
    int hh = height / 2;
    float one_plus_eps = 1.001f;

    do {
        // bounds check
        if (x1 - hw < 0.0f || nc - (x1 + hw) < one_plus_eps ||
            *x2 - hw < 0.0f || nc - (*x2 + hw) < one_plus_eps ||
            y1 - hh < 0.0f || nr - (y1 + hh) < one_plus_eps ||
            *y2 - hh < 0.0f || nr - (*y2 + hh) < one_plus_eps) {
            status = KLT_OOB;
            break;
        }

        // intensity & gradient computation
        if (lighting_insensitive) {
            gpu_computeIntensityDifferenceLightingInsensitive(
                img1, img2, x1, y1, *x2, *y2, 
                width, height, nc, nr, imgdiff);
            gpu_computeGradientSumLightingInsensitive(
                gradx1, grady1, gradx2, grady2,
                img1, img2, x1, y1, *x2, *y2,
                width, height, nc, nr, gradx, grady);
        } else {
            gpu_computeIntensityDifference(
                img1, img2, nc, nr, x1, y1, *x2, *y2, 
                width, height, imgdiff);
            gpu_computeGradientSum(
                gradx1, grady1, gradx2, grady2,
                gradx, grady, nc, nr, x1, y1, *x2, *y2, 
                width, height);
        }

        // compute matrices
        gpu_compute2by2GradientMatrix(gradx, grady, width, height, &gxx, &gxy, &gyy);
        gpu_compute2by1ErrorVector(imgdiff, gradx, grady, width, height, step_factor, &ex, &ey);

        // solve for displacement
        status = gpu_solveEquation(gxx, gxy, gyy, ex, ey, small, &dx, &dy);
        if (status == KLT_SMALL_DET)
            break;

        *x2 += dx;
        *y2 += dy;
        iteration++;

    } while ((fabsf(dx) >= th || fabsf(dy) >= th) && iteration < max_iterations);

    // check out of bounds again
    if (status == KLT_TRACKED && 
        (*x2 - hw < 0.0f || nc - (*x2 + hw) < one_plus_eps ||
         *y2 - hh < 0.0f || nr - (*y2 + hh) < one_plus_eps)) {
        status = KLT_OOB;
    }

    // compute residue
    if (status == KLT_TRACKED) {
        if (lighting_insensitive) {
            gpu_computeIntensityDifferenceLightingInsensitive(
                img1, img2, x1, y1, *x2, *y2, 
                width, height, nc, nr, imgdiff);
        } else {
            gpu_computeIntensityDifference(
                img1, img2, nc, nr, x1, y1, *x2, *y2, 
                width, height, imgdiff);
        }

        float residue = gpu_sumAbsFloatWindow(imgdiff, width, height) / (width * height);
        if (residue > max_residue) {
            status = KLT_LARGE_RESIDUE;
        }
    }

    if (status == KLT_SMALL_DET) return KLT_SMALL_DET;
    else if (status == KLT_OOB) return KLT_OOB;
    else if (status == KLT_LARGE_RESIDUE) return KLT_LARGE_RESIDUE;
    else if (iteration >= max_iterations) return KLT_MAX_ITERATIONS;
    else return KLT_TRACKED;
}

/*********************************** KERNEL *****************/

__global__ void trackFeatureKernel(
    const float* img1, const float* gradx1, const float* grady1,
    const float* img2, const float* gradx2, const float* grady2,
    int nc, int nr,
    float* x1_list, float* y1_list,
    float* x2_list, float* y2_list,
    int num_features,
    int width, int height,
    float step_factor,
    int max_iterations,
    float small,
    float th,
    float max_residue,
    int lighting_insensitive,
    float* imgdiff_buf,    // Pre-allocated buffers for all features
    float* gradx_buf,
    float* grady_buf,
    int* status_out) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_features) return;

    // Get pointers to this thread's buffers
    float* imgdiff = imgdiff_buf + idx * width * height;
    float* gradx = gradx_buf + idx * width * height;
    float* grady = grady_buf + idx * width * height;

    float x1 = x1_list[idx];
    float y1 = y1_list[idx];
    float x2 = x2_list[idx];
    float y2 = y2_list[idx];

    int status = gpu_trackFeature(
        x1, y1, &x2, &y2,
        img1, gradx1, grady1,
        img2, gradx2, grady2,
        nc, nr, width, height,
        step_factor, max_iterations,
        small, th, max_residue,
        lighting_insensitive,
        imgdiff, gradx, grady);

    x2_list[idx] = x2;
    y2_list[idx] = y2;
    status_out[idx] = status;
}

/*********************************************************************/

static KLT_BOOL _outOfBounds(
    float x,
    float y,
    int ncols,
    int nrows,
    int borderx,
    int bordery) {
    
    return (x < borderx || x > ncols - 1 - borderx ||
            y < bordery || y > nrows - 1 - bordery);
}

/**********************************************************************
* CONSISTENCY CHECK OF FEATURES BY AFFINE MAPPING (BEGIN)
* 
* Created by: Thorsten Thormaehlen (University of Hannover) June 2004    
* thormae@tnt.uni-hannover.de
* 
* Permission is granted to any individual or institution to use, copy, modify,
* and distribute this part of the software, provided that this complete authorship 
* and permission notice is maintained, intact, in all copies. 
*
* This software is provided  "as is" without express or implied warranty.
*/

#define SWAP_ME(X,Y) {float temp=(X);(X)=(Y);(Y)=temp;}

static float **_am_matrix(long nr, long nc) {
    float **m;
    int a;
    m = (float **)malloc((size_t)(nr * sizeof(float*)));
    m[0] = (float *)malloc((size_t)((nr * nc) * sizeof(float)));
    for (a = 1; a < nr; a++) m[a] = m[a - 1] + nc;
    return m;
}

static void _am_free_matrix(float **m) {
    free(m[0]);
    free(m);
}

__device__ int gpu_am_gauss_jordan_elimination(float *a, int n, float *b, int m) {
    int* indxc = (int*)malloc(n * sizeof(int));
    int* indxr = (int*)malloc(n * sizeof(int));
    int* ipiv = (int*)malloc(n * sizeof(int));
    
    int i, j, k, l, ll;
    float big, dum, pivinv;
    int col = 0;
    int row = 0;

    for (j = 0; j < n; j++) ipiv[j] = 0;
    for (i = 0; i < n; i++) {
        big = 0.0f;
        for (j = 0; j < n; j++) {
            if (ipiv[j] != 1) {
                for (k = 0; k < n; k++) {
                    if (ipiv[k] == 0) {
                        float val = fabsf(a[j * n + k]);
                        if (val >= big) {
                            big = val;
                            row = j;
                            col = k;
                        }
                    } else if (ipiv[k] > 1) {
                        free(indxc); free(indxr); free(ipiv);
                        return KLT_SMALL_DET;
                    }
                }
            }
        }
        ipiv[col] += 1;
        if (row != col) {
            for (l = 0; l < n; l++) {
                SWAP_ME(a[row * n + l], a[col * n + l]);
            }
            for (l = 0; l < m; l++) {
                SWAP_ME(b[row * m + l], b[col * m + l]);
            }
        }
        indxr[i] = row;
        indxc[i] = col;
        if (a[col * n + col] == 0.0f) {
            free(indxc); free(indxr); free(ipiv);
            return KLT_SMALL_DET;
        }
        pivinv = 1.0f / a[col * n + col];
        a[col * n + col] = 1.0f;
        for (l = 0; l < n; l++) a[col * n + l] *= pivinv;
        for (l = 0; l < m; l++) b[col * m + l] *= pivinv;
        for (ll = 0; ll < n; ll++) {
            if (ll != col) {
                dum = a[ll * n + col];
                a[ll * n + col] = 0.0f;
                for (l = 0; l < n; l++) a[ll * n + l] -= a[col * n + l] * dum;
                for (l = 0; l < m; l++) b[ll * m + l] -= b[col * m + l] * dum;
            }
        }
    }
    for (l = n - 1; l >= 0; l--) {
        if (indxr[l] != indxc[l]) {
            for (k = 0; k < n; k++) {
                SWAP_ME(a[k * n + indxr[l]], a[k * n + indxc[l]]);
            }
        }
    }

    free(indxc); free(indxr); free(ipiv);
    return KLT_TRACKED;
}

/*********************************************************************
 * _am_getGradientWinAffine
 *
 * aligns the gradients with the affine transformed window 
 */

__device__ void gpu_am_getGradientWinAffine(
    const float* in_gradx,
    const float* in_grady,
    float x, float y,                    /* center of window*/
    float Axx, float Ayx, float Axy, float Ayy,  /* affine mapping */
    int width, int height,               /* size of window */
    int cols, int rows,
    float* out_gradx,                    /* output */
    float* out_grady) {                  /* output */
    
    int hw = width / 2, hh = height / 2;
    int idx = 0;
    
    for (int j = -hh; j <= hh; j++) {
        for (int i = -hw; i <= hw; i++) {
            float mi = Axx * i + Axy * j;
            float mj = Ayx * i + Ayy * j;

            /* Compute values */
            float gx = gpu_interpolate(x + mi, y + mj, in_gradx, cols, rows);
            float gy = gpu_interpolate(x + mi, y + mj, in_grady, cols, rows);
            
            out_gradx[idx] = gx;
            out_grady[idx] = gy;
            idx++;
        }
    }
}

/*********************************************************************
 * _am_computeIntensityDifferenceAffine
 *
 * Given two images and the window center in both images,
 * aligns the images with the window and computes the difference 
 * between the two overlaid images using the affine mapping.
 *       A =  [ Axx Axy]
 *            [ Ayx Ayy]        
*/

__device__ void gpu_am_computeIntensityDifferenceAffine(
    const float* img1,                   /* images */
    const float* img2,
    float x1, float y1,                  /* center of window in 1st img */
    float x2, float y2,                  /* center of window in 2nd img */
    float Axx, float Ayx, float Axy, float Ayy,  /* affine mapping */   
    int width, int height,               /* size of window */
    int cols1, int rows1,
    int cols2, int rows2,
    float* imgdiff) {                    /* output */
    
    int hw = width / 2, hh = height / 2;
    int idx = 0;
    
    for (int j = -hh; j <= hh; j++) {
        for (int i = -hw; i <= hw; i++) {
            float g1 = gpu_interpolate(x1 + i, y1 + j, img1, cols1, rows1);

            float mi = Axx * i + Axy * j;
            float mj = Ayx * i + Ayy * j;
            
            float g2 = gpu_interpolate(x2 + mi, y2 + mj, img2, cols2, rows2);
            
            imgdiff[idx++] = g1 - g2;
        }
    }
}

/*********************************************************************
 * _am_compute6by6GradientMatrix
 *
 */

__device__ void gpu_am_compute6by6GradientMatrix(
    const float* gradx,   // flattened gradient window
    const float* grady,
    int width,
    int height,
    float* T              // 6x6 matrix, flattened as row-major (T[row*6 + col])
) {
    int hw = width / 2;
    int hh = height / 2;

    // Initialize T to zero
    for (int i = 0; i < 36; i++) T[i] = 0.0f;

    int idx = 0; // index in flattened window
    for (int j = -hh; j <= hh; j++) {
        for (int i = -hw; i <= hw; i++) {
            float gx = gradx[idx];
            float gy = grady[idx];
            float gxx = gx * gx;
            float gxy = gx * gy;
            float gyy = gy * gy;
            float x = (float)i;
            float y = (float)j;
            float xx = x * x;
            float xy = x * y;
            float yy = y * y;

            T[0 * 6 + 0] += xx * gxx;
            T[0 * 6 + 1] += xx * gxy;
            T[0 * 6 + 2] += xy * gxx;
            T[0 * 6 + 3] += xy * gxy;
            T[0 * 6 + 4] += x * gxx;
            T[0 * 6 + 5] += x * gxy;

            T[1 * 6 + 1] += xx * gyy;
            T[1 * 6 + 2] += xy * gxy;
            T[1 * 6 + 3] += xy * gyy;
            T[1 * 6 + 4] += x * gxy;
            T[1 * 6 + 5] += x * gyy;

            T[2 * 6 + 2] += yy * gxx;
            T[2 * 6 + 3] += yy * gxy;
            T[2 * 6 + 4] += y * gxx;
            T[2 * 6 + 5] += y * gxy;

            T[3 * 6 + 3] += yy * gyy;
            T[3 * 6 + 4] += y * gxy;
            T[3 * 6 + 5] += y * gyy;

            T[4 * 6 + 4] += gxx;
            T[4 * 6 + 5] += gxy;

            T[5 * 6 + 5] += gyy;

            idx++;
        }
    }

    // Mirror upper triangle to lower triangle
    for (int j = 0; j < 5; j++) {
        for (int i = j + 1; i < 6; i++) {
            T[i * 6 + j] = T[j * 6 + i];
        }
    }
}

/*********************************************************************
 * _am_compute6by1ErrorVector
 *
 */

__device__ void gpu_am_compute6by1ErrorVector(
    const float* imgdiff,   // flattened window
    const float* gradx,
    const float* grady,
    int width,
    int height,
    float* e                // 6x1 vector, flattened
) {
    int hw = width / 2;
    int hh = height / 2;

    // Initialize e to zero
    for (int i = 0; i < 6; i++) e[i] = 0.0f;

    int idx = 0; // flattened index into the window
    for (int j = -hh; j <= hh; j++) {
        for (int i = -hw; i <= hw; i++) {
            float diff = imgdiff[idx];
            float diffgradx = diff * gradx[idx];
            float diffgrady = diff * grady[idx];

            e[0] += diffgradx * i;
            e[1] += diffgrady * i;
            e[2] += diffgradx * j;
            e[3] += diffgrady * j;
            e[4] += diffgradx;
            e[5] += diffgrady;

            idx++;
        }
    }

    // Multiply each element by 0.5
    for (int i = 0; i < 6; i++) e[i] *= 0.5f;
}

/*********************************************************************
 * _am_compute4by4GradientMatrix
 *
 */

__device__ void gpu_am_compute4by4GradientMatrix(
    const float* gradx,
    const float* grady,
    int width,   /* size of window */
    int height,
    float* T) {  /* return values */
    
    int hw = width / 2, hh = height / 2;
 
    /* Set values to zero */ 
    for (int i = 0; i < 16; i++) {
        T[i] = 0.0f;
    }
  
    int idx = 0;
    for (int j = -hh; j <= hh; j++) {
        for (int i = -hw; i <= hw; i++) {
            float gx = gradx[idx];
            float gy = grady[idx];
            float x = (float)i;
            float y = (float)j;

            float xgx_ygy = x * gx + y * gy;
            float xgy_ygx = x * gy - y * gx;

            T[0 * 4 + 0] += xgx_ygy * xgx_ygy;
            T[0 * 4 + 1] += xgx_ygy * xgy_ygx;
            T[0 * 4 + 2] += xgx_ygy * gx;
            T[0 * 4 + 3] += xgx_ygy * gy;

            T[1 * 4 + 1] += xgy_ygx * xgy_ygx;
            T[1 * 4 + 2] += xgy_ygx * gx;
            T[1 * 4 + 3] += xgy_ygx * gy;

            T[2 * 4 + 2] += gx * gx;
            T[2 * 4 + 3] += gx * gy;

            T[3 * 4 + 3] += gy * gy;

            idx++;
        }
    }
  
    // Fill symmetric elements
    T[1 * 4 + 0] = T[0 * 4 + 1];
    T[2 * 4 + 0] = T[0 * 4 + 2];
    T[3 * 4 + 0] = T[0 * 4 + 3];
    T[2 * 4 + 1] = T[1 * 4 + 2];
    T[3 * 4 + 1] = T[1 * 4 + 3];
    T[3 * 4 + 2] = T[2 * 4 + 3];
}

/*********************************************************************
 * _am_compute4by1ErrorVector
 *
 */

__device__ void gpu_am_compute4by1ErrorVector(
    const float* imgdiff,
    const float* gradx,
    const float* grady,
    int width,   /* size of window */
    int height,
    float* e) {  /* return values */
    
    int hw = width / 2, hh = height / 2;

    /* Set values to zero */  
    for (int i = 0; i < 4; i++) e[i] = 0.0f;
  
    /* Compute values */
    int idx = 0; 
    for (int j = -hh; j <= hh; j++) {
        for (int i = -hw; i <= hw; i++) {
            float diff = imgdiff[idx];
            float diffgradx = diff * gradx[idx];
            float diffgrady = diff * grady[idx];

            e[0] += diffgradx * i + diffgrady * j;
            e[1] += diffgrady * i - diffgradx * j;
            e[2] += diffgradx;
            e[3] += diffgrady;

            idx++;
        }
    }
  
    for (int i = 0; i < 4; i++) e[i] *= 0.5f;
}

/*********************************************************************
 * _am_trackFeatureAffine
 *
 * Tracks a feature point from the image of first occurrence to the actual image.
 */

__device__ int gpu_am_trackFeatureAffine(
    float x1,  /* location of window in first image */
    float y1,
    float* x2, /* starting location of search in second image */
    float* y2,
    const float* img1, int nc1, int nr1,
    const float* gradx1,
    const float* grady1,
    const float* img2, int nc2, int nr2,
    const float* gradx2,
    const float* grady2,
    int width,           /* size of window */
    int height,
    float step_factor,
    int max_iterations,
    float small,         /* determinant threshold */
    float th,            /* displacement threshold */
    float th_aff,
    float max_residue,   /* residue threshold */
    int lighting_insensitive,
    int affine_map,      /* affine mapping type */
    float mdd,           /* max displacement difference */
    float* Axx, float* Ayx, 
    float* Axy, float* Ayy,
    float* imgdiff, float* gradx, float* grady, 
    float* T, float* a) {
    
    int iteration = 0;
    int status = KLT_TRACKED;
    int hw = width / 2;
    int hh = height / 2;
    float one_plus_eps = 1.001f;
    bool convergence = false;

    /* Iteratively update the window position */
    do {
        if (!affine_map) {
            /* Standard tracking */
            if (x1 - hw < 0.0f || nc1 - (x1 + hw) < one_plus_eps ||
                *x2 - hw < 0.0f || nc2 - (*x2 + hw) < one_plus_eps ||
                y1 - hh < 0.0f || nr1 - (y1 + hh) < one_plus_eps ||
                *y2 - hh < 0.0f || nr2 - (*y2 + hh) < one_plus_eps) {
                status = KLT_OOB;
                break;
            }
            
            if (lighting_insensitive) {
                gpu_computeIntensityDifferenceLightingInsensitive(
                    img1, img2, x1, y1, *x2, *y2, 
                    width, height, nc1, nr1, imgdiff);
                gpu_computeGradientSumLightingInsensitive(
                    gradx1, grady1, gradx2, grady2,
                    img1, img2, x1, y1, *x2, *y2,
                    width, height, nc1, nr1, gradx, grady);
            } else {
                gpu_computeIntensityDifference(
                    img1, img2, nc1, nr1, x1, y1, *x2, *y2, 
                    width, height, imgdiff);
                gpu_computeGradientSum(
                    gradx1, grady1, gradx2, grady2,
                    gradx, grady, nc1, nr1, x1, y1, *x2, *y2, 
                    width, height);
            }
            
            float gxx, gxy, gyy, ex, ey, dx, dy;
            gpu_compute2by2GradientMatrix(gradx, grady, width, height, &gxx, &gxy, &gyy);
            gpu_compute2by1ErrorVector(imgdiff, gradx, grady, width, height, step_factor, &ex, &ey);
            
            status = gpu_solveEquation(gxx, gxy, gyy, ex, ey, small, &dx, &dy);
            convergence = (fabsf(dx) < th && fabsf(dy) < th);
            
            *x2 += dx;
            *y2 += dy;
            
        } else {
            /* Affine tracking */
            float ul_x = *Axx * (-hw) + *Axy * hh + *x2;
            float ul_y = *Ayx * (-hw) + *Ayy * hh + *y2;
            float ll_x = *Axx * (-hw) + *Axy * (-hh) + *x2;
            float ll_y = *Ayx * (-hw) + *Ayy * (-hh) + *y2;
            float ur_x = *Axx * hw + *Axy * hh + *x2;
            float ur_y = *Ayx * hw + *Ayy * hh + *y2;
            float lr_x = *Axx * hw + *Axy * (-hh) + *x2;
            float lr_y = *Ayx * hw + *Ayy * (-hh) + *y2;

            if (x1 - hw < 0.0f || nc1 - (x1 + hw) < one_plus_eps ||
                y1 - hh < 0.0f || nr1 - (y1 + hh) < one_plus_eps ||
                ul_x < 0.0f || nc2 - ul_x < one_plus_eps ||
                ll_x < 0.0f || nc2 - ll_x < one_plus_eps ||
                ur_x < 0.0f || nc2 - ur_x < one_plus_eps ||
                lr_x < 0.0f || nc2 - lr_x < one_plus_eps ||
                ul_y < 0.0f || nr2 - ul_y < one_plus_eps ||
                ll_y < 0.0f || nr2 - ll_y < one_plus_eps ||
                ur_y < 0.0f || nr2 - ur_y < one_plus_eps ||
                lr_y < 0.0f || nr2 - lr_y < one_plus_eps) {
                status = KLT_OOB;
                break;
            }
            
            gpu_am_computeIntensityDifferenceAffine(
                img1, img2, x1, y1, *x2, *y2, *Axx, *Ayx, *Axy, *Ayy,
                width, height, nc1, nr1, nc2, nr2, imgdiff);
            gpu_am_getGradientWinAffine(
                gradx2, grady2, *x2, *y2, *Axx, *Ayx, *Axy, *Ayy,
                width, height, nc2, nr2, gradx, grady);

            float dx, dy;
            switch (affine_map) {
                case 1:
                    gpu_am_compute4by1ErrorVector(imgdiff, gradx, grady, width, height, a);
                    gpu_am_compute4by4GradientMatrix(gradx, grady, width, height, T);
                    if (gpu_am_gauss_jordan_elimination(T, 4, a, 1) != KLT_TRACKED) {
                        status = KLT_SMALL_DET;
                        break;
                    }
                    
                    *Axx += a[0];
                    *Ayx += a[1];
                    *Ayy = *Axx;
                    *Axy = -(*Ayx);
                    
                    dx = a[2];
                    dy = a[3];
                    break;
                    
                case 2:
                    gpu_am_compute6by1ErrorVector(imgdiff, gradx, grady, width, height, a);
                    gpu_am_compute6by6GradientMatrix(gradx, grady, width, height, T);
                    if (gpu_am_gauss_jordan_elimination(T, 6, a, 1) != KLT_TRACKED) {
                        status = KLT_SMALL_DET;
                        break;
                    }
                    
                    *Axx += a[0];
                    *Ayx += a[1];
                    *Axy += a[2];
                    *Ayy += a[3];
                    dx = a[4];
                    dy = a[5];
                    break;
                    
                default:
                    status = KLT_SMALL_DET;
                    break;
            }
            
            if (status == KLT_SMALL_DET) break;
            
            *x2 += dx;
            *y2 += dy;
            
            // Check convergence for affine case
            float new_ul_x = *Axx * (-hw) + *Axy * hh + *x2;
            float new_ul_y = *Ayx * (-hw) + *Ayy * hh + *y2;
            float new_ll_x = *Axx * (-hw) + *Axy * (-hh) + *x2;
            float new_ll_y = *Ayx * (-hw) + *Ayy * (-hh) + *y2;
            float new_ur_x = *Axx * hw + *Axy * hh + *x2;
            float new_ur_y = *Ayx * hw + *Ayy * hh + *y2;
            float new_lr_x = *Axx * hw + *Axy * (-hh) + *x2;
            float new_lr_y = *Ayx * hw + *Ayy * (-hh) + *y2;

            convergence = (fabsf(dx) < th && fabsf(dy) < th &&
                         fabsf(ul_x - new_ul_x) < th_aff && fabsf(ul_y - new_ul_y) < th_aff &&
                         fabsf(ll_x - new_ll_x) < th_aff && fabsf(ll_y - new_ll_y) < th_aff &&
                         fabsf(ur_x - new_ur_x) < th_aff && fabsf(ur_y - new_ur_y) < th_aff &&
                         fabsf(lr_x - new_lr_x) < th_aff && fabsf(lr_y - new_lr_y) < th_aff);
        }
        
        iteration++;
        
    } while (!convergence && iteration < max_iterations && status == KLT_TRACKED);

    // Final residue check
    if (status == KLT_TRACKED) {
        if (!affine_map) {
            if (lighting_insensitive) {
                gpu_computeIntensityDifferenceLightingInsensitive(
                    img1, img2, x1, y1, *x2, *y2, 
                    width, height, nc1, nr1, imgdiff);
            } else {
                gpu_computeIntensityDifference(
                    img1, img2, nc1, nr1, x1, y1, *x2, *y2, 
                    width, height, imgdiff);
            }
        } else {
            gpu_am_computeIntensityDifferenceAffine(
                img1, img2, x1, y1, *x2, *y2, *Axx, *Ayx, *Axy, *Ayy,
                width, height, nc1, nr1, nc2, nr2, imgdiff);
        }
        
        float residue = gpu_sumAbsFloatWindow(imgdiff, width, height) / (width * height);
        if (residue > max_residue) {
            status = KLT_LARGE_RESIDUE;
        }
    }

    if (status == KLT_SMALL_DET) return KLT_SMALL_DET;
    else if (status == KLT_OOB) return KLT_OOB;
    else if (status == KLT_LARGE_RESIDUE) return KLT_LARGE_RESIDUE;
    else if (iteration >= max_iterations) return KLT_MAX_ITERATIONS;
    else return KLT_TRACKED;
}

// Kernel for affine tracking
__global__ void am_trackFeaturesAffineKernel(
    int num_features,
    const float* x1_arr, const float* y1_arr,
    float* x2_arr, float* y2_arr,
    const float* img1, int nc1, int nr1,
    const float* gradx1, const float* grady1,
    const float* img2, int nc2, int nr2,
    const float* gradx2, const float* grady2,
    int width, int height,
    float step_factor,
    int max_iterations,
    float small,
    float th, float th_aff,
    float max_residue,
    int lighting_insensitive,
    int affine_map,
    float mdd,
    float* Axx_arr, float* Ayx_arr,
    float* Axy_arr, float* Ayy_arr,
    int* status_arr,
    float* imgdiff_buf,
    float* gradx_buf,
    float* grady_buf,
    float* T_buf,
    float* a_buf) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_features) return;

    // Get pointers to this thread's buffers
    float* imgdiff = imgdiff_buf + idx * width * height;
    float* gradx = gradx_buf + idx * width * height;
    float* grady = grady_buf + idx * width * height;
    float* T = T_buf + idx * 36;  // 6x6 max
    float* a = a_buf + idx * 6;   // 6x1 max

    int status = gpu_am_trackFeatureAffine(
        x1_arr[idx], y1_arr[idx],
        &x2_arr[idx], &y2_arr[idx],
        img1, nc1, nr1, gradx1, grady1,
        img2, nc2, nr2, gradx2, grady2,
        width, height,
        step_factor, max_iterations,
        small, th, th_aff, max_residue,
        lighting_insensitive, affine_map, mdd,
        &Axx_arr[idx], &Ayx_arr[idx], &Axy_arr[idx], &Ayy_arr[idx],
        imgdiff, gradx, grady, T, a);

    status_arr[idx] = status;
}

/*
 * CONSISTENCY CHECK OF FEATURES BY AFFINE MAPPING (END)
 **********************************************************************/

/*********************************************************************
 * KLTTrackFeatures
 *
 * Tracks feature points from one image to the next.
 */

void KLTTrackFeatures(
    KLT_TrackingContext tc,
    KLT_PixelType* img1,
    KLT_PixelType* img2,
    int ncols,
    int nrows,
    KLT_FeatureList featurelist) {
    
    _KLT_FloatImage tmpimg, floatimg1, floatimg2;
    _KLT_Pyramid pyramid1, pyramid1_gradx, pyramid1_grady;
    _KLT_Pyramid pyramid2, pyramid2_gradx, pyramid2_grady;
    float subsampling = (float)tc->subsampling;
    int i, indx;
    KLT_BOOL floatimg1_created = FALSE;

    int nFeatures = featurelist->nFeatures;

    if (KLT_verbose >= 1) {
        fprintf(stderr, "(KLT) Tracking %d features in a %d by %d image...  ",
            KLTCountRemainingFeatures(featurelist), ncols, nrows);
        fflush(stderr);
    }

    if (tc->window_width % 2 != 1) tc->window_width++;
    if (tc->window_height % 2 != 1) tc->window_height++;
    if (tc->window_width < 3) tc->window_width = 3;
    if (tc->window_height < 3) tc->window_height = 3;

    // --- Temporary float image ---
    tmpimg = _KLTCreateFloatImage(ncols, nrows);

    // --- First image pyramid ---
    if (tc->sequentialMode && tc->pyramid_last != NULL) {
        pyramid1 = (_KLT_Pyramid)tc->pyramid_last;
        pyramid1_gradx = (_KLT_Pyramid)tc->pyramid_last_gradx;
        pyramid1_grady = (_KLT_Pyramid)tc->pyramid_last_grady;
        if (pyramid1->ncols[0] != ncols || pyramid1->nrows[0] != nrows)
            KLTError("(KLTTrackFeatures) Incoming image size differs from previous.\n");
        assert(pyramid1_gradx != NULL && pyramid1_grady != NULL);
    } else {
        floatimg1_created = TRUE;
        floatimg1 = _KLTCreateFloatImage(ncols, nrows);
        _KLTToFloatImage(img1, ncols, nrows, tmpimg);
        _KLTComputeSmoothedImage(tmpimg, _KLTComputeSmoothSigma(tc), floatimg1);

        pyramid1 = _KLTCreatePyramid(ncols, nrows, (int)subsampling, tc->nPyramidLevels);
        _KLTComputePyramid(floatimg1, pyramid1, tc->pyramid_sigma_fact);

        pyramid1_gradx = _KLTCreatePyramid(ncols, nrows, (int)subsampling, tc->nPyramidLevels);
        pyramid1_grady = _KLTCreatePyramid(ncols, nrows, (int)subsampling, tc->nPyramidLevels);
        for (i = 0; i < tc->nPyramidLevels; i++)
            _KLTComputeGradients(pyramid1->img[i], tc->grad_sigma, pyramid1_gradx->img[i], pyramid1_grady->img[i]);
    }

    floatimg2 = _KLTCreateFloatImage(ncols, nrows);
    _KLTToFloatImage(img2, ncols, nrows, tmpimg);
    _KLTComputeSmoothedImage(tmpimg, _KLTComputeSmoothSigma(tc), floatimg2);

    pyramid2 = _KLTCreatePyramid(ncols, nrows, (int)subsampling, tc->nPyramidLevels);
    _KLTComputePyramid(floatimg2, pyramid2, tc->pyramid_sigma_fact);

    pyramid2_gradx = _KLTCreatePyramid(ncols, nrows, (int)subsampling, tc->nPyramidLevels);
    pyramid2_grady = _KLTCreatePyramid(ncols, nrows, (int)subsampling, tc->nPyramidLevels);
    for (i = 0; i < tc->nPyramidLevels; i++)
        _KLTComputeGradients(pyramid2->img[i], tc->grad_sigma, pyramid2_gradx->img[i], pyramid2_grady->img[i]);

    // --- Host arrays for feature data ---
    float* h_x1 = (float*)malloc(nFeatures * sizeof(float));
    float* h_y1 = (float*)malloc(nFeatures * sizeof(float));
    float* h_x2 = (float*)malloc(nFeatures * sizeof(float));
    float* h_y2 = (float*)malloc(nFeatures * sizeof(float));
    int* h_status = (int*)malloc(nFeatures * sizeof(int));

    for (indx = 0; indx < nFeatures; indx++) {
        h_x1[indx] = featurelist->feature[indx]->x;
        h_y1[indx] = featurelist->feature[indx]->y;
        h_x2[indx] = featurelist->feature[indx]->x;
        h_y2[indx] = featurelist->feature[indx]->y;
        h_status[indx] = -1;
    }

    // --- Device memory allocation for feature data ---
    float *d_x1, *d_y1, *d_x2, *d_y2;
    int* d_status;
    cudaMalloc(&d_x1, nFeatures * sizeof(float));
    cudaMalloc(&d_y1, nFeatures * sizeof(float));
    cudaMalloc(&d_x2, nFeatures * sizeof(float));
    cudaMalloc(&d_y2, nFeatures * sizeof(float));
    cudaMalloc(&d_status, nFeatures * sizeof(int));

    // --- Device memory for temporary buffers ---
    int window_size = tc->window_width * tc->window_height;
    float *d_imgdiff, *d_gradx, *d_grady;
    cudaMalloc(&d_imgdiff, nFeatures * window_size * sizeof(float));
    cudaMalloc(&d_gradx, nFeatures * window_size * sizeof(float));
    cudaMalloc(&d_grady, nFeatures * window_size * sizeof(float));

    // --- Copy feature data to device ---
    cudaMemcpy(d_x1, h_x1, nFeatures * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y1, h_y1, nFeatures * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x2, h_x2, nFeatures * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y2, h_y2, nFeatures * sizeof(float), cudaMemcpyHostToDevice);

    // --- CRITICAL FIX: Allocate and copy pyramid images to GPU ---
    _KLT_FloatImage base_img1 = pyramid1->img[0];
    _KLT_FloatImage base_gradx1 = pyramid1_gradx->img[0];
    _KLT_FloatImage base_grady1 = pyramid1_grady->img[0];
    _KLT_FloatImage base_img2 = pyramid2->img[0];
    _KLT_FloatImage base_gradx2 = pyramid2_gradx->img[0];
    _KLT_FloatImage base_grady2 = pyramid2_grady->img[0];

    // Get pyramid level 0 dimensions (NOT original image dimensions)
    int base_cols = pyramid1->ncols[0];
    int base_rows = pyramid1->nrows[0];
    int img_size = base_cols * base_rows * sizeof(float);

    // Allocate GPU memory for pyramid images
    float *d_img1, *d_gradx1, *d_grady1;
    float *d_img2, *d_gradx2, *d_grady2;
    
    cudaMalloc(&d_img1, img_size);
    cudaMalloc(&d_gradx1, img_size);
    cudaMalloc(&d_grady1, img_size);
    cudaMalloc(&d_img2, img_size);
    cudaMalloc(&d_gradx2, img_size);
    cudaMalloc(&d_grady2, img_size);

    // Copy pyramid images to GPU
    cudaMemcpy(d_img1, base_img1->data, img_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_gradx1, base_gradx1->data, img_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_grady1, base_grady1->data, img_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_img2, base_img2->data, img_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_gradx2, base_gradx2->data, img_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_grady2, base_grady2->data, img_size, cudaMemcpyHostToDevice);

    // --- Launch kernel with CORRECT GPU pointers and dimensions ---
    dim3 block(256);
    dim3 grid((nFeatures + block.x - 1) / block.x);
    
    trackFeatureKernel<<<grid, block>>>(
        d_img1, d_gradx1, d_grady1,      // GPU pointers to pyramid images
        d_img2, d_gradx2, d_grady2,      // GPU pointers to pyramid images  
        base_cols, base_rows,            // CORRECT: Use pyramid dimensions, not original
        d_x1, d_y1, d_x2, d_y2,
        nFeatures,
        tc->window_width, tc->window_height,
        tc->step_factor, tc->max_iterations,
        tc->min_determinant, tc->min_displacement,
        tc->max_residue,
        tc->lighting_insensitive,
        d_imgdiff, d_gradx, d_grady,
        d_status);

    cudaDeviceSynchronize();

    // --- Copy results back from device ---
    cudaMemcpy(h_x2, d_x2, nFeatures * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_y2, d_y2, nFeatures * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_status, d_status, nFeatures * sizeof(int), cudaMemcpyDeviceToHost);

    // --- Update feature list with results ---
    for (indx = 0; indx < nFeatures; indx++) {
        int status = h_status[indx];
        if (status != KLT_TRACKED) {
            featurelist->feature[indx]->x = -1.0;
            featurelist->feature[indx]->y = -1.0;
            featurelist->feature[indx]->val = status;
            if (featurelist->feature[indx]->aff_img) 
                _KLTFreeFloatImage(featurelist->feature[indx]->aff_img);
            if (featurelist->feature[indx]->aff_img_gradx) 
                _KLTFreeFloatImage(featurelist->feature[indx]->aff_img_gradx);
            if (featurelist->feature[indx]->aff_img_grady) 
                _KLTFreeFloatImage(featurelist->feature[indx]->aff_img_grady);
            featurelist->feature[indx]->aff_img = NULL;
            featurelist->feature[indx]->aff_img_gradx = NULL;
            featurelist->feature[indx]->aff_img_grady = NULL;
        } else {
            featurelist->feature[indx]->x = h_x2[indx];
            featurelist->feature[indx]->y = h_y2[indx];
            featurelist->feature[indx]->val = KLT_TRACKED;
        }
    }

    // --- Free ALL device memory ---
    cudaFree(d_x1); cudaFree(d_y1); cudaFree(d_x2); cudaFree(d_y2);
    cudaFree(d_status);
    cudaFree(d_imgdiff); cudaFree(d_gradx); cudaFree(d_grady);
    cudaFree(d_img1); cudaFree(d_gradx1); cudaFree(d_grady1);
    cudaFree(d_img2); cudaFree(d_gradx2); cudaFree(d_grady2);

    // --- Free host memory ---
    free(h_x1); free(h_y1); free(h_x2); free(h_y2); free(h_status);

    // --- Cleanup pyramid memory ---
    if (tc->sequentialMode) {
        tc->pyramid_last = pyramid2;
        tc->pyramid_last_gradx = pyramid2_gradx;
        tc->pyramid_last_grady = pyramid2_grady;
    } else {
        _KLTFreePyramid(pyramid2);
        _KLTFreePyramid(pyramid2_gradx);
        _KLTFreePyramid(pyramid2_grady);
    }

    _KLTFreeFloatImage(tmpimg);
    if (floatimg1_created) _KLTFreeFloatImage(floatimg1);
    _KLTFreeFloatImage(floatimg2);
    
    if (!tc->sequentialMode) {
        _KLTFreePyramid(pyramid1);
        _KLTFreePyramid(pyramid1_gradx);
        _KLTFreePyramid(pyramid1_grady);
    }

    if (KLT_verbose >= 1) {
        fprintf(stderr, "\n\t%d features successfully tracked.\n",
            KLTCountRemainingFeatures(featurelist));
        if (tc->writeInternalImages)
            fprintf(stderr, "\tWrote images to 'kltimg_tf*.pgm'.\n");
        fflush(stderr);
    }
}
