/* Standard includes */
#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>

/* Our includes */
#include "base.h"
#include "error.h"
#include "convolveGPU.h"
#include "kltGPU.h"
#include "klt_utilGPU.h"
#include "pyramidGPU.h"

extern int KLT_verbose;

typedef float *_FloatWindow;

__device__ float gpu_interpolate(float x, float y, float *img, int ncols, int nrows)
{
    int xt = (int)x;
    int yt = (int)y;
    float ax = x - xt;
    float ay = y - yt;

    // Handle boundary conditions
    if (xt < 0 || yt < 0 || xt >= ncols - 1 || yt >= nrows - 1)
    {
        return 0.0f;
    }

    int idx = yt * ncols + xt;
    float *ptr = &img[idx];

    return ((1 - ax) * (1 - ay) * ptr[0] + ax * (1 - ay) * ptr[1] + (1 - ax) * ay * ptr[ncols] + ax * ay * ptr[ncols + 1]);
}

__device__ void gpu_compute2by1ErrorVector(
    float *imgdiff, float *gradx, float *grady, int width, int height, float step_factor,
    float *ex, float *ey)
{
    *ex = 0.0f;
    *ey = 0.0f;
    int total_pixels = width * height;

    for (int i = 0; i < total_pixels; i++)
    {
        float diff = imgdiff[i];
        *ex += diff * gradx[i];
        *ey += diff * grady[i];
    }
    *ex *= step_factor;
    *ey *= step_factor;
}

__device__ float gpu_sumAbsFloatWindow(float *fw, int width, int height)
{
    float sum = 0.0f;
    int total_pixels = width * height;

    for (int i = 0; i < total_pixels; i++)
    {
        sum += fabsf(fw[i]);
    }
    return sum;
}

__device__ int gpu_solveEquation(
    float gxx, float gxy, float gyy,
    float ex, float ey, float small,
    float *dx, float *dy)
{

    float det = gxx * gyy - gxy * gxy;

    if (det < small)
        return KLT_SMALL_DET;

    *dx = (gyy * ex - gxy * ey) / det;
    *dy = (gxx * ey - gxy * ex) / det;
    return KLT_TRACKED;
}

__device__ void gpu_computeGradientSum(float *gradx1, float *grady1, float *gradx2, float *grady2,
                                       int ncols1, int nrows1, int ncols2, int nrows2, float x1, float y1, float x2, float y2,
                                       int width, int height, float *gradx, float *grady)
{
    int hw = width / 2, hh = height / 2;

    for (int j = -hh; j <= hh; j++)
    {
        for (int i = -hw; i <= hw; i++)
        {
            float gx1 = gpu_interpolate(x1 + i, y1 + j, gradx1, ncols1, nrows1);
            float gx2 = gpu_interpolate(x2 + i, y2 + j, gradx2, ncols2, nrows2);
            float gy1 = gpu_interpolate(x1 + i, y1 + j, grady1, ncols1, nrows1);
            float gy2 = gpu_interpolate(x2 + i, y2 + j, grady2, ncols2, nrows2);

            int idx = (j + hh) * width + (i + hw);
            gradx[idx] = gx1 + gx2;
            grady[idx] = gy1 + gy2;
        }
    }
}

__device__ void gpu_computeIntensityDifference(float *img1, float *img2, int ncols1, int nrows1, int ncols2, int nrows2,
                                               float x1, float y1, float x2, float y2, int width, int height, float *imgdiff)
{
    int hw = width / 2, hh = height / 2;

    for (int j = -hh; j <= hh; j++)
    {
        for (int i = -hw; i <= hw; i++)
        {
            float g1 = gpu_interpolate(x1 + i, y1 + j, img1, ncols1, nrows1);
            float g2 = gpu_interpolate(x2 + i, y2 + j, img2, ncols2, nrows2);
            int idx = (j + hh) * width + (i + hw);
            imgdiff[idx] = g1 - g2;
        }
    }
}

__device__ void gpu_compute2by2GradientMatrix(float *gradx, float *grady, int width, int height,
                                              float *gxx, float *gxy, float *gyy)
{
    *gxx = 0.0f;
    *gxy = 0.0f;
    *gyy = 0.0f;
    int total_pixels = width * height;

    for (int i = 0; i < total_pixels; i++)
    {
        float gx = gradx[i];
        float gy = grady[i];
        *gxx += gx * gx;
        *gxy += gx * gy;
        *gyy += gy * gy;
    }
}

__device__ int gpu_trackFeatureSingleLevel(
    float x1, float y1, float *x2, float *y2,
    float *img1, float *gradx1, float *grady1,
    float *img2, float *gradx2, float *grady2,
    int ncols, int nrows,
    int width, int height, float step_factor, int max_iterations,
    float small, float th, float max_residue, int lighting_insensitive,
    float *imgdiff, float *gradx, float *grady)
{

    float gxx, gxy, gyy, ex, ey, dx, dy;
    int iteration = 0;
    int status = KLT_TRACKED;
    int hw = width / 2;
    int hh = height / 2;
    float one_plus_eps = 1.001f;

    do
    {
        if (x1 - hw < 0.0f || ncols - (x1 + hw) < one_plus_eps ||
            *x2 - hw < 0.0f || ncols - (*x2 + hw) < one_plus_eps ||
            y1 - hh < 0.0f || nrows - (y1 + hh) < one_plus_eps ||
            *y2 - hh < 0.0f || nrows - (*y2 + hh) < one_plus_eps)
        {
            status = KLT_OOB;
            break;
        }

        // Compute windows
        gpu_computeIntensityDifference(img1, img2, ncols, nrows, ncols, nrows,
                                       x1, y1, *x2, *y2, width, height, imgdiff);
        gpu_computeGradientSum(gradx1, grady1, gradx2, grady2,
                               ncols, nrows, ncols, nrows,
                               x1, y1, *x2, *y2, width, height, gradx, grady);

        // Compute matrices
        gpu_compute2by2GradientMatrix(gradx, grady, width, height, &gxx, &gxy, &gyy);
        gpu_compute2by1ErrorVector(imgdiff, gradx, grady, width, height,
                                   step_factor, &ex, &ey);

        // Solve equation
        status = gpu_solveEquation(gxx, gxy, gyy, ex, ey, small, &dx, &dy);
        if (status == KLT_SMALL_DET)
            break;

        *x2 += dx;
        *y2 += dy;
        iteration++;

    } while ((fabsf(dx) >= th || fabsf(dy) >= th) && iteration < max_iterations);

    // Check if window is out of bounds after iterations
    if (*x2 - hw < 0.0f || ncols - (*x2 + hw) < one_plus_eps || *y2 - hh < 0.0f || nrows - (*y2 + hh) < one_plus_eps)
        status = KLT_OOB;

    // Check residue
    if (status == KLT_TRACKED)
    {
        gpu_computeIntensityDifference(img1, img2, ncols, nrows, ncols, nrows, x1, y1, *x2, *y2, width, height, imgdiff);
        float residue = gpu_sumAbsFloatWindow(imgdiff, width, height) / (width * height);
        if (residue > max_residue)
            status = KLT_LARGE_RESIDUE;
    }

    if (status == KLT_SMALL_DET)
        return KLT_SMALL_DET;
    else if (status == KLT_OOB)
        return KLT_OOB;
    else if (status == KLT_LARGE_RESIDUE)
        return KLT_LARGE_RESIDUE;
    else if (iteration >= max_iterations)
        return KLT_MAX_ITERATIONS;
    else
        return KLT_TRACKED;
}

__global__ void trackFeaturesKernel(
    float **pyramid1_img, float **pyramid1_gradx, float **pyramid1_grady,
    float **pyramid2_img, float **pyramid2_gradx, float **pyramid2_grady,
    int *pyramid_ncols, int *pyramid_nrows,
    float *feature_x, float *feature_y, int *feature_val,
    int window_width, int window_height,
    float step_factor, int max_iterations,
    float min_determinant, float min_displacement,
    float max_residue, int lighting_insensitive,
    float subsampling, int nPyramidLevels,
    int ncols, int nrows, int borderx, int bordery)
{

    int feature_idx = blockIdx.x;
    if (feature_idx >= gridDim.x)
        return;

    // Only track features that are not lost
    if (feature_val[feature_idx] < 0)
        return;

    // Use shared memory for temporary windows
    extern __shared__ float shared_mem[];
    int window_size = window_width * window_height;
    float *imgdiff = shared_mem;
    float *gradx = &shared_mem[window_size];
    float *grady = &shared_mem[2 * window_size];

    float xloc = feature_x[feature_idx];
    float yloc = feature_y[feature_idx];
    float xlocout = xloc;
    float ylocout = yloc;

    int val = KLT_TRACKED;

    // Transform location to coarsest resolution
    for (int r = nPyramidLevels - 1; r >= 0; r--)
    {
        xloc /= subsampling;
        yloc /= subsampling;
    }
    xlocout = xloc;
    ylocout = yloc;

    // Track through pyramid levels from coarsest to finest
    for (int r = nPyramidLevels - 1; r >= 0; r--)
    {
        // Track feature at current resolution
        xloc *= subsampling;
        yloc *= subsampling;
        xlocout *= subsampling;
        ylocout *= subsampling;

        float *img1 = pyramid1_img[r];
        float *gradx1 = pyramid1_gradx[r];
        float *grady1 = pyramid1_grady[r];
        float *img2 = pyramid2_img[r];
        float *gradx2 = pyramid2_gradx[r];
        float *grady2 = pyramid2_grady[r];
        int level_ncols = pyramid_ncols[r];
        int level_nrows = pyramid_nrows[r];

        val = gpu_trackFeatureSingleLevel(
            xloc, yloc, &xlocout, &ylocout,
            img1, gradx1, grady1,
            img2, gradx2, grady2,
            level_ncols, level_nrows,
            window_width, window_height,
            step_factor, max_iterations,
            min_determinant, min_displacement,
            max_residue, lighting_insensitive,
            imgdiff, gradx, grady);

        if (val == KLT_SMALL_DET || val == KLT_OOB)
            break;
    }

    // Check out of bounds with border
    if (xlocout < borderx || xlocout > ncols - 1 - borderx || ylocout < bordery || ylocout > nrows - 1 - bordery)
    {
        val = KLT_OOB;
    }

    // Update feature
    if (val != KLT_TRACKED)
    {
        feature_x[feature_idx] = -1.0f;
        feature_y[feature_idx] = -1.0f;
        feature_val[feature_idx] = val;
    }
    else
    {
        feature_x[feature_idx] = xlocout;
        feature_y[feature_idx] = ylocout;
        feature_val[feature_idx] = KLT_TRACKED;
    }
}

void KLTTrackFeatures(
    KLT_TrackingContext tc,
    KLT_PixelType *img1,
    KLT_PixelType *img2,
    int ncols,
    int nrows,
    KLT_FeatureList featurelist)
{

    // GPU pyramid processing
    _KLT_FloatImage tmpimg, floatimg1, floatimg2;
    _KLT_Pyramid pyramid1, pyramid1_gradx, pyramid1_grady;
    _KLT_Pyramid pyramid2, pyramid2_gradx, pyramid2_grady;
    float subsampling = (float)tc->subsampling;
    KLT_BOOL floatimg1_created = FALSE;
    int i;

    /* if (KLT_verbose >= 1)
     {
         fprintf(stderr, "(KLT) Tracking %d features in a %d by %d image...  ",
                 KLTCountRemainingFeatures(featurelist), ncols, nrows);
         fflush(stderr);
     }*/

    // Create temporary image on GPU
    tmpimg = _KLTCreateFloatImageGPU(ncols, nrows);

    // Process first image pyramid on GPU
    if (tc->sequentialMode && tc->pyramid_last != NULL)
    {
        pyramid1 = (_KLT_Pyramid)tc->pyramid_last;
        pyramid1_gradx = (_KLT_Pyramid)tc->pyramid_last_gradx;
        pyramid1_grady = (_KLT_Pyramid)tc->pyramid_last_grady;
    }
    else
    {
        floatimg1_created = TRUE;
        floatimg1 = _KLTCreateFloatImageGPU(ncols, nrows);
        _KLTToFloatImageGPU(img1, ncols, nrows, tmpimg);
        _KLTComputeSmoothedImageGPU(tmpimg, _KLTComputeSmoothSigma(tc), floatimg1);
        pyramid1 = _KLTCreatePyramidGPU(ncols, nrows, (int)subsampling, tc->nPyramidLevels);
        _KLTComputePyramidGPU(floatimg1, pyramid1, tc->pyramid_sigma_fact);
        pyramid1_gradx = _KLTCreatePyramidGPU(ncols, nrows, (int)subsampling, tc->nPyramidLevels);
        pyramid1_grady = _KLTCreatePyramidGPU(ncols, nrows, (int)subsampling, tc->nPyramidLevels);
        for (i = 0; i < tc->nPyramidLevels; i++)
            _KLTComputeGradientsGPU(pyramid1->img[i], tc->grad_sigma,
                                    pyramid1_gradx->img[i], pyramid1_grady->img[i]);
    }

    floatimg2 = _KLTCreateFloatImageGPU(ncols, nrows);
    _KLTToFloatImageGPU(img2, ncols, nrows, tmpimg);
    _KLTComputeSmoothedImageGPU(tmpimg, _KLTComputeSmoothSigma(tc), floatimg2);
    pyramid2 = _KLTCreatePyramidGPU(ncols, nrows, (int)subsampling, tc->nPyramidLevels);
    _KLTComputePyramidGPU(floatimg2, pyramid2, tc->pyramid_sigma_fact);
    pyramid2_gradx = _KLTCreatePyramidGPU(ncols, nrows, (int)subsampling, tc->nPyramidLevels);
    pyramid2_grady = _KLTCreatePyramidGPU(ncols, nrows, (int)subsampling, tc->nPyramidLevels);
    for (i = 0; i < tc->nPyramidLevels; i++)
        _KLTComputeGradientsGPU(pyramid2->img[i], tc->grad_sigma,
                                pyramid2_gradx->img[i], pyramid2_grady->img[i]);

    int nFeatures = featurelist->nFeatures;

    // Allocate device memory for features
    float *d_feature_x, *d_feature_y;
    int *d_feature_val;
    cudaMalloc(&d_feature_x, nFeatures * sizeof(float));
    cudaMalloc(&d_feature_y, nFeatures * sizeof(float));
    cudaMalloc(&d_feature_val, nFeatures * sizeof(int));

    // Copy feature data to device
    float *h_feature_x = (float *)malloc(nFeatures * sizeof(float));
    float *h_feature_y = (float *)malloc(nFeatures * sizeof(float));
    int *h_feature_val = (int *)malloc(nFeatures * sizeof(int));

    for (int indx = 0; indx < nFeatures; indx++)
    {
        h_feature_x[indx] = featurelist->feature[indx]->x;
        h_feature_y[indx] = featurelist->feature[indx]->y;
        h_feature_val[indx] = featurelist->feature[indx]->val;
    }

    cudaMemcpy(d_feature_x, h_feature_x, nFeatures * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_feature_y, h_feature_y, nFeatures * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_feature_val, h_feature_val, nFeatures * sizeof(int), cudaMemcpyHostToDevice);

    // Allocate device memory for pyramid data pointers
    float **d_pyramid1_img, **d_pyramid1_gradx, **d_pyramid1_grady;
    float **d_pyramid2_img, **d_pyramid2_gradx, **d_pyramid2_grady;
    cudaMalloc(&d_pyramid1_img, tc->nPyramidLevels * sizeof(float *));
    cudaMalloc(&d_pyramid1_gradx, tc->nPyramidLevels * sizeof(float *));
    cudaMalloc(&d_pyramid1_grady, tc->nPyramidLevels * sizeof(float *));
    cudaMalloc(&d_pyramid2_img, tc->nPyramidLevels * sizeof(float *));
    cudaMalloc(&d_pyramid2_gradx, tc->nPyramidLevels * sizeof(float *));
    cudaMalloc(&d_pyramid2_grady, tc->nPyramidLevels * sizeof(float *));

    // Allocate device memory for pyramid dimensions
    int *d_pyramid_ncols, *d_pyramid_nrows;
    cudaMalloc(&d_pyramid_ncols, tc->nPyramidLevels * sizeof(int));
    cudaMalloc(&d_pyramid_nrows, tc->nPyramidLevels * sizeof(int));

    // Copy pyramid dimensions to device
    int *h_pyramid_ncols = (int *)malloc(tc->nPyramidLevels * sizeof(int));
    int *h_pyramid_nrows = (int *)malloc(tc->nPyramidLevels * sizeof(int));
    for (i = 0; i < tc->nPyramidLevels; i++)
    {
        h_pyramid_ncols[i] = pyramid1->ncols[i];
        h_pyramid_nrows[i] = pyramid1->nrows[i];
    }
    cudaMemcpy(d_pyramid_ncols, h_pyramid_ncols, tc->nPyramidLevels * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pyramid_nrows, h_pyramid_nrows, tc->nPyramidLevels * sizeof(int), cudaMemcpyHostToDevice);

    // Prepare pyramid pointer arrays
    float **h_pyramid1_img_ptrs = (float **)malloc(tc->nPyramidLevels * sizeof(float *));
    float **h_pyramid1_gradx_ptrs = (float **)malloc(tc->nPyramidLevels * sizeof(float *));
    float **h_pyramid1_grady_ptrs = (float **)malloc(tc->nPyramidLevels * sizeof(float *));
    float **h_pyramid2_img_ptrs = (float **)malloc(tc->nPyramidLevels * sizeof(float *));
    float **h_pyramid2_gradx_ptrs = (float **)malloc(tc->nPyramidLevels * sizeof(float *));
    float **h_pyramid2_grady_ptrs = (float **)malloc(tc->nPyramidLevels * sizeof(float *));

    for (i = 0; i < tc->nPyramidLevels; i++)
    {
        // Use existing GPU pointers directly
        h_pyramid1_img_ptrs[i] = pyramid1->img[i]->data;
        h_pyramid1_gradx_ptrs[i] = pyramid1_gradx->img[i]->data;
        h_pyramid1_grady_ptrs[i] = pyramid1_grady->img[i]->data;
        h_pyramid2_img_ptrs[i] = pyramid2->img[i]->data;
        h_pyramid2_gradx_ptrs[i] = pyramid2_gradx->img[i]->data;
        h_pyramid2_grady_ptrs[i] = pyramid2_grady->img[i]->data;
    }

    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

    // Copy pyramid1 data in stream1
    cudaMemcpyAsync(d_pyramid1_img, h_pyramid1_img_ptrs, tc->nPyramidLevels * sizeof(float *), cudaMemcpyHostToDevice, stream1);
    cudaMemcpyAsync(d_pyramid1_gradx, h_pyramid1_gradx_ptrs, tc->nPyramidLevels * sizeof(float *), cudaMemcpyHostToDevice, stream1);
    cudaMemcpyAsync(d_pyramid1_grady, h_pyramid1_grady_ptrs, tc->nPyramidLevels * sizeof(float *), cudaMemcpyHostToDevice, stream1);

    // Copy pyramid2 data in stream2
    cudaMemcpyAsync(d_pyramid2_img, h_pyramid2_img_ptrs, tc->nPyramidLevels * sizeof(float *), cudaMemcpyHostToDevice, stream2);
    cudaMemcpyAsync(d_pyramid2_gradx, h_pyramid2_gradx_ptrs, tc->nPyramidLevels * sizeof(float *), cudaMemcpyHostToDevice, stream2);
    cudaMemcpyAsync(d_pyramid2_grady, h_pyramid2_grady_ptrs, tc->nPyramidLevels * sizeof(float *), cudaMemcpyHostToDevice, stream2);
    // Calculate shared memory size
    int window_size = tc->window_width * tc->window_height;
    size_t shared_mem_size = 3 * window_size * sizeof(float);

    // Wait for both streams to complete before launching kernel
    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);

    trackFeaturesKernel<<<nFeatures, 1, shared_mem_size>>>(
        d_pyramid1_img, d_pyramid1_gradx, d_pyramid1_grady,
        d_pyramid2_img, d_pyramid2_gradx, d_pyramid2_grady,
        d_pyramid_ncols, d_pyramid_nrows,
        d_feature_x, d_feature_y, d_feature_val,
        tc->window_width, tc->window_height,
        tc->step_factor, tc->max_iterations,
        tc->min_determinant, tc->min_displacement,
        tc->max_residue, tc->lighting_insensitive,
        subsampling, tc->nPyramidLevels,
        ncols, nrows, tc->borderx, tc->bordery);

    cudaDeviceSynchronize();

    // Copy results back to host
    cudaMemcpy(h_feature_x, d_feature_x, nFeatures * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_feature_y, d_feature_y, nFeatures * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_feature_val, d_feature_val, nFeatures * sizeof(int), cudaMemcpyDeviceToHost);

    // Update feature list with results
    for (int indx = 0; indx < nFeatures; indx++)
    {
        featurelist->feature[indx]->x = h_feature_x[indx];
        featurelist->feature[indx]->y = h_feature_y[indx];
        featurelist->feature[indx]->val = h_feature_val[indx];
    }

    // Free device memory
    cudaFree(d_feature_x);
    cudaFree(d_feature_y);
    cudaFree(d_feature_val);
    cudaFree(d_pyramid_ncols);
    cudaFree(d_pyramid_nrows);
    cudaFree(d_pyramid1_img);
    cudaFree(d_pyramid1_gradx);
    cudaFree(d_pyramid1_grady);
    cudaFree(d_pyramid2_img);
    cudaFree(d_pyramid2_gradx);
    cudaFree(d_pyramid2_grady);

    // Free host memory
    free(h_pyramid1_img_ptrs);
    free(h_pyramid1_gradx_ptrs);
    free(h_pyramid1_grady_ptrs);
    free(h_pyramid2_img_ptrs);
    free(h_pyramid2_gradx_ptrs);
    free(h_pyramid2_grady_ptrs);
    free(h_feature_x);
    free(h_feature_y);
    free(h_feature_val);
    free(h_pyramid_ncols);
    free(h_pyramid_nrows);

    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    // Cleanup GPU resources
    _KLTFreeFloatImageGPU(tmpimg);
    if (floatimg1_created)
    {
        _KLTFreeFloatImageGPU(floatimg1);
        if (!tc->sequentialMode)
        {
            _KLTFreePyramidGPU(pyramid1);
            _KLTFreePyramidGPU(pyramid1_gradx);
            _KLTFreePyramidGPU(pyramid1_grady);
        }
    }
    _KLTFreeFloatImageGPU(floatimg2);

    if (tc->sequentialMode)
    {
        tc->pyramid_last = pyramid2;
        tc->pyramid_last_gradx = pyramid2_gradx;
        tc->pyramid_last_grady = pyramid2_grady;
    }
    else
    {
        _KLTFreePyramidGPU(pyramid2);
        _KLTFreePyramidGPU(pyramid2_gradx);
        _KLTFreePyramidGPU(pyramid2_grady);
    }

    /*if (KLT_verbose >= 1)
    {
        fprintf(stderr, "\n\t%d features successfully tracked.\n",
                KLTCountRemainingFeatures(featurelist));
        fflush(stderr);
    }*/
}
