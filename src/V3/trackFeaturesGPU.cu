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
#include "cuda_utils.h"

extern int KLT_verbose;

typedef float *_FloatWindow;

typedef struct {
    float x1, y1, x2, y2;
    int status;
} FeatureData;

typedef struct{
    float* base_ptr;
    float* img;
    float* gradx;
    float* grady;
} PyramidData;

typedef struct{
    PyramidData pyramid1;
    PyramidData pyramid2;
} PyramidDataCombined;

__device__ float gpu_interpolate(float x, float y, const float* img, int cols, int rows) {
    int xt = (int)x;
    int yt = (int)y;

    if (xt < 0 || yt < 0 || xt >= cols - 1 ||yt >= rows - 1)
        return 0.0f;
    
    float ax = x - xt;
    float ay = y - yt;
    
    const float* ptr = img + (cols * yt) + xt;
    
    return (1.0f - ax) * (1.0f - ay) * ptr[0] +
           ax * (1.0f - ay) * ptr[1] +
           (1.0f - ax) * ay * ptr[cols] +
           ax * ay * ptr[cols + 1];
}

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
            float g1 = gpu_interpolate(x1 + i, y1 + j, img1, cols, rows);
            float g2 = gpu_interpolate(x2 + i, y2 + j, img2, cols, rows);
            imgDiff[idx] = g1 - g2;
        }
    }
}

// Compute gradient sum (standard version)
__device__ void gpu_computeGradientSum(
    const float* gradx1, const float* grady1, 
    const float* gradx2, const float* grady2,
    int cols, int rows,
    float x1, float y1, float x2, float y2, 
    int width, int height,
    float* out_gradx, float* out_grady) {
    
    int w = width / 2;
    int h = height / 2;

    for (int j = -h; j <= h; j++) {
        for (int i = -w; i <= w; i++) {
            int idx = (j + h) * width + (i + w);
            
            float g1x = gpu_interpolate(x1 + i, y1 + j, gradx1, cols, rows);
            float g2x = gpu_interpolate(x2 + i, y2 + j, gradx2, cols, rows);
            out_gradx[idx] = g1x + g2x;
            
            float g1y = gpu_interpolate(x1 + i, y1 + j, grady1, cols, rows);
            float g2y = gpu_interpolate(x2 + i, y2 + j, grady2, cols, rows);
            out_grady[idx] = g1y + g2y;
        }
    }
}

// Lighting insensitive intensity difference
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

    //compute values
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
    float alpha = sqrtf(mean1_sq / (mean2_sq + 1e-6f));

    float mean1 = sum1 / total;
    float mean2 = sum2 / total;
    float beta = mean1 - alpha * mean2;

    int idx = 0;
    for (int j = -h; j <= h; j++) {
        for (int i = -w; i <= w; i++) {
            float g1 = gpu_interpolate(x1 + i, y1 + j, img1, cols, rows);
            float g2 = gpu_interpolate(x2 + i, y2 + j, img2, cols, rows);
            imgDiff[idx++] = g1 - (alpha * g2 + beta);
        }
    }
}

// Lighting insensitive gradient sum
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

    // Compute intensity statistics for normalization
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
    float alpha =  (float) sqrtf(mean1 / (mean2 + 1e-6f));

    // Compute values
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

// Compute 2x2 gradient matrix
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

    *ex = sum_ex * step_factor;
    *ey = sum_ey * step_factor;
}

// Solve the 2x2 system
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

// Sum absolute values in window
__device__ float gpu_sumAbsFloatWindow(const float* fw, int width, int height) {
    float sum = 0.0f;
    int total = width * height;

    for (int i = 0; i < total; i++)
        sum += fabsf(fw[i]);
    return sum;
}

// Out of bounds check
__device__ int gpu_outOfBounds(float x, float y, int ncols, int nrows, int borderx, int bordery) {
    return (x < borderx || x > ncols - 1 - borderx || 
            y < bordery || y > nrows - 1 - bordery);
}

__device__ int gpu_trackFeatureSingleLevel(
    float x1, float y1,                    
    float* x2, float* y2,                  
    const float* img1, const float* gradx1, const float* grady1,
    const float* img2, const float* gradx2, const float* grady2,
    int nc, int nr,                        
    int width, int height,                 
    float step_factor,
    int max_iterations,
    float small,
    float th,
    float max_residue,
    int lighting_insensitive,
    float* imgdiff,                        
    float* gradx,
    float* grady) {
    
    float gxx, gxy, gyy, ex, ey, dx, dy;
    int iteration = 0;
    int status = KLT_TRACKED;
    int hw = width / 2;
    int hh = height / 2;
    float one_plus_eps = 1.001f;

    do {
        // Check bounds
        if (x1 - hw < 0.0f || nc - (x1 + hw) < one_plus_eps ||
            *x2 - hw < 0.0f || nc - (*x2 + hw) < one_plus_eps ||
            y1 - hh < 0.0f || nr - (y1 + hh) < one_plus_eps ||
            *y2 - hh < 0.0f || nr - (*y2 + hh) < one_plus_eps) {
            status = KLT_OOB;
            break;
        }

        // Compute windows
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
                nc, nr, x1, y1, *x2, *y2, 
                width, height, gradx, grady);
        }

        // Compute matrices and solve
        gpu_compute2by2GradientMatrix(gradx, grady, width, height, &gxx, &gxy, &gyy);
        gpu_compute2by1ErrorVector(imgdiff, gradx, grady, width, height, step_factor, &ex, &ey);

        status = gpu_solveEquation(gxx, gxy, gyy, ex, ey, small, &dx, &dy);
        if (status == KLT_SMALL_DET) break;

        *x2 += dx;
        *y2 += dy;
        iteration++;

    } while ((fabsf(dx) >= th || fabsf(dy) >= th) && iteration < max_iterations);

    if (*x2-hw < 0.0f || nc-(*x2+hw) < one_plus_eps || 
      *y2-hh < 0.0f || nr-(*y2+hh) < one_plus_eps)
    status = KLT_OOB;

    // Final residue check
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
    if (status == KLT_SMALL_DET)  return KLT_SMALL_DET;
    else if (status == KLT_OOB)  return KLT_OOB;
    else if (status == KLT_LARGE_RESIDUE)  return KLT_LARGE_RESIDUE;
    else if (iteration >= max_iterations)  return KLT_MAX_ITERATIONS;
    else  return KLT_TRACKED;
}

__global__ void trackFeatureKernel(
    const PyramidData pyramid1,
    const PyramidData pyramid2,
    const int* pyramid_dims,           
    const int* pyramid_offsets,
    FeatureData* feature_data,        
    int num_features,
    int width, int height,
    float step_factor,
    int max_iterations,
    float small,
    float th,
    float max_residue,
    int lighting_insensitive,
    float subsampling,
    int nPyramidLevels,
    float* imgdiff_buf,
    float* gradx_buf,
    float* grady_buf) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_features) return;

    // Get workspace for this feature
    int window_size = width * height;
    float* imgdiff = imgdiff_buf + idx * window_size;
    float* gradx = gradx_buf + idx * window_size;
    float* grady = grady_buf + idx * window_size;

    // Get feature data
    FeatureData* feat = &feature_data[idx];
    float x1 = feat->x1;
    float y1 = feat->y1;
    float x2 = feat->x2;
    float y2 = feat->y2;
    
    int status = KLT_TRACKED;

    for (int r = 0; r < nPyramidLevels - 1; r++) {
        x1 /= subsampling;
        y1 /= subsampling;
        x2 /= subsampling;
        y2 /= subsampling;
    }

    // Track through pyramid levels from coarsest to finest
    for (int r = nPyramidLevels - 1; r >= 0; r--) {
        if (status != KLT_TRACKED) break;
        
        // Get current level information
        int level_offset = pyramid_offsets[r];
        int nc = pyramid_dims[r * 2];
        int nr = pyramid_dims[r * 2 + 1];
        
        const float* img1_level = pyramid1.img + level_offset;
        const float* gradx1_level = pyramid1.gradx + level_offset;
        const float* grady1_level = pyramid1.grady + level_offset;
        const float* img2_level = pyramid2.img + level_offset;
        const float* gradx2_level = pyramid2.gradx + level_offset;
        const float* grady2_level = pyramid2.grady + level_offset;

        // Track at current level
        status = gpu_trackFeatureSingleLevel(
            x1, y1, &x2, &y2,
            img1_level, gradx1_level, grady1_level,
            img2_level, gradx2_level, grady2_level,
            nc, nr, width, height,
            step_factor, max_iterations,
            small, th, max_residue,
            lighting_insensitive,
            imgdiff, gradx, grady);

        // Scale coordinates for next finer level
        if (r > 0) {
            x1 *= subsampling;
            y1 *= subsampling;
            x2 *= subsampling;
            y2 *= subsampling;
        }
    }

    // Update results
    feat->x2 = x2;
    feat->y2 = y2;
    feat->status = status;
}

void allocatePyramidonGPU(PyramidData* pd, int total_size){
    size_t total_bytes = total_size * sizeof(float) * 3;
    cudaMalloc(&pd->base_ptr, total_bytes);

    pd->img = pd->base_ptr;
    pd->gradx = pd->img + total_size;
    pd->grady = pd->gradx + total_size;
}

void allocatePyramidDataCombined(PyramidDataCombined* data, int total_size){
    allocatePyramidonGPU(&data->pyramid1, total_size);
    allocatePyramidonGPU(&data->pyramid2, total_size);
}

void freePyramidData(PyramidData* pd){
    if (pd->img != NULL) {
        cudaFree(pd->base_ptr);  // Free the base pointer
        pd->base_ptr = NULL;
        pd->img = NULL;
        pd->gradx = NULL;
        pd->grady = NULL;
    }
}

// free combined data
void freePyramidDataCombined(PyramidDataCombined* data){
    freePyramidData(&data->pyramid1);
    freePyramidData(&data->pyramid2);
}

float* transferPyramidData(PyramidData* d_pyramid, 
                        _KLT_Pyramid img_pyramid,
                        _KLT_Pyramid gradx_pyramid, 
                        _KLT_Pyramid grady_pyramid,
                        int* pyramid_offsets, 
                        int nLevels, 
                        int total_size,
                        cudaStream_t stream) {
    
    // Allocate pinned host memory for efficient transfer
    float* h_buffer;
    cudaMallocHost(&h_buffer, total_size * 3 * sizeof(float));
    
    // Set up host pointers in SOA layout
    float* h_img = h_buffer;
    float* h_gradx = h_img + total_size;
    float* h_grady = h_gradx + total_size;
    
    // Copy pyramid data to host buffer in SOA layout
    for (int i = 0; i < nLevels; i++) {
        int level_size = img_pyramid->ncols[i] * img_pyramid->nrows[i];
        int offset = pyramid_offsets[i];
        
        memcpy(h_img + offset, img_pyramid->img[i]->data, level_size * sizeof(float));
        memcpy(h_gradx + offset, gradx_pyramid->img[i]->data, level_size * sizeof(float));
        memcpy(h_grady + offset, grady_pyramid->img[i]->data, level_size * sizeof(float));
    }
    
    // Single asynchronous transfer
    cudaMemcpyAsync(d_pyramid->base_ptr, h_buffer, total_size * 3 * sizeof(float),
                   cudaMemcpyHostToDevice, stream);
    
    //cudaStreamSynchronize(stream);
    return h_buffer;
}

void transferPyramidDataCombined(PyramidDataCombined* d_data,
                                _KLT_Pyramid img1_pyramid, _KLT_Pyramid gradx1_pyramid, _KLT_Pyramid grady1_pyramid,
                                _KLT_Pyramid img2_pyramid, _KLT_Pyramid gradx2_pyramid, _KLT_Pyramid grady2_pyramid,
                                int* pyramid_offsets, int nLevels, int total_size,
                                cudaStream_t stream1, cudaStream_t stream2) {
    
    // Transfer pyramid1 in stream1
    float* h_buffer1=transferPyramidData(&d_data->pyramid1, img1_pyramid, gradx1_pyramid, grady1_pyramid,
                       pyramid_offsets, nLevels, total_size, stream1);
    // Transfer pyramid2 in stream2
    float* h_buffer2 = transferPyramidData(&d_data->pyramid2, img2_pyramid, gradx2_pyramid, grady2_pyramid,
                       pyramid_offsets, nLevels, total_size, stream2);
    
    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);
    cudaFreeHost(h_buffer1);
    cudaFreeHost(h_buffer2);
}

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

    // Create CUDA streams for overlapping operations
    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

    tmpimg = _KLTCreateFloatImage(ncols, nrows);

    if (tc->sequentialMode && tc->pyramid_last != NULL) {
        pyramid1 = (_KLT_Pyramid)tc->pyramid_last;
        pyramid1_gradx = (_KLT_Pyramid)tc->pyramid_last_gradx;
        pyramid1_grady = (_KLT_Pyramid)tc->pyramid_last_grady;
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

    int total_pyramid_size = 0;
    int* h_pyramid_dims = (int*)malloc(tc->nPyramidLevels * 2 * sizeof(int));
    int* h_pyramid_offsets = (int*)malloc(tc->nPyramidLevels * sizeof(int));
    
    for (i = 0; i < tc->nPyramidLevels; i++) {
        h_pyramid_dims[i * 2] = pyramid1->ncols[i];
        h_pyramid_dims[i * 2 + 1] = pyramid1->nrows[i];
        h_pyramid_offsets[i] = total_pyramid_size;
        total_pyramid_size += pyramid1->ncols[i] * pyramid1->nrows[i];
    }

    int nFeatures = featurelist->nFeatures;
    int window_size = tc->window_width * tc->window_height;

    PyramidDataCombined pyramid_data;
    allocatePyramidDataCombined(&pyramid_data, total_pyramid_size);

    // Feature data
    FeatureData* h_feature_data;
    cudaMallocHost(&h_feature_data, nFeatures * sizeof(FeatureData));
    
    for (indx = 0; indx < nFeatures; indx++) {
        h_feature_data[indx].x1 = featurelist->feature[indx]->x;
        h_feature_data[indx].y1 = featurelist->feature[indx]->y;
        h_feature_data[indx].x2 = featurelist->feature[indx]->x;
        h_feature_data[indx].y2 = featurelist->feature[indx]->y;
        h_feature_data[indx].status = -1;
    }

    FeatureData* d_feature_data;
    cudaMalloc(&d_feature_data, nFeatures * sizeof(FeatureData));

    // Workspace buffers
    float *d_imgdiff, *d_gradx, *d_grady;
    cudaMalloc(&d_imgdiff, nFeatures * window_size * sizeof(float));
    cudaMalloc(&d_gradx, nFeatures * window_size * sizeof(float));
    cudaMalloc(&d_grady, nFeatures * window_size * sizeof(float));

    // Pyramid metadata
    int* d_pyramid_dims, *d_pyramid_offsets;
    cudaMalloc(&d_pyramid_dims, tc->nPyramidLevels * 2 * sizeof(int));
    cudaMalloc(&d_pyramid_offsets, tc->nPyramidLevels * sizeof(int));

    // Copy common data first
    cudaMemcpyAsync(d_feature_data, h_feature_data, nFeatures * sizeof(FeatureData), cudaMemcpyHostToDevice, stream1);
    cudaMemcpyAsync(d_pyramid_dims, h_pyramid_dims, tc->nPyramidLevels * 2 * sizeof(int), cudaMemcpyHostToDevice, stream1);
    cudaMemcpyAsync(d_pyramid_offsets, h_pyramid_offsets, tc->nPyramidLevels * sizeof(int), cudaMemcpyHostToDevice, stream1);

    // Start Pyramid 1 transfer
    transferPyramidData(&pyramid_data.pyramid1, 
                       pyramid1, pyramid1_gradx, pyramid1_grady,
                       h_pyramid_offsets, tc->nPyramidLevels, total_pyramid_size, stream1);

    floatimg2 = _KLTCreateFloatImage(ncols, nrows);
    _KLTToFloatImage(img2, ncols, nrows, tmpimg);
    _KLTComputeSmoothedImage(tmpimg, _KLTComputeSmoothSigma(tc), floatimg2);
    pyramid2 = _KLTCreatePyramid(ncols, nrows, (int)subsampling, tc->nPyramidLevels);
    _KLTComputePyramid(floatimg2, pyramid2, tc->pyramid_sigma_fact);
    pyramid2_gradx = _KLTCreatePyramid(ncols, nrows, (int)subsampling, tc->nPyramidLevels);
    pyramid2_grady = _KLTCreatePyramid(ncols, nrows, (int)subsampling, tc->nPyramidLevels);
    for (i = 0; i < tc->nPyramidLevels; i++)
        _KLTComputeGradients(pyramid2->img[i], tc->grad_sigma, pyramid2_gradx->img[i], pyramid2_grady->img[i]);

    transferPyramidData(&pyramid_data.pyramid2, 
                       pyramid2, pyramid2_gradx, pyramid2_grady,
                       h_pyramid_offsets, tc->nPyramidLevels, total_pyramid_size, stream2);

    size_t workspace_size = nFeatures * window_size * sizeof(float);
    cudaMemsetAsync(d_imgdiff, 0, workspace_size, stream1);
    cudaMemsetAsync(d_gradx, 0, workspace_size, stream1);
    cudaMemsetAsync(d_grady, 0, workspace_size, stream1);

    // Wait for all preparations to complete
    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);

    int block_size = 256;
    dim3 block(block_size);
    dim3 grid((nFeatures + block.x - 1) / block.x);

    if (KLT_verbose >= 1) {
        fprintf(stderr, "(KLT) Launching kernel with %d features, %d blocks, %d threads per block...\n",
                nFeatures, grid.x, block.x);
        fflush(stderr);
    }

    trackFeatureKernel<<<grid, block>>>(
        pyramid_data.pyramid1, pyramid_data.pyramid2,
        d_pyramid_dims, d_pyramid_offsets,
        d_feature_data, nFeatures,
        tc->window_width, tc->window_height,
        tc->step_factor, tc->max_iterations,
        tc->min_determinant, tc->min_displacement,
        tc->max_residue, tc->lighting_insensitive,
        subsampling, tc->nPyramidLevels,
        d_imgdiff, d_gradx, d_grady);

    cudaDeviceSynchronize();
    
    cudaMemcpy(h_feature_data, d_feature_data, nFeatures * sizeof(FeatureData), cudaMemcpyDeviceToHost);

    // Update feature list
    int tracked_count = 0;
    for (indx = 0; indx < nFeatures; indx++) {
        FeatureData* feat = &h_feature_data[indx];
        if (feat->status != KLT_TRACKED) {
            featurelist->feature[indx]->x = -1.0;
            featurelist->feature[indx]->y = -1.0;
            featurelist->feature[indx]->val = feat->status;
        } else {
            featurelist->feature[indx]->x = feat->x2;
            featurelist->feature[indx]->y = feat->y2;
            featurelist->feature[indx]->val = KLT_TRACKED;
            tracked_count++;
        }
    }

    freePyramidDataCombined(&pyramid_data);
    cudaFree(d_feature_data);
    cudaFree(d_imgdiff);
    cudaFree(d_gradx);
    cudaFree(d_grady);
    cudaFree(d_pyramid_dims);
    cudaFree(d_pyramid_offsets);
    cudaFreeHost(h_feature_data);
    
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    
    free(h_pyramid_dims);
    free(h_pyramid_offsets);

    // Free CPU resources
    _KLTFreeFloatImage(tmpimg);
    if (floatimg1_created) {
        _KLTFreeFloatImage(floatimg1);
    }
    _KLTFreeFloatImage(floatimg2);
    
    if (!tc->sequentialMode) {
        _KLTFreePyramid(pyramid1);
        _KLTFreePyramid(pyramid1_gradx);
        _KLTFreePyramid(pyramid1_grady);
    }
    
    if (tc->sequentialMode) {
        tc->pyramid_last = pyramid2;
        tc->pyramid_last_gradx = pyramid2_gradx;
        tc->pyramid_last_grady = pyramid2_grady;
    } else {
        _KLTFreePyramid(pyramid2);
        _KLTFreePyramid(pyramid2_gradx);
        _KLTFreePyramid(pyramid2_grady);
    }

    if (KLT_verbose >= 1) {
        fprintf(stderr, "\n\t%d features successfully tracked.\n", tracked_count);
        fflush(stderr);
    }
}