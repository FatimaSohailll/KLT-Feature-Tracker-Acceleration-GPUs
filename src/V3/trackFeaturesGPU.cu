/* Standard includes */
#include <assert.h>
#include <math.h>		/* fabs() */
#include <stdlib.h>		/* malloc() */
#include <stdio.h>		/* fflush() */
#include <cuda_runtime.h>

/* Our includes */
#include "base.h"
#include "error.h"
#include "convolveGPU.h"	/* for computing pyramid */
#include "kltGPU.h"
#include "klt_utilGPU.h"	/* _KLT_FloatImage */
#include "pyramidGPU.h"	/* _KLT_Pyramid */
#include "cuda_utils.h"

extern int KLT_verbose;

typedef float *_FloatWindow;

typedef struct{
  float* data;
  int ncols;
  int nrows;
} FloatImage_GPU;

typedef struct{
  float x1, y1;
  float x2, y2;
  int status;
} FeatureData_GPU;

typedef struct {
  FloatImage_GPU img, gradx, grady;
} PyramidLevel_GPU;


// interpolate a pixel
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

// compute summed gradient windows
__device__ void gpu_computeGradientSum(
    const float* gradx1, const float* grady1, 
    const float* gradx2, const float* grady2,
    float* out_gradx, float* out_grady, 
    int cols, int rows, 
    float x1, float y1, float x2, float y2, 
    int width, int height) {
    
    int w= width/2, h=height/2;

    for (int j= -h;j<= h; j++) {
        for (int i = -w;i <= w;i++) {
            int idx = (j + h)*width + (i+ w);
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

// Given two images and the window center in both images, aligns the images wrt the window and computes the difference 
// between the two overlaid images.
__device__ void gpu_computeIntensityDifference(
    const float* img1, const float* img2, 
    int cols, int rows,
    float x1, float y1, float x2, float y2, 
    int width, int height, 
    float* imgDiff) {
    
    int w = width / 2, h = height / 2;

    for (int j = -h; j <= h; j++) {
        for (int i = -w; i <= w; i++) {
            int idx= (j +h) * width +(i+ w);

            // calculate interpolated values
            float g1 = gpu_interpolate(x1 + i, y1 + j, img1, cols, rows);
            float g2 = gpu_interpolate(x2 + i, y2 + j, img2, cols, rows);
    
            imgDiff[idx] = g1 - g2;
        }
    }
}

/*Given two images and the window center in both images,
 * aligns the images wrt the window and computes the difference 
 * between the two overlaid images; normalizes for overall gain and bias.
 */

__device__ void gpu_computeIntensityDifferenceLightingInsensitive(
    const float* img1, const float* img2, 
    float x1, float y1, float x2, float y2,
    int width, int height, 
    int cols, int rows, 
    float* imgDiff) {
    
    int w = width / 2, h = height / 2;
    float sum1 = 0.0f, sum2 = 0.0f;
    float sum1_sq = 0.0f, sum2_sq = 0.0f;
    int total = width* height;

    // compute sums and squared sums
    for (int j = -h; j<= h; j++) {
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

/* Given two gradients and the window center in both images,
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

    // apply step factor
    *ex = sum_ex * step_factor;
    *ey = sum_ey * step_factor;
}


// gpu version of solve equation
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


__device__ float gpu_sumAbsFloatWindow(const float* fw, int width, int height) {
    float sum = 0.0f;
    int total = width * height;

    for (int i = 0; i < total; i++)
        sum += fabsf(fw[i]);
    return sum;
}

// gpu version of track features
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
        /* If out of bounds, exit loop */
        if (  x1-hw < 0.0f || nc-( x1+hw) < one_plus_eps ||
            *x2-hw < 0.0f || nc-(*x2+hw) < one_plus_eps ||
            y1-hh < 0.0f || nr-( y1+hh) < one_plus_eps ||
            *y2-hh < 0.0f || nr-(*y2+hh) < one_plus_eps) {
        status = KLT_OOB;
        break;
        }

        /* Compute gradient and difference windows */
        if (lighting_insensitive) {
            gpu_computeIntensityDifferenceLightingInsensitive(img1, img2, x1, y1, *x2, *y2, 
                width, height, nc, nr, imgdiff);
            gpu_computeGradientSumLightingInsensitive(gradx1, grady1, gradx2, grady2,
                img1, img2, x1, y1, *x2, *y2, width, height, nc, nr, gradx, grady);
        }
        else {
            gpu_computeIntensityDifference(img1, img2, nc, nr, x1, y1, *x2, *y2, 
                width, height, imgdiff);
            gpu_computeGradientSum(
                gradx1, grady1, gradx2, grady2, gradx, grady, nc, nr, x1, y1, *x2, *y2, 
                width, height);
        }

         /* Use these windows to construct matrices */
        gpu_compute2by2GradientMatrix(gradx, grady, width, height, &gxx, &gxy, &gyy);
        gpu_compute2by1ErrorVector(imgdiff, gradx, grady, width, height, step_factor, &ex, &ey);

        //solve for displacement
        status = gpu_solveEquation(gxx, gxy, gyy, ex, ey, small, &dx, &dy);
        if (status == KLT_SMALL_DET)
            break;

        *x2 += dx;
        *y2 += dy;
        iteration++;

    } while ((fabsf(dx) >= th || fabsf(dy) >= th) && iteration < max_iterations);

    /* Check whether window is out of bounds */
    if (status == KLT_TRACKED && 
        (*x2 - hw < 0.0f || nc - (*x2 + hw) < one_plus_eps ||
         *y2 - hh < 0.0f || nr - (*y2 + hh) < one_plus_eps)) {
        status = KLT_OOB;
    }

    /* Check whether residue is too large */
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

// kernel
__global__ void trackFeatureKernel(
    FloatImage_GPU img1, FloatImage_GPU gradx1, FloatImage_GPU grady1,
    FloatImage_GPU img2, FloatImage_GPU gradx2, FloatImage_GPU grady2,
    FeatureData_GPU* features,
    int num_features,
    int width, int height,
    float step_factor,
    int max_iterations,
    float small,
    float th,
    float max_residue,
    int lighting_insensitive) {

    // declare shared memory
    extern __shared__ float shared_buffers[];
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_features) return;

    int window_size = width*height;

    int thread_offset = threadIdx.x * 3 * window_size;

    float* imgdiff = shared_buffers + thread_offset;
    float* gradx = imgdiff + window_size;
    float* grady = gradx+window_size;

    float x1 = features[idx].x1;
    float y1 = features[idx].y1;
    float x2 = features[idx].x2;
    float y2 = features[idx].y2;

    int status = gpu_trackFeature(
        x1, y1, &x2, &y2,
        img1.data, gradx1.data, grady1.data,
        img2.data, gradx2.data, grady2.data,
        img1.ncols, img1.nrows, width, height,
        step_factor, max_iterations,
        small, th, max_residue,
        lighting_insensitive,
        imgdiff, gradx, grady);

    // final results in global memory
    features[idx].x2 = x2;
    features[idx].y2 = y2;
    features[idx].status = status;
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

    // CUDA events for timing
    cudaEvent_t start_total, stop_total;
    cudaEvent_t start_pyramid, stop_pyramid;
    cudaEvent_t start_memcpy_h2d, stop_memcpy_h2d;
    cudaEvent_t start_kernel, stop_kernel;
    cudaEvent_t start_memcpy_d2h, stop_memcpy_d2h;
    
    float total_time = 0.0f, pyramid_time = 0.0f;
    float memcpy_h2d_time = 0.0f, kernel_time = 0.0f, memcpy_d2h_time = 0.0f;

    // Create CUDA events
    cudaEventCreate(&start_total);
    cudaEventCreate(&stop_total);
    cudaEventCreate(&start_pyramid);
    cudaEventCreate(&stop_pyramid);
    cudaEventCreate(&start_memcpy_h2d);
    cudaEventCreate(&stop_memcpy_h2d);
    cudaEventCreate(&start_kernel);
    cudaEventCreate(&stop_kernel);
    cudaEventCreate(&start_memcpy_d2h);
    cudaEventCreate(&stop_memcpy_d2h);

    // Start total timer
    cudaEventRecord(start_total);

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

    // Start pyramid construction timer
    cudaEventRecord(start_pyramid);

    //Temporary float image
    tmpimg = _KLTCreateFloatImage(ncols, nrows);

    //First image pyramid
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

    // Stop pyramid construction timer
    cudaEventRecord(stop_pyramid);
    cudaEventSynchronize(stop_pyramid);
    cudaEventElapsedTime(&pyramid_time, start_pyramid, stop_pyramid);


    // create streams to overlap data transfers
    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);


    FeatureData_GPU* h_features; 
    // allocate pinned memory for host oordinates of 2 images
    cudaMallocHost(&h_features, nFeatures* sizeof(FeatureData_GPU));

    for (indx = 0; indx < nFeatures; indx++) {
        h_features[indx].x1 = featurelist->feature[indx]->x;
        h_features[indx].y1 = featurelist->feature[indx]->y;
        h_features[indx].x2 = featurelist->feature[indx]->x;
        h_features[indx].y2 = featurelist->feature[indx]->y;
        h_features[indx].status = -1;
    }

    // create gpu struct arrays for features
    FeatureData_GPU* d_features;
    cudaMalloc(&d_features, nFeatures* sizeof(FeatureData_GPU));

    _KLT_FloatImage base_img1 = pyramid1->img[0];
    _KLT_FloatImage base_gradx1 = pyramid1_gradx->img[0];
    _KLT_FloatImage base_grady1 = pyramid1_grady->img[0];
    _KLT_FloatImage base_img2 = pyramid2->img[0];
    _KLT_FloatImage base_gradx2 = pyramid2_gradx->img[0];
    _KLT_FloatImage base_grady2 = pyramid2_grady->img[0];

    int base_cols = pyramid1->ncols[0];
    int base_rows = pyramid1->nrows[0];
    int img_size = base_cols * base_rows * sizeof(float);

    //Allocate GPU memory for pyramid images
    PyramidLevel_GPU d_pyramid1, d_pyramid2;

    // initialise 
    d_pyramid1.img.ncols = d_pyramid1.gradx.ncols = d_pyramid1.grady.ncols = base_cols;
    d_pyramid1.img.nrows = d_pyramid1.gradx.nrows = d_pyramid1.grady.nrows = base_rows;
    d_pyramid2.img.ncols = d_pyramid2.gradx.ncols = d_pyramid2.grady.ncols = base_cols;
    d_pyramid2.img.nrows = d_pyramid2.gradx.nrows = d_pyramid2.grady.nrows = base_rows;
    
    cudaMalloc(&d_pyramid1.img.data, img_size);
    cudaMalloc(&d_pyramid1.gradx.data, img_size);
    cudaMalloc(&d_pyramid1.grady.data, img_size);
    cudaMalloc(&d_pyramid2.img.data, img_size);
    cudaMalloc(&d_pyramid2.gradx.data, img_size);
    cudaMalloc(&d_pyramid2.grady.data, img_size);

    // sart host to device transfer timer
    cudaEventRecord(start_memcpy_h2d);

    // batch transfer in streams
    cudaMemcpyAsync(d_features, h_features, nFeatures* sizeof(FeatureData_GPU), cudaMemcpyHostToDevice, stream1);

    cudaMemcpyAsync(d_pyramid1.img.data, base_img1->data, img_size, cudaMemcpyHostToDevice, stream1);
    cudaMemcpyAsync(d_pyramid1.gradx.data, base_gradx1->data, img_size, cudaMemcpyHostToDevice, stream1);
    cudaMemcpyAsync(d_pyramid1.grady.data, base_grady1->data, img_size, cudaMemcpyHostToDevice, stream1);

    // concurrently transfer second pyramid data in stream 2
    cudaMemcpyAsync(d_pyramid2.img.data, base_img2->data, img_size, cudaMemcpyHostToDevice, stream2);
    cudaMemcpyAsync(d_pyramid2.gradx.data, base_gradx2->data, img_size, cudaMemcpyHostToDevice, stream2);
    cudaMemcpyAsync(d_pyramid2.grady.data, base_grady2->data, img_size, cudaMemcpyHostToDevice, stream2);

    // wait for all streams to finish their transfers
    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);

    // Stop Host-to-Device memory transfer timer
    cudaEventRecord(stop_memcpy_h2d);
    cudaEventSynchronize(stop_memcpy_h2d);
    cudaEventElapsedTime(&memcpy_h2d_time, start_memcpy_h2d, stop_memcpy_h2d);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    // Get GPU-specific information
    int max_threads_per_block = prop.maxThreadsPerBlock;
    int multiprocessor_count = prop.multiProcessorCount;
    int warp_size = prop.warpSize;

    // Calculate optimal block size for nFeatures
    int block_size;
    if (nFeatures <= 32) {
        block_size = 32; // 1 warp
    } else if (nFeatures <= 64) {
        block_size = 64; // 2 warps
    } else if (nFeatures <= 128) {
        block_size = 128; // 4 warps
    } else if (nFeatures <= 256) {
        block_size = 256; // 8 warps
    } else {
        block_size = 256;
    }

    int window_size = tc->window_height * tc->window_width;
    
    size_t shared_mem_per_thread = 3*window_size * sizeof(float);
    size_t max_shared_mem = prop.sharedMemPerBlock;

    int max_threads_by_shared_mem = max_shared_mem/ shared_mem_per_thread;

    // Ensure block size is multiple of warp size and within limits
    block_size = min(block_size, max_threads_by_shared_mem);
    block_size = min(block_size, max_threads_per_block);
    block_size = (block_size / warp_size) * warp_size;

    dim3 block(block_size);
    dim3 grid((nFeatures + block.x - 1) / block.x);

    size_t shared_mem = block_size *3 * window_size* sizeof(float);

    printf("Launch: %d features -> %d blocks x %d threads, shared memory: %zu bytes\n", nFeatures, grid.x, block.x, shared_mem);
    
    // Start kernel execution timer
    cudaEventRecord(start_kernel);

    // launch kernel on stream 3
    
    trackFeatureKernel<<<grid, block, shared_mem, stream1>>>(
        d_pyramid1.img, d_pyramid1.gradx, d_pyramid1.grady,
        d_pyramid2.img, d_pyramid2.gradx, d_pyramid2.grady,
        d_features, nFeatures,
        tc->window_width, tc->window_height,
        tc->step_factor, tc->max_iterations,
        tc->min_determinant, tc->min_displacement,
        tc->max_residue, tc->lighting_insensitive);

    cudaStreamSynchronize(stream1);

    // Stop kernel execution timer
    cudaEventRecord(stop_kernel);
    cudaEventSynchronize(stop_kernel);
    cudaEventElapsedTime(&kernel_time, start_kernel, stop_kernel);

    // Start Device-to-Host memory transfer timer
    cudaEventRecord(start_memcpy_d2h);

    //copy results back from device
    cudaMemcpyAsync(h_features, d_features, nFeatures * sizeof(FeatureData_GPU),
                   cudaMemcpyDeviceToHost, stream1);

    cudaStreamSynchronize(stream1);

    // Stop Device-to-Host memory transfer timer
    cudaEventRecord(stop_memcpy_d2h);
    cudaEventSynchronize(stop_memcpy_d2h);
    cudaEventElapsedTime(&memcpy_d2h_time, start_memcpy_d2h, stop_memcpy_d2h);

    //Update feature list with result
    for (indx = 0; indx < nFeatures; indx++) {
        int status = h_features[indx].status;
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
            featurelist->feature[indx]->x = h_features[indx].x2;
            featurelist->feature[indx]->y = h_features[indx].y2;
            featurelist->feature[indx]->val = KLT_TRACKED;
        }
    }

    //free ALL device memory
    cudaFree(d_features);
    cudaFree(d_pyramid1.img.data); cudaFree(d_pyramid1.gradx.data); cudaFree(d_pyramid1.grady.data);
    cudaFree(d_pyramid2.img.data); cudaFree(d_pyramid2.gradx.data); cudaFree(d_pyramid2.grady.data);

    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);

    //free host pinned memory
    cudaFreeHost(h_features);

    //free pyramid memory
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

    // Stop total timer
    cudaEventRecord(stop_total);
    cudaEventSynchronize(stop_total);
    cudaEventElapsedTime(&total_time, start_total, stop_total);

    // Print timing results
    printf("\n=== KLT GPU Performance Metrics ===\n");
    printf("Total execution time:        %8.3f ms\n", total_time);
    printf("  Pyramid construction:      %8.3f ms (%5.1f%%)\n", 
           pyramid_time, (pyramid_time / total_time) * 100.0f);
    printf("  Memory H2D transfer:       %8.3f ms (%5.1f%%)\n", 
           memcpy_h2d_time, (memcpy_h2d_time / total_time) * 100.0f);
    printf("  Kernel execution:          %8.3f ms (%5.1f%%)\n", 
           kernel_time, (kernel_time / total_time) * 100.0f);
    printf("  Memory D2H transfer:       %8.3f ms (%5.1f%%)\n", 
           memcpy_d2h_time, (memcpy_d2h_time / total_time) * 100.0f);
    printf("  Other overhead:            %8.3f ms (%5.1f%%)\n", 
           total_time - pyramid_time - memcpy_h2d_time - kernel_time - memcpy_d2h_time,
           ((total_time - pyramid_time - memcpy_h2d_time - kernel_time - memcpy_d2h_time) / total_time) * 100.0f);
    
    // Calculate performance metrics
    if (kernel_time > 0 && nFeatures > 0) {
        float features_per_ms = nFeatures / kernel_time;
        float features_per_second = features_per_ms * 1000.0f;
        printf("\nPerformance Metrics:\n");
        printf("  Features processed:        %d features\n", nFeatures);
        printf("  Tracking throughput:       %.1f features/ms\n", features_per_ms);
        printf("  Tracking throughput:       %.1f features/second\n", features_per_second);
        printf("  Time per feature:          %.3f ms/feature\n", kernel_time / nFeatures);
    }

    if (KLT_verbose >= 1) {
        fprintf(stderr, "\n\t%d features successfully tracked.\n",
            KLTCountRemainingFeatures(featurelist));
        if (tc->writeInternalImages)
            fprintf(stderr, "\tWrote images to 'kltimg_tf*.pgm'.\n");
        fflush(stderr);
    }

    // Destroy CUDA events
    cudaEventDestroy(start_total);
    cudaEventDestroy(stop_total);
    cudaEventDestroy(start_pyramid);
    cudaEventDestroy(stop_pyramid);
    cudaEventDestroy(start_memcpy_h2d);
    cudaEventDestroy(stop_memcpy_h2d);
    cudaEventDestroy(start_kernel);
    cudaEventDestroy(stop_kernel);
    cudaEventDestroy(start_memcpy_d2h);
    cudaEventDestroy(stop_memcpy_d2h);
}