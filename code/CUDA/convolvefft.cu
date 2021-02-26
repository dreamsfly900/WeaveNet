#ifndef __CONV2D3X3_H__
#define __CONV2D3X3_H__
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
extern void conv2Mex(float* in, float* out, int numRows, int numCols, float* kernel);

#endif // __CONV2D3X3_H__

extern "C"
__global__ void conv2MexCuda(float* src,
    float* dst,
    int numRows,
    int numCols,
    float* kernel)
{
    int row = blockIdx.x;
    if (row < 1 || row > numRows - 1)
        return;

    int col = blockIdx.y;
    if (col < 1 || col > numCols - 1)
        return;

    int dstIndex = col * numRows + row;
    dst[dstIndex] = 0;
    int kerIndex = 3 * 3 - 1;
    for (int kc = -1; kc < 2; kc++)
    {
        int srcIndex = (col + kc) * numRows + row;
        for (int kr = -1; kr < 2; kr++)
        {
            dst[dstIndex] += kernel[kerIndex--] * src[srcIndex + kr];
        }
    }
}

void conv2Mex(float* src, float* dst, int numRows, int numCols, float* ker)
{
    int totalPixels = numRows * numCols;
    float* deviceSrc, * deviceKer, * deviceDst;

    cudaMalloc(&deviceSrc, sizeof(float) * totalPixels);
    cudaMalloc(&deviceDst, sizeof(float) * totalPixels);
    cudaMalloc(&deviceKer, sizeof(float) * 3 * 3);

    cudaMemcpy(deviceSrc, src, sizeof(float) * totalPixels, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceKer, ker, sizeof(float) * 3 * 3, cudaMemcpyHostToDevice);
    cudaMemset(deviceDst, 0, sizeof(float) * totalPixels);

    dim3 gridSize(numRows, numCols);
    conv2MexCuda<<<gridSize,1>>>(deviceSrc, deviceDst, numRows, numCols, deviceKer);

    cudaMemcpy(dst, deviceDst, sizeof(float) * totalPixels, cudaMemcpyDeviceToHost);

    cudaFree(deviceSrc);
    cudaFree(deviceDst);
    cudaFree(deviceKer);
}