#include "window_partition.h"

#include <cuda_runtime.h>
#include <iostream>

#define CUDA_CHECK(call)                                           \
do {                                                               \
    cudaError_t err = call;                                        \
    if (err != cudaSuccess) {                                      \
        std::cerr << "CUDA Error: "                                \
                  << cudaGetErrorString(err)                       \
                  << " at line " << __LINE__ << std::endl;          \
        exit(EXIT_FAILURE);                                        \
    }                                                              \
} while (0)

__global__
void WindowPartitionKernel(
    const float* input,
    float* output,
    int B,
    int D,
    int H,
    int W,
    int C,
    int windowD,
    int windowH,
    int windowW)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    int totalElements = B * D * H * W * C;

    if (tid >= totalElements)
        return;
    // Recover original tensor coordinates
    int temp = tid;

    int c = temp % C;
    temp /= C;

    int w = temp % W;
    temp /= W;

    int h = temp % H;
    temp /= H;

    int d = temp % D;
    temp /= D;

    int b = temp;

    // Number of windows along each dimension
    int windowsPerD = D / windowD;
    int windowsPerH = H / windowH;
    int windowsPerW = W / windowW;

    // Compute which window this element belongs to
    int window_d = d / windowD;
    int window_h = h / windowH;
    int window_w = w / windowW;
    // Compute the linear window ID
    int windowId =
        ((b * windowsPerD + window_d) * windowsPerH + window_h)
        * windowsPerW + window_w;

    // Local coordinates inside the window
    int local_d = d % windowD;
    int local_h = h % windowH;
    int local_w = w % windowW;

    int windowVolume = windowD * windowH * windowW;

    // Offset of the element within its window
    int localIndex =
        ((local_d * windowH + local_h) * windowW + local_w);

    // Compute output index
    int outputIndex =
        (windowId * windowVolume + localIndex) * C + c;

    // Copy the element
    output[outputIndex] = input[tid];
}

void WindowPartition(
    const float* input,
    float* output,
    int B,
    int D,
    int H,
    int W,
    int C,
    int windowD,
    int windowH,
    int windowW)
{
    int totalElements = B * D * H * W * C;

    const int threadsPerBlock = 256;
    const int blocks =
        (totalElements + threadsPerBlock - 1) / threadsPerBlock;

    WindowPartitionKernel<<<blocks, threadsPerBlock>>>(
        input,
        output,
        B,
        D,
        H,
        W,
        C,
        windowD,
        windowH,
        windowW);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}
