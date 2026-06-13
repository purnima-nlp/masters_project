#include <iostream>
#include <vector>
#include <cuda_runtime.h>

#include "window_partition.h"

int main()
{
    // Tensor dimensions
    const int B = 1;
    const int D = 4;
    const int H = 4;
    const int W = 4;
    const int C = 2;

    // Window size
    const int windowD = 2;
    const int windowH = 2;
    const int windowW = 2;

    const int inputSize = B * D * H * W * C;

    std::vector<float> h_input(inputSize);

    // Initialize input tensor
    for (int i = 0; i < inputSize; i++)
        h_input[i] = static_cast<float>(i);

    // Number of windows
    int numWindows =
        B *
        (D / windowD) *
        (H / windowH) *
        (W / windowW);

    int windowVolume =
        windowD *
        windowH *
        windowW;

    const int outputSize =
        numWindows *
        windowVolume *
        C;

    std::vector<float> h_output(outputSize);

    float* d_input = nullptr;
    float* d_output = nullptr;

    cudaMalloc(&d_input, inputSize * sizeof(float));
    cudaMalloc(&d_output, outputSize * sizeof(float));

    cudaMemcpy(
        d_input,
        h_input.data(),
        inputSize * sizeof(float),
        cudaMemcpyHostToDevice
    );

    // Launch CUDA kernel
    WindowPartition(
        d_input,
        d_output,
        B,
        D,
        H,
        W,
        C,
        windowD,
        windowH,
        windowW
    );

    cudaMemcpy(
        h_output.data(),
        d_output,
        outputSize * sizeof(float),
        cudaMemcpyDeviceToHost
    );

    std::cout << "Window Partition Output\n";
    std::cout << "=======================\n";

    for (int i = 0; i < outputSize; i++)
    {
        std::cout << h_output[i] << " ";

        if ((i + 1) % (windowVolume * C) == 0)
            std::cout << "\n\n";
    }

    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}
