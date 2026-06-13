#ifndef WINDOW_PARTITION_H
#define WINDOW_PARTITION_H

/**
 * @brief Performs 3D window partitioning on a 5D tensor.
 *
 * Input Tensor Layout:
 *      [B][D][H][W][C]
 *
 * Output Tensor Layout:
 *      [num_windows][window_volume][C]
 *
 * where:
 *      num_windows =
 *          B * (D/windowD) * (H/windowH) * (W/windowW)
 *
 *      window_volume =
 *          windowD * windowH * windowW
 *
 * The input tensor is assumed to be stored in contiguous row-major order.
 */

/**
 * @brief Launches the CUDA window partition kernel.
 *
 * @param input     Pointer to input tensor on GPU.
 * @param output    Pointer to output tensor on GPU.
 *
 * @param B         Batch size.
 * @param D         Depth.
 * @param H         Height.
 * @param W         Width.
 * @param C         Number of channels.
 *
 * @param windowD   Window size along depth.
 * @param windowH   Window size along height.
 * @param windowW   Window size along width.
 */
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
    int windowW
);

#endif // WINDOW_PARTITION_H
