#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include <vector>
#include <iostream>

int const CUDA_NUM_THREADS = 1024;

inline int CUDA_GET_BLOCKS(const int N) {
    return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}

#define CUDA_1D_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); \
       i += blockDim.x * gridDim.x)

__device__ float sgn(float x) {
    return x > 0 ? 1.0 : -1.0;
}

__global__ void afm_kernel(const int nthreads, const float* lines, const int* shape_info, const int num, const int height, const int width, float* afmap, int* aflabel) {
    CUDA_1D_KERNEL_LOOP(index, nthreads) {
        int w = index % width;
        int h = (index / width) % height;
        int n = index / width / height;
        int x_index = n * 2 * height * width + h * width + w;
        int y_index = n * 2 * height * width + height * width + h * width + w;
        int label_index = n * height * width + h * width + w;

        float px = (float) w;
        float py = (float) h;
        int start = shape_info[n * 4];
        int end = shape_info[n * 4 + 1];
        float min_dis = 1e30;

        for (int i = start; i < end; ++i) {
            float xs = (float)width / (float)shape_info[n * 4 + 3];
            float ys = (float)height / (float)shape_info[n * 4 + 2];
            float x1 = lines[4 * i] * xs;
            float y1 = lines[4 * i + 1] * ys;
            float x2 = lines[4 * i + 2] * xs;
            float y2 = lines[4 * i + 3] * ys;

            float dx = x2 - x1;
            float dy = y2 - y1;
            float norm2 = dx * dx + dy * dy;

            float t = ((px - x1) * dx + (py - y1) * dy) / (norm2 + 1e-6);
            t = t < 1.0 ? t : 1.0;
            t = t > 0.0 ? t : 0.0;

            float ax = x1 + t * dx - px;
            float ay = y1 + t * dy - py;

            float dis = ax * ax + ay * ay;
            if (dis < min_dis) {
                min_dis = dis;
                afmap[x_index] = -sgn(ax) * logf(fabsf(ax / float(width)) + 1e-6f);
                afmap[y_index] = -sgn(ay) * logf(fabsf(ay / float(height)) + 1e-6f);
                aflabel[label_index] = i - start;
            }
        }
    }
}

std::tuple<at::Tensor, at::Tensor> afm_cuda(
    const at::Tensor& lines,
    const at::Tensor& shape_info,
    const int height,
    const int width) 
{
    auto batch_size = shape_info.size(0);
    auto afmap = at::zeros({batch_size, 2, height, width}, lines.options());
    auto aflabel = at::zeros({batch_size, 1, height, width}, lines.options().dtype(at::kInt));

    auto nthreads = batch_size * height * width;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    float* afmap_data = afmap.data_ptr<float>();
    int* aflabel_data = aflabel.data_ptr<int>();

    afm_kernel<<<CUDA_GET_BLOCKS(nthreads), CUDA_NUM_THREADS, 0, stream>>>(
        nthreads,
        lines.contiguous().data_ptr<float>(),
        shape_info.contiguous().data_ptr<int>(),
        batch_size, height, width,
        afmap_data,
        aflabel_data);

    // ? CUDA ???????? THCudaCheck
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }

    return std::make_tuple(afmap, aflabel);
}