#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>

#include "../include/cuda_allocator.cuh"

constexpr std::size_t SIZE = 1024;

template <typename T>
__global__ void vectorAdd(T* a, T* b, T* c)
{
    std::uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    c[idx] = a[idx] + b[idx];
}

int main()
{
    CudaAllocator<SIZE>& allocator = CudaAllocator<SIZE>::getAllocator();

    std::array<int, 10> a{0,1,2,3,4,5,6,7,8,9};
    std::array<int, 10> b{0,2,4,6,8,0,2,4,6,8};
    std::array<int, 10> c{};

    int* d_a = allocator.alloc<int>(10);
    int* d_b = allocator.alloc<int>(10);
    int* d_c = allocator.alloc<int>(10);

    cudaMemcpy(d_a, a.data(), 10 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), 10 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, c.data(), 10 * sizeof(int), cudaMemcpyHostToDevice);

    vectorAdd<int><<<1, 10>>>(d_a, d_b, d_c);
    cudaDeviceSynchronize();

    cudaMemcpy(c.data(), d_c, 10 * sizeof(int), cudaMemcpyDeviceToHost);

    for (auto val : c) std::cout << val << "\t";
    std::cout << "\n";

    return 0;
}
