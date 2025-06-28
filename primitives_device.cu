#pragma once

#include "primitives_device.cuh"

// Dot Product between two matrices
__global__ void computeDotProduct(const double* m1, const double* m2, double* m3, size_t* m, size_t* k, size_t* n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < *m && col < *n) {
        double sum = 0;
        for (int i = 0; i < *k; ++i) {
            sum += m1[row * (*k) + i] * m2[i * (*n) + col];
        }
        m3[row * (*n) + col] = sum;
    }
}

__global__ void computeDotProductSIMD(const double* A, const double* B, double* C, size_t* M, size_t* K, size_t* N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N) return;

    double sum = 0.0;
    int i = 0;

    // SIMD-style computation using double4
    for (; i <= K - 4; i += 4) {
        double4 a = *((double4*)&A[row * K + i]);

        double4 b;
        b.x = B[(i + 0) * N + col];
        b.y = B[(i + 1) * N + col];
        b.z = B[(i + 2) * N + col];
        b.w = B[(i + 3) * N + col];

        sum += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
    }

    // Handle remaining tail if K is not divisible by 4
    for (; i < K; ++i) {
        sum += A[row * K + i] * B[i * N + col];
    }

    C[row * N + col] = sum;
}
