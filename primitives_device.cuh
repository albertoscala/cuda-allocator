#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

// Dot Product between two matrices
__global__ void computeDotProduct(const double* m1, const double* m2, double* m3, size_t* m, size_t* k, size_t* n);

__global__ void computeDotProductSIMD(const double* A, const double* B, double* C, size_t* M, size_t* K, size_t* N);