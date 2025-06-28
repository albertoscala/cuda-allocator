#pragma once

#include "allocator.cuh"

// Alignment function
size_t CudaAllocator::alignUp(size_t n, size_t alignment) {
    return (n + alignment - 1) & ~(alignment - 1);
}

// Constructor
CudaAllocator::CudaAllocator(size_t size) : basePtr(nullptr), capacity(size), offset(0) {
    // Alloc
    cudaError_t cudaStatus = cudaMalloc(&this->basePtr, size);

    // In case of error
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "CudaAllocator: Preallocation failed: %s\n", cudaGetErrorString(cudaStatus));
        this->basePtr = nullptr;
        this->capacity = 0;
    }

}

// Instance of the single allocator
CudaAllocator* CudaAllocator::instance = nullptr;

// Destructor
CudaAllocator::~CudaAllocator() {
    cudaFree(this->basePtr);
}

// Settings for the total memory for the allocator
size_t CudaAllocator::total_mem = 0;

// Function that returns the allocator instance
CudaAllocator* CudaAllocator::getAllocator() {
    if (instance == nullptr) {
        instance = new CudaAllocator(CudaAllocator::total_mem);
    }
    return instance;
}

// Alloc function
void* CudaAllocator::allocate(size_t size, size_t alignment) {
    if (!basePtr) return nullptr;

    size_t current = offset;
    size_t aligned = alignUp(current, alignment);

    // Se la memoria richiesta eccede la memoria disponibile
    if (aligned + size > this->capacity) {
        fprintf(stderr, "CudaAllocator: Out of memory\n");
        return nullptr;
    }

    // Cast a char per aritmetica e assegnazione spazio di memoria
    void* ptr = static_cast<char*>(this->basePtr) + aligned;

    // Aggiornamento offset da cui la memoria � libera
    this->offset = aligned + size;
    return ptr;
}

// Offset/memory reset
void CudaAllocator::reset() {
    this->offset = 0;
}

// DEBUG functions
size_t CudaAllocator::used() { return this->offset; }       // Returns until which point the memory is used
size_t CudaAllocator::total() { return this->capacity; }    // Returns the memory total size