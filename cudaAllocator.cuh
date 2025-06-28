#pragma once

#include "cuda_runtime.h"

#include <stdio.h>

// Bump (linear) allocator for GPU memory 
class CudaAllocator {
private:
    void* basePtr = nullptr;    // Buffer start
    size_t capacity = 0;        // Buffer capacity
    size_t offset = 0;          // Pointer to the first
                                // free memory address

    // Alignment function
    static size_t alignUp(size_t n, size_t alignment);
    
    // Constructor
    CudaAllocator(size_t size);

    // Delete copy constructor and assignment operator
    CudaAllocator(const CudaAllocator&) = delete;
    CudaAllocator& operator=(const CudaAllocator&) = delete;

    // Instance of the single allocator
    static CudaAllocator* instance;
    
public:
    // Destructor
    ~CudaAllocator();

    // Settings for the total memory for the allocator
    static size_t total_mem;

    // Function that returns the allocator instance
    static CudaAllocator* getAllocator();

    // Alloc function
    void* allocate(size_t size, size_t alignment = alignof(std::max_align_t));

    // Offset/memory reset
    void reset();

    // DEBUG functions
    size_t used();      // Returns until which point the memory is used
    size_t total();     // Returns the memory total size
};