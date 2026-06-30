#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <iostream>

// Bump (linear) allocator for GPU memory
template<const std::size_t SIZE>
class CudaAllocator {
private:
    void* basePtr = nullptr;    // Buffer start
    size_t capacity = 0;        // Buffer capacity
    size_t offset = 0;          // Pointer to the first free memory address

    // Alignment function
    static size_t alignUp(size_t n, size_t alignment)
    {
        return (n + alignment - 1) & ~(alignment - 1);
    }

    // Delete copy constructor and assignment operator
    CudaAllocator(const CudaAllocator&) = delete;
    CudaAllocator& operator=(const CudaAllocator&) = delete;

public:
    // Constructor
    CudaAllocator() : basePtr(nullptr), capacity(SIZE), offset(0)
    {
        // Alloc
        cudaError_t cudaStatus = cudaMalloc(&this->basePtr, SIZE);

        // In case of error
        if (cudaStatus != cudaSuccess)
        {
            std::cerr << "CudaAllocator: Preallocation failed: " << cudaGetErrorString(cudaStatus) << "\n";
            basePtr = nullptr;
            capacity = 0;
        }

    }

    // Enable move semantics
    CudaAllocator(CudaAllocator&& other) noexcept
        : basePtr(other.basePtr), capacity(other.capacity), offset(other.offset) 
    {
        other.basePtr = nullptr;
        other.capacity = 0;
        other.offset = 0;
    }

    CudaAllocator& operator=(CudaAllocator&& other) noexcept
    {
        if (this != &other) 
        {
            // The "old" buffer i now useless
            cudaFree(basePtr);

            // Move the values to the new object
            basePtr = other.basePtr;
            capacity = other.capacity;
            offset = other.offset;

            // Clear the previous object
            other.basePtr = nullptr;
            other.capacity = 0;
            other.offset = 0;
        }
        return *this;
    }

    // Destructor
    ~CudaAllocator()
    {
        cudaFree(this->basePtr);
    }

    // Alloc function
    void* alloc(size_t size, size_t alignment = alignof(std::max_align_t))
    {
        if (!basePtr) return nullptr;

        size_t aligned = alignUp(offset, alignment);

        // If memory available is less than the requested one
        if (aligned + size > this->capacity) {
            std::cerr << "CudaAllocator: Out of memory\n";
            return nullptr;
        }

        void* ptr = static_cast<std::uint8_t*>(this->basePtr) + aligned;

        // Updated offset
        this->offset = aligned + size;
        return ptr;
    }

    template<typename T>
    T* alloc(size_t size, size_t alignment = alignof(T))
    {
        return static_cast<T*>(alloc(size * sizeof(T), alignment));
    }

    // Offset/memory reset
    void reset()
    {
        cudaError_t status = cudaMemset(basePtr, 0, SIZE);
        if (status != cudaSuccess)
            std::cerr << "CudaAllocator: reset failed: " << cudaGetErrorString(status) << "\n";
        offset = 0;
    }

    // DEBUG functions
    size_t used() const { return offset; }      // Returns until which point the memory is used
    constexpr size_t total() const { return SIZE; }     // Returns the memory total size
};
