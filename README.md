# cuda-allocator

A lightweight, header-only **bump (linear) allocator** for GPU memory in CUDA C++. Instead of calling `cudaMalloc` repeatedly, it pre-allocates a single large buffer and hands out aligned slices from it, fast, simple, and with zero fragmentation.

---

## How it works

A bump allocator keeps a single offset pointer into a pre-allocated buffer. Every allocation just advances the offset by the requested size (aligned). There is no per-allocation bookkeeping and no free list, resetting the entire allocator is O(1).

```
 basePtr                         offset        capacity
    │                               │               │
    ▼                               ▼               ▼
    ┌──────────┬──────────┬─────────┬───────────────┐
    │  alloc 1 │  alloc 2 │ alloc 3 │   free space  │
    └──────────┴──────────┴─────────┴───────────────┘
```

---

## Features

- **Single pre-allocation**: one `cudaMalloc` at startup, no per-call overhead
- **Alignment support**: every allocation is aligned to the type's natural alignment
- **Singleton pattern**: one global allocator instance per buffer size
- **Typed allocations**: `alloc<T>(count)` returns a `T*` directly, no casting needed
- **Reset**: wipe and reuse the entire buffer in one call
- **Header-only**: just include the `.cuh` file, no linking required

---

## Requirements

- CUDA Toolkit (tested with CUDA 12+)
- C++17 or later
- CMake 3.18+

---

## Usage

```cpp
#include "cuda_allocator.cuh"

constexpr std::size_t POOL_SIZE = 1024 * 1024; // 1 MB

int main() {
    // Get the singleton allocator
    auto& allocator = CudaAllocator<POOL_SIZE>::getAllocator();

    // Allocate typed GPU memory no sizeof, no cast
    int*   d_a = allocator.alloc<int>(256);
    float* d_b = allocator.alloc<float>(256);

    // Use normally with cudaMemcpy, kernels, etc.
    // ...

    // Reset the entire pool (O(1))
    allocator.reset();

    // Debug
    std::cout << allocator.used()  << " bytes used\n";
    std::cout << allocator.total() << " bytes total\n";
}
```

---

## Example - vector addition

```cpp
constexpr std::size_t SIZE = 1024;

template <typename T>
__global__ void vectorAdd(T* a, T* b, T* c, std::uint32_t n) {
    std::uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    c[idx] = a[idx] + b[idx];
}

int main() {
    auto& allocator = CudaAllocator<SIZE>::getAllocator();

    std::array<int, 10> a{0,1,2,3,4,5,6,7,8,9};
    std::array<int, 10> b{0,2,4,6,8,0,2,4,6,8};
    std::array<int, 10> c{};

    int* d_a = allocator.alloc<int>(10);
    int* d_b = allocator.alloc<int>(10);
    int* d_c = allocator.alloc<int>(10);

    cudaMemcpy(d_a, a.data(), 10 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), 10 * sizeof(int), cudaMemcpyHostToDevice);

    vectorAdd<int><<<1, 10>>>(d_a, d_b, d_c, 10);
    cudaDeviceSynchronize();

    cudaMemcpy(c.data(), d_c, 10 * sizeof(int), cudaMemcpyDeviceToHost);

    for (auto val : c) std::cout << val << "\t";
    std::cout << "\n";
}
```

---

## Building

```bash
mkdir build && cd build
cmake ..
cmake --build .
```

CMakeLists.txt minimum:

```cmake
cmake_minimum_required(VERSION 3.18)
project(CudaAllocator LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

add_executable(main src/main.cu)
target_include_directories(main PRIVATE include/)
set_target_properties(main PROPERTIES CUDA_ARCHITECTURES native)
```

---

## Project structure

```
CudaAllocator/
├── CMakeLists.txt
├── README.md
├── include/
│   └── cuda_allocator.cuh
└── src/
    └── main.cu
```

---

## API

| Method | Description |
|---|---|
| `getAllocator()` | Returns the singleton instance |
| `alloc<T>(count)` | Allocates `count` elements of type `T`, returns `T*` |
| `alloc(bytes)` | Allocates raw bytes, returns `void*` |
| `reset()` | Resets the offset to 0 and zeroes the buffer |
| `used()` | Returns bytes currently allocated |
| `total()` | Returns total buffer capacity in bytes |

---

## Limitations

- **No individual frees**: bump allocators only support resetting the whole pool
- **No thread safety**: not safe to call `alloc` concurrently from multiple CPU threads
- Move semantics are disabled alongside copy: the allocator is a singleton and should never be moved or copied

---

## License

MIT
