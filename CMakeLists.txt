cmake_minimum_required(VERSION 3.18)
project(CudaAllocator LANGUAGES CXX CUDA)

# C++ and CUDA standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Your executable
add_executable(main
    src/main.cu
)

# Include directories
target_include_directories(main PRIVATE
    include/
)

# CUDA architecture — set this to match your GPU
set_target_properties(main PROPERTIES
    CUDA_ARCHITECTURES 52
)
