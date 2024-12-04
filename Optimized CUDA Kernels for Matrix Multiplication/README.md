# Optimized CUDA kernels for Matrix Multiplication
Thanks for visiting my project!

Overview

The goal of this project is to optimize CUDA kernels for matrix multiplication by evaluating performance metrics across different implementations. The project starts with a simple CPU-based implementation and progressively optimizes GPU-based approaches, exploring various CUDA techniques such as shared memory usage, loop unrolling, and leveraging cuBLAS.

Project Objectives

-To implement and compare different methods for matrix multiplication:

  1. **CPU Implementation**: Baseline comparison.
  2. **Naïve GPU Approach**: Initial CUDA implementation.
  3. **Shared Memory Optimization**: Use shared memory to improve global memory latency.
  4. **Loop Unrolling**: Increase parallelism by unrolling loops.
  5. **cuBLAS**: Compare with NVIDIA's highly optimized matrix multiplication library.

## Implementations Explained

### 1. CPU Implementation

This serves as a baseline to measure the performance gains obtained through GPU parallelization. The implementation is straightforward, using triple nested loops to compute matrix products. The algorithm has a time complexity of **O(N^3)** and is limited by the sequential nature of CPU processing.

- **Language**: C++
- **Libraries Used**: `<vector>`, `<chrono>` for timing.

#### Key Features:
- Implemented using nested loops.
- Measures performance using the `std::chrono` library.
- Uses input matrices initialized with constant values for validation.

### 2. Naive GPU implementation
The first GPU-based implementation uses a simple approach to parallelize matrix multiplication using CUDA. Each thread calculates a single element of the result matrix, which distributes the workload efficiently across the GPU cores. This implementation serves as a starting point for GPU optimization.

- **Language**: CUDA C++
- **Optimization level**: Basic
#### Key Features:
- Each element of the output matrix is computed by a single GPU thread.
- Uses a grid of thread blocks to parallelize computations.
- Significant speedup compared to the CPU, but limited by unoptimized global memory accesses.
- Measures performance using CUDA events for timing.

