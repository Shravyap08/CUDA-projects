# Optimized CUDA kernels for Matrix Multiplication

Overview

The goal of this project is to optimize CUDA kernels for matrix multiplication by evaluating performance metrics across different implementations. The project starts with a simple CPU-based implementation and progressively optimizes GPU-based approaches, exploring various CUDA techniques such as shared memory usage, loop unrolling, and leveraging cuBLAS.

Project Objectives

To implement and compare different methods for matrix multiplication:

CPU Implementation: Baseline comparison.

Naïve GPU Approach: Initial CUDA implementation.

Shared Memory Optimization: Use shared memory to improve global memory latency.

Loop Unrolling: Increase parallelism by unrolling loops.

cuBLAS: Compare with NVIDIA's highly optimized matrix multiplication library.


