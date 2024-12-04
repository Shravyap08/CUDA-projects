// naive_matrix_mul.cu
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <chrono>

// CUDA Kernel for naive matrix multiplication
__global__ void matrixMulNaive(float* A, float* B, float* C, int M, int K, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; // Row index of C
    int col = blockIdx.x * blockDim.x + threadIdx.x; // Column index of C

    if (row < M && col < N) {
        float value = 0.0f;
        for (int e = 0; e < K; ++e) {
            value += A[row * K + e] * B[e * N + col];
        }
        C[row * N + col] = value;
    }
}

int main() {
    // Matrix dimensions
    int M = 1024; // Rows of A and C
    int K = 1024; // Columns of A and Rows of B
    int N = 1024; // Columns of B and C

    // Host memory allocation
    std::vector<float> h_A(M * K, 1.0f); // Initialize A with 1.0
    std::vector<float> h_B(K * N, 1.0f); // Initialize B with 1.0
    std::vector<float> h_C(M * N, 0.0f); // Initialize C with 0.0

    // Device memory allocation
    float *d_A, *d_B, *d_C;
    size_t bytes_A = M * K * sizeof(float);
    size_t bytes_B = K * N * sizeof(float);
    size_t bytes_C = M * N * sizeof(float);

    cudaMalloc(&d_A, bytes_A);
    cudaMalloc(&d_B, bytes_B);
    cudaMalloc(&d_C, bytes_C);

    // Copy data from host to device
    cudaMemcpy(d_A, h_A.data(), bytes_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), bytes_B, cudaMemcpyHostToDevice);

    // Define block and grid sizes
    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);

    // Launch kernel
    matrixMulNaive<<<grid, block>>>(d_A, d_B, d_C, M, K, N);

    // Copy result back to host
    cudaMemcpy(h_C.data(), d_C, bytes_C, cudaMemcpyDeviceToHost);

    // Optional: Verify result
    bool correct = true;
    for (int i = 0; i < M * N; ++i) {
        if (h_C[i] != K * 1.0f) { // Since A and B are initialized to 1.0
            correct = false;
            break;
        }
    }
    if (correct)
        std::cout << "Naive Matrix Multiplication Successful!\n";
    else
        std::cout << "Naive Matrix Multiplication Failed!\n";

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
