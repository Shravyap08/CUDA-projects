// cpu_matrix_mul.cpp
#include <iostream>
#include <vector>
#include <chrono>

// CPU Matrix Multiplication
void matrixMulCPU(const std::vector<float>& A, const std::vector<float>& B, std::vector<float>& C, int M, int K, int N) {
    for(int i = 0; i < M; ++i){
        for(int j = 0; j < N; ++j){
            float sum = 0.0f;
            for(int e = 0; e < K; ++e){
                sum += A[i * K + e] * B[e * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

int main(){
    // Matrix dimensions
    int M = 1024;
    int K = 1024;
    int N = 1024;

    // Host memory allocation
    std::vector<float> h_A(M * K, 1.0f); // Initialize A with 1.0
    std::vector<float> h_B(K * N, 1.0f); // Initialize B with 1.0
    std::vector<float> h_C(M * N, 0.0f); // Initialize C with 0.0

    // Measure execution time
    auto start = std::chrono::high_resolution_clock::now();
    matrixMulCPU(h_A, h_B, h_C, M, K, N);
    auto end = std::chrono::high_resolution_clock::now();

    // Calculate duration
    std::chrono::duration<double> duration = end - start;
    std::cout << "CPU Matrix Multiplication Time: " << duration.count() << " seconds\n";

    // Optional: Verify result
    bool correct = true;
    for (int i = 0; i < M * N; ++i) {
        if (h_C[i] != K * 1.0f) { // Since A and B are initialized to 1.0
            correct = false;
            break;
        }
    }
    if (correct)
        std::cout << "CPU Matrix Multiplication Successful!\n";
    else
        std::cout << "CPU Matrix Multiplication Failed!\n";

    return 0;
}

