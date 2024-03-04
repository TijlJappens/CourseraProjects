#include <iostream>
#include <cmath>

// CUDA kernel to add two vectors
__global__ void vectorAddition(const float* A, const float* B, float* C, int numElements) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < numElements) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    const int numElements = 1 << 20; // 2^20 elements

    // Allocate memory for host vectors
    float *h_A = new float[numElements];
    float *h_B = new float[numElements];
    float *h_C = new float[numElements];

    // Initialize host vectors with some values
    for (int i = 0; i < numElements; ++i) {
        h_A[i] = static_cast<float>(i);
        h_B[i] = static_cast<float>(i * 2);
    }

    // Allocate memory for device vectors
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, numElements * sizeof(float));
    cudaMalloc((void**)&d_B, numElements * sizeof(float));
    cudaMalloc((void**)&d_C, numElements * sizeof(float));

    // Copy host vectors to device
    cudaMemcpy(d_A, h_A, numElements * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, numElements * sizeof(float), cudaMemcpyHostToDevice);

    // Define grid and block size
    int blockSize = 256;
    int gridSize = (numElements + blockSize - 1) / blockSize;

    // Launch the vector addition kernel
    vectorAddition<<<gridSize, blockSize>>>(d_A, d_B, d_C, numElements);

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    // Copy result from device to host
    cudaMemcpy(h_C, d_C, numElements * sizeof(float), cudaMemcpyDeviceToHost);

    // Check for errors in kernel launch or execution
    cudaError_t cudaError = cudaGetLastError();
    if (cudaError != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(cudaError) << std::endl;
    }

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Free host memory
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    std::cout << "Reached end of program." << std::endl;
    // Do something with the result if needed

    return 0;
}
