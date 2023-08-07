#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <assert.h>

// CUDA runtime
#include <cuda_runtime.h>
#include <cublas_v2.h>

void initializeMatrix(float* matrix, int rows, int cols) {
    for (int i = 0; i < rows * cols; ++i) {
        matrix[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}

void imprimirVectorIntPorPantalla(char* texto_mostrar, float vector[], int inicio, int fin) {
    printf("\n%s [ ", texto_mostrar);
    for (int i = inicio; i < fin; i++) {
        printf("%.0f", vector[i]);
        if (i < fin - 1) { printf(","); }
        printf(" ");
    }
    printf("]");
}

void imprimirMatrizPorPantalla(char* texto_mostrar, float matriz[], int n_filas, int n_columnas) {
    printf("\n%s\n", texto_mostrar);
    for (int i = 0; i < n_filas; i++) {
        imprimirVectorIntPorPantalla(" ", matriz, i * n_columnas, i * n_columnas + n_columnas);
    }
    printf("\n");
}

int main() {
    /*
    const int m = 3;
    const int n = 6;
    const int k = 9;
    
    // Initialize cuBLAS
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Allocate memory on the host for input matrices A and B
    float* h_A = new float[m * k];
    float* h_B = new float[k * n];  // Transposed B matrix

    // Initialize input matrices A and B
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < k; j++) {
            h_A[i * k + j] = (j+1);
        }
    }

    for (int i = 0; i < k; i++) {
        for (int j = 0; j < n; j++) {
            h_B[i * n + j] = (j + 1);
        }
    }

    imprimirMatrizPorPantalla("A", h_A, m, k);
    imprimirMatrizPorPantalla("B", h_B, k, n);

    // Allocate memory on the device for input matrices A and B
    float* d_A, * d_B, * d_C;
    cudaMalloc((void**)&d_A, m * k * sizeof(float));
    cudaMalloc((void**)&d_B, n * k * sizeof(float));
    cudaMalloc((void**)&d_C, m * n * sizeof(float));

    // Copy input matrices A and B from host to device
    cudaMemcpy(d_A, h_A, m * k * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, n * k * sizeof(float), cudaMemcpyHostToDevice);

    // Perform matrix multiplication using cuBLAS with transposed B matrix
    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, d_B, n, d_A, k, &beta, d_C, n);

    // Allocate memory on the host for the result matrix C
    float* h_C = new float[m * n];

    // Copy the result matrix C from device to host
    cudaMemcpy(h_C, d_C, m * n * sizeof(float), cudaMemcpyDeviceToHost);

    // Print the result matrix C (optional)
    std::cout << "Result matrix C:" << std::endl;
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            std::cout << h_C[i * n + j] << " ";
        }
        std::cout << std::endl;
    }

    // Clean up
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cublasDestroy(handle);
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    return 0;
    */

    const int rows = 3;
    const int cols = 6;

    // Initialize cuBLAS
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Allocate memory on the host for input matrix A
    float* h_A = new float[rows * cols];

    // Initialize input matrix A
    initializeMatrix(h_A, rows, cols);

    std::cout << "matrix A:" << std::endl;
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::cout << h_A[i * cols + j] << " ";
        }
        std::cout << std::endl;
    }

    // Allocate memory on the device for input matrix A and transposed matrix B
    float* d_A, * d_B;
    cudaMalloc((void**)&d_A, rows * cols * sizeof(float));
    cudaMalloc((void**)&d_B, rows * cols * sizeof(float));

    // Copy input matrix A from host to device
    cudaMemcpy(d_A, h_A, rows * cols * sizeof(float), cudaMemcpyHostToDevice);

    // Transpose matrix A using cuBLAS
    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, rows, cols, &alpha, d_A, cols, &beta, d_A, rows, d_B, rows);

    // Allocate memory on the host for the transposed matrix B
    float* h_B = new float[cols * rows];

    // Copy the transposed matrix B from device to host
    cudaMemcpy(h_B, d_B, cols * rows * sizeof(float), cudaMemcpyDeviceToHost);

    // Print the transposed matrix B (optional)
    std::cout << "Transposed matrix B:" << std::endl;
    for (int i = 0; i < cols; ++i) {
        for (int j = 0; j < rows; ++j) {
            std::cout << h_B[i * rows + j] << " ";
        }
        std::cout << std::endl;
    }

    // Clean up
    cudaFree(d_A);
    cudaFree(d_B);
    cublasDestroy(handle);
    delete[] h_A;
    delete[] h_B;

    return 0;

}
