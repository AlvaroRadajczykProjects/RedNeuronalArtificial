#include <stdio.h>
#include <stdlib.h>

#include <time.h>
#include <curand.h>
#include <curand_kernel.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define CUDA_CHECK(call)                                 \
    cudaError_t status = call;                           \
    if (status != cudaSuccess) {                         \
        fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(status));                \
        exit(EXIT_FAILURE);                                        \
    }                                                    \

#define CUDA_CHECK_SYNC(call)                                 \
    cudaError_t status = call;                           \
    if (status != cudaSuccess) {                         \
        fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(status));                \
        exit(EXIT_FAILURE);                                        \
    }                                               \
    cudaDeviceSynchronize();            \

void manageCUDAError(cudaError_t status);

curandGenerator_t crearGeneradorNumerosAleatoriosEnDistribucionNormal();
void generarNumerosAleatoriosEnDistribucionNormal(curandGenerator_t curandGenerator, float mean, float sdev, float* pointer, long long nelems);

//FORWARD

// A* B = C, A = MxN, B = NxP, C = MxP
__global__ void productoMatrices(const float* A, const float* B, float* C, int M, int N, int P);

//m-> [ m[i] + v for i = numero_filas(m) ]
__global__ void sumarCadaFilaMatrizVector(float* m, float* v, int nrows, int ncols);

//m-> [ [ funcion_sigmoide( m[i][j] ) for j = numero_columnas(m) ] for i = numero_filas(m) ]
__global__ void aplicarFuncionSigmoideCadaElementoMatriz(float* zl, float* al, int nrows, int ncols);

__global__ void aplicarFuncionTahnCadaElementoMatriz(float* zl, float* al, int nrows, int ncols);

__global__ void aplicarFuncionCosenoEspecialCadaElementoMatriz(float* zl, float* al, int nrows, int ncols);

__global__ void aplicarFuncionPReluCadaElementoMatriz(float* zl, float* al, int nrows, int ncols);

//ERROR CAPA OUTPUT

__global__ void aplicarDerivadaFuncionPerdidaMSECadaElementoPredY(int batch_size, int nvalssalida, float* pred_y, float* real_y);

__global__ void aplicarFuncionCosteMSE(int batch_size, int nvalssalida, const float* pred_y, const float* real_y, float* res);

//BACKWARD

//m-> [ [ derivada_funcion_sigmoide( m[i][j] ) for j = numero_columnas(m) ] for i = numero_filas(m) ]
__global__ void aplicarDerivadaFuncionSigmoideCadaElementoMatriz(float* m, int nrows, int ncols);

__global__ void aplicarDerivadaFuncionTahnCadaElementoMatriz(float* m, int nrows, int ncols);

__global__ void aplicarDerivadaFuncionCosenoEspecialCadaElementoMatriz(float* m, int nrows, int ncols);

__global__ void aplicarDerivadaFuncionPReluCadaElementoMatriz(float* m, int nrows, int ncols);

//T(idata) = odata
__global__ void matrizTraspuesta(float* odata, float* idata, int nrows, int ncols);

//APPLICAR VECTOR GRADIENTE

//A = A + B; dim(A) = dim(B)
__global__ void sumarAMatrizAMatrizB(float* a, float* b, int nrows, int ncols);

//A = A * B (cada elemento, no es un producto matricial); dim(A) = dim(B)
__global__ void multiplicarAMatrizAMatrizB(float* a, float* b, int nrows, int ncols);

//m-> [ [ m[i][j]*x for j = numero_columnas(m) ] for i = numero_filas(m) ]
__global__ void multiplicarCadaElementoMatriz(float* m, float x, int nrows, int ncols);

__global__ void sumarACadaElementoVectorColumnaMatriz(const float* matrix, float* columnSums, int numRows, int numCols);