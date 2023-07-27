#include "RedNeuronalSecuencial.cuh"

#include <stdio.h>
#include <stdlib.h>

#include <time.h>
#include <curand.h>
#include <curand_kernel.h>

#include <iostream>

#include <windows.h>

using namespace std;

int main()
{
    srand(time(NULL));

    /*

    float* m = new float[10*34];
    float* n = new float[34*10];
    float* d_m = 0;
    float* d_n = 0;

    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 34; j++) {
            m[i * 34 + j] = i + 1;
        }
    }

    manageCUDAError(cudaMalloc(&d_m, 10 * 34 * sizeof(float) ));
    manageCUDAError(cudaMalloc(&d_n, 34 * 10 * sizeof(float)));
    manageCUDAError(cudaDeviceSynchronize());

    manageCUDAError(cudaMemcpy(d_m, m, 10 * 34 * sizeof(float), cudaMemcpyHostToDevice));
    manageCUDAError(cudaDeviceSynchronize());

    matrizTraspuestaDevice(d_n, d_m, 10, 34);
    
    manageCUDAError(cudaMemcpy(n, d_n, 34 * 10 * sizeof(float), cudaMemcpyDeviceToHost));
    manageCUDAError(cudaDeviceSynchronize());

    imprimirMatrizPorPantalla("", m, 10, 34);

    printf("\n================================================================================\n");

    imprimirMatrizPorPantalla( "", n, 34, 10 );

    */

    /*
    int dx = 33;
    int dy = 10;

    float* a = new float[dx * dy];

    float* b = new float[dy * dx];

    for (int i = 0; i < dx * dy; i++) { a[i] = 1; b[i] = 1; }

    float* c = new float[dx * dx];

    float* d_a = 0;
    float* d_b = 0;
    float* d_c = 0;

    manageCUDAError(cudaMalloc(&d_a, dx * dy * sizeof(float)));
    manageCUDAError(cudaMalloc(&d_b, dx * dy * sizeof(float)));
    manageCUDAError(cudaMalloc(&d_c, dx * dx * sizeof(float)));
    manageCUDAError(cudaDeviceSynchronize());

    manageCUDAError(cudaMemcpy(d_a, a, dx * dy * sizeof(float), cudaMemcpyHostToDevice));
    manageCUDAError(cudaMemcpy(d_b, b, dx * dy * sizeof(float), cudaMemcpyHostToDevice));
    manageCUDAError(cudaDeviceSynchronize());

    productoMatricesDevice(d_a, d_b, d_c, dx, dy, dx);

    manageCUDAError(cudaMemcpy(c, d_c, dx * dx * sizeof(float), cudaMemcpyDeviceToHost));
    manageCUDAError(cudaDeviceSynchronize());

    imprimirMatrizPorPantalla("", c, dx, dx);
    */

    float** weights = new float* [2] { new float[4] {-6.167499407984588, -4.607113545803228, -6.215967356057629, -4.616921683516318}, new float[2] {-9.475190768811604, 9.26662990483002} };
    float** biases = new float* [2] { new float[2] {2.48990533590185, 6.857492627195812}, new float[1] {-4.361658270629812} };
    float* de = new float[8] { 0, 0, 0, 1, 1, 0, 1, 1 };
    float* ds = new float[4] { 1, 0, 0, 1 };

    //for (int i = 0; i < 300; i++) {
        RedNeuronalSecuencial* r = new RedNeuronalSecuencial(7, new int[7] { 2, 10, 33, 65, 33, 10, 1 }, NULL);
        //r->copiarPesosHostDevice(weights, biases);
        r->entrenarRedMSE_SGD(0.3, 10000, 4, 4, 2, 1, de, ds);
        float* res = r->propagacionHaciaDelante(4, 2, de);
        imprimirMatrizPorPantalla("", res, 4, 1);
        delete res;
        delete r;
    //}

    return 0;
}