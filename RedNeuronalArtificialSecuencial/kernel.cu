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

    float** weights = new float* [2] { new float[4] {-6.167499407984588, -4.607113545803228, -6.215967356057629, -4.616921683516318}, new float[2] {-9.475190768811604, 9.26662990483002} };
    float** biases = new float* [2] { new float[2] {2.48990533590185, 6.857492627195812}, new float[1] {-4.361658270629812} };
    float* de = new float[8] { 0, 0, 0, 1, 1, 0, 1, 1 };
    float* ds = new float[4] { 0, 1, 1, 0 };

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