#include "RedNeuronalSecuencial.cuh"

#include <stdio.h>
#include <stdlib.h>

#include <time.h>
#include <curand.h>
#include <curand_kernel.h>

using namespace std;

int main()
{
    srand(time(NULL));

    /*
    RedNeuronalSecuencial* r = new RedNeuronalSecuencial(4, new int[4] { 1024, 64, 64, 1024 }, new int[3] {3, 3, 3});

    float* de = new float[1024];
    float* ds = new float[1024];

    r->entrenarRedMSE_SGD(0.03, 10000, 1, 1, 1024, 1024, de, ds);
    float* res = r->propagacionHaciaDelante(1, 1024, de);
    imprimirMatrizPorPantalla("", res, 1, 1024);
    delete res;

    r->exportarRedComoArchivo("caca.data");

    delete r;

    printf("\n\ncargo el archivo:\n");

    r = new RedNeuronalSecuencial("caca.data");

    res = r->propagacionHaciaDelante(1, 1024, de);
    imprimirMatrizPorPantalla("", res, 1, 1024);
    delete res;
    delete r;
    */

    /*
    RedNeuronalSecuencial* r = new RedNeuronalSecuencial("caca.data");

    float* de = new float[1024];
    float* ds = new float[1024];

    float* res = r->propagacionHaciaDelante(1, 1024, de);
    imprimirMatrizPorPantalla("", res, 1, 1024);
    delete res;
    delete r;
    */

    RedNeuronalSecuencial* r = new RedNeuronalSecuencial(4, new int[4] { 2, 10, 10, 1 }, new int[3] {3, 3, 3});

    float* de = new float[8] { 0, 0, 0, 1, 1, 0, 1, 1 };
    float* ds = new float[4] { 1, 0, 0, 1 };

    r->entrenarRedMSE_SGD(0.03, 10000, 4, 4, 2, 1, de, ds);
    float* res = r->propagacionHaciaDelante(4, 2, de);
    imprimirMatrizPorPantalla("", res, 4, 1);
    delete res;

    r->exportarRedComoArchivo("caca.data");

    delete r;

    printf("\n\ncargo el archivo:\n");

    r = new RedNeuronalSecuencial("caca.data");

    res = r->propagacionHaciaDelante(4, 2, de);
    imprimirMatrizPorPantalla("", res, 4, 1);
    delete res;
    delete r;

    /*
    RedNeuronalSecuencial* r = new RedNeuronalSecuencial("caca.data");

    float* de = new float[8] { 0, 0, 0, 1, 1, 0, 1, 1 };
    float* ds = new float[4] { 1, 0, 0, 1 };

    float* res = r->propagacionHaciaDelante(4, 2, de);
    imprimirMatrizPorPantalla("", res, 4, 1);
    delete res;
    delete r;
    */

    return 0;
}