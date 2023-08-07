#include "RedNeuronalSecuencial.cuh"

#include <stdio.h>
#include <stdlib.h>

#include <time.h>
#include <curand.h>
#include <curand_kernel.h>

#include <random>

using namespace std;

int main()
{
    srand(time(NULL));

    /*
    RedNeuronalSecuencial* r = new RedNeuronalSecuencial(4, new int[4] { 1024, 64, 64, 1024 }, new int[3] {1, 1, 1});

    float* de = new float[1024];
    float* ds = new float[1024];

    //std::default_random_engine generator;
    //std::normal_distribution<float> distribution(0.0, 1.0);
    for (int i = 0; i < 1024; i++) {
        //de[i] = distribution(generator);
        //ds[i] = distribution(generator);
        de[i] =  (2 * (((float)rand()) / RAND_MAX)) - 1;
        ds[i] = (2 * (((float)rand()) / RAND_MAX)) - 1;
    }

    r->entrenarRedMSE_SGD(0.03, 10, 1000, 1, 1, 1024, 1024, de, ds);
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

    const int nentradas = 2;
    const int nsalidas = 1;
    float tapren = 0.01;
    int nepochs = 3000;

    RedNeuronalSecuencial* r = new RedNeuronalSecuencial(5, new int[5] { nentradas, 10, 100, 10, nsalidas }, new int[4] { 1, 1, 1, 1 });

    const int nejemplos = 4;
    const int batch_size = 4;

    float* de = new float[nentradas * nejemplos] { -1, -1, -1, 1, 1, -1, 1, 1 };
    float* ds = new float[nsalidas * nejemplos] { 1, -1, -1, 1 };

    r->entrenarRedMSE_SGD(tapren, 100, nepochs, nejemplos, batch_size, nentradas, nsalidas, de, ds);

    float* res = r->propagacionHaciaDelante(4, nentradas, de);
    imprimirMatrizPorPantalla("", res, 4, nsalidas);
    delete res;
    delete r;

    /*
    RedNeuronalSecuencial* r = new RedNeuronalSecuencial("caca.data");

    float* de = new float[1024];
    float* ds = new float[1024];

    float* res = r->propagacionHaciaDelante(1, 1024, de);
    imprimirMatrizPorPantalla("", res, 1, 1024);
    delete res;
    delete r;
    */

    /*printf("\n\ncargo el archivo:\n");

    r = new RedNeuronalSecuencial("caca.data");

    res = r->propagacionHaciaDelante(4, 2, de);
    imprimirMatrizPorPantalla("", res, 4, 1);
    delete res;
    delete r;*/

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