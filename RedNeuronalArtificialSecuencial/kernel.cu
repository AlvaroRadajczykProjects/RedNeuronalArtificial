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

    RedNeuronalSecuencial* r; 

    const int nentradas = 2;
    const int nsalidas = 1;
    float tapren = 0.0005;
    int nepochs = 20000;
    float* res;

    const int nejemplos = 4;
    const int batch_size = 4;

    float* de = new float[nentradas * nejemplos] { 0, 0, 0, 1, 1, 0, 1, 1 };
    float* ds = new float[nsalidas * nejemplos] { 0, 1, 1, 0 };

    r = new RedNeuronalSecuencial(4, new int[4] { nentradas, 10, 10, nsalidas }, new int[3] { 3, 3, 3 });

    r->entrenarRedMSE_Adam(tapren, 0.9, 0.999, 0.000000001, 500, nepochs, nejemplos, batch_size, nentradas, nsalidas, de, ds);

    r->exportarRedComoArchivo("caca.data");

    r->iniciarCublas();

    res = r->propagacionHaciaDelante(4, nentradas, de);
    imprimirMatrizPorPantalla("", res, 4, nsalidas);
    delete res;

    r->terminarCublas();

    delete r;

    return 0;
}