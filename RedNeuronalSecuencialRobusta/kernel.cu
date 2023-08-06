#include <stdio.h>
#include <math.h>

#include "RedNeuronalSecuencialRobusta.cuh"

// /*
int main( int argc, char** argv ) {

    srand(time(NULL));
    
    MatrizDevice* entrada = new MatrizDevice(4, 2, new float[8] {
        0, 0,
        0, 1,
        1, 0,
        1, 1
    });

    //MatrizDevice** biases = new MatrizDevice * [2] { new MatrizDevice(1, 2, new float[4] {-2.9644369720979182, 2.731554195035574}), new MatrizDevice(1, 1, new float[2] {3.7210060097287063}) };
    //MatrizDevice** pesos = new MatrizDevice * [2] { new MatrizDevice(2, 2, new float[4] {-5.555248698525353, -5.419452574610639, 5.414225066183912, 5.60560978641461}), new MatrizDevice(2, 1, new float[2] {8.317090817819748, -7.961249848208193}) };
    
    RedNeuronalSecuencialRobusta* red = new RedNeuronalSecuencialRobusta(2, 1, 5, new int[5] { 10, 100, 1000, 100, 10 }, 1, 1);
    
    //red->copiarBiasesYPesos(biases, pesos);
    //red->mostrarBiasesYPesos();

    MatrizDevice* resultado = red->forwardPropagation(entrada);
    resultado->show("propagation");
    delete resultado;

    delete red;

    /*float* m1 = new float[4 * 2] {
        1, 2, 3, 4,
        5, 6, 7, 8,
    };

    float* m2 = new float[2 * 4] {
        1, 2,
        3, 4,
        5, 6,
        7, 8
    };

    float* mr = new float[4 * 4];

    multiplyMatrices(m1, 4, 2, m2, 2, 4, mr, 4, 4);

    imprimirMatrizPorPantalla("", mr, 4, 4);*/

    return 0;
}

// */