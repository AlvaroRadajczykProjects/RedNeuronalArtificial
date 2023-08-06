#include "MatrizDevice.cuh"

void multiplyMatrices(float* matA, int rA, int cA, float* matB, int rB, int cB, float* matC, int rC, int cC);

class RedNeuronalSecuencialRobusta {

	private:
		int tam_capa_entrada;
		int tam_capa_salida;
		int numero_capas_ocultas;
		int* tams_capas_ocultas;
		int funciones_capas_ocultas;
		int funcion_capa_salida;

		MatrizDevice** matrices_pesos = NULL;
		MatrizDevice** matrices_biases = NULL;

	public:
		RedNeuronalSecuencialRobusta(int tce, int tcs, int nco, int* tmsco, int fco, int fcs);
		~RedNeuronalSecuencialRobusta();
		int getTamCapaEntrada();
		int getTamCapaSalida();
		int getNumeroCapasOcultas();
		int* getTamsCapasOcultas();
		int getFuncionesCapasOcultas();
		int getFuncionCapaSalida();
		void mostrarBiasesYPesos();
		void copiarBiasesYPesos(MatrizDevice** biases, MatrizDevice** pesos);

		MatrizDevice* forwardPropagation(MatrizDevice* input);
		MatrizDevice* forwardPropagationTrainning(MatrizDevice* input);

};