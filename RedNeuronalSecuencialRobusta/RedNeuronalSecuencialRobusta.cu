#include "RedNeuronalSecuencialRobusta.cuh"

bool compare_float(float x, float y, float epsilon = 0.01f) {
	if (fabs(x - y) < epsilon)
		return true; //they are same
	return false; //they are not same
}

const char* boolToString(bool b) {
	return b ? "true" : "false";
}

void checkDifferences(float* host, float* device, int nr, int nc) {
	bool problemas = false;
	for (int i = 0; i < nr; i++) {
		for (int j = 0; j < nc; j++) {
			if ( !isfinite( host[i * nc + j] ) && !isfinite( device[i * nc + j] ) ) {
				printf("\npos %d %d host y device numero no finito, vhost = %f, vdev = %f", i, j, host[i * nc + j], device[i * nc + j]);
				problemas = true;
			} else if ( !isfinite(device[i * nc + j] ) ) {
				printf("\npos %d %d device numero no finito, vhost = %f, vdev = %f", i, j, host[i * nc + j], device[i * nc + j]);
				problemas = true;
			} else if ( !compare_float( host[i * nc + j], device[i * nc + j], 0.01f ) ) {
				printf("\npos %d %d device es bastante diferente a host, vhost = %f, vdev = %f", i, j, host[i * nc + j], device[i * nc + j]);
				problemas = true;
			}
		}
	}
	if (!problemas) { printf("\nno hubo problemas en esta operacion\n"); }
	else {
		//imprimirMatrizPorPantalla("host: ", host, nr, nc);
		//imprimirMatrizPorPantalla("device: ", device, nr, nc);
		printf("\nmirar arriba si hay mensajes de error\n");
	}
}

void multiplyMatrices(float* matA, int rA, int cA, float* matB, int rB, int cB, float* matC, int rC, int cC) {
	for (int i = 0; i < rA; i++) {
		for (int j = 0; j < cB; j++) {
			float sum = 0.0;
			for (int k = 0; k < rB; k++)
				sum = sum + matA[i * cA + k] * matB[k * cB + j];
			matC[i * cC + j] = sum;
		}
	}
}

void sumarACadaFilaMatrizVector( float*m, float* v, int nr, int nc ) {
	for (int i = 0; i < nr; i++) {
		for (int j = 0; j < nc; j++) {
			m[i * nr + j] += v[i];
		}
	}
}

void aplicarFuncionSigmoide(float* m, int nr, int nc) {
	for (int i = 0; i < nr; i++) {
		for (int j = 0; j < nc; j++) {
			//if (m[i * nc + j] < 0) { m[i * nc + j] = m[i * nc + j] * 0.01; }
			//else{ m[i * nc + j] = m[i * nc + j] * 1; }
			/*float res = (exp(m[i * nc + j]) - exp(-m[i * nc + j])) / (float)(exp(m[i * nc + j]) + exp(-m[i * nc + j]));
			if (!isfinite(res)) {
				printf("\nantes: %f, despues: %f | %f  %f\n", m[i * nc + j], res, exp(m[i * nc + j]), exp(-m[i * nc + j]));
			}
			m[i * nc + j] = res;
			*/
			//m[i * nc + j] = 1 / (1 + exp(-m[i * nc + j]));
			//tanhf ( float  x )
			m[i * nc + j] = tanh(m[i * nc + j]);
		}
	}
}

RedNeuronalSecuencialRobusta::RedNeuronalSecuencialRobusta(int tce, int tcs, int nco, int* tmsco, int fco, int fcs) {
	tam_capa_entrada = tce;
	tam_capa_salida = tcs;
	numero_capas_ocultas = nco;
	tams_capas_ocultas = tmsco;
	funciones_capas_ocultas = fco;
	funcion_capa_salida = fcs;

	matrices_pesos = (MatrizDevice**)malloc((numero_capas_ocultas + 1) * sizeof(MatrizDevice*));
	matrices_pesos[0] = new MatrizDevice(tam_capa_entrada, tams_capas_ocultas[0]);
	for (int i = 0; i < numero_capas_ocultas - 1; i++) { matrices_pesos[i + 1] = new MatrizDevice(tams_capas_ocultas[i], tams_capas_ocultas[i + 1]); }
	matrices_pesos[numero_capas_ocultas] = new MatrizDevice(tams_capas_ocultas[numero_capas_ocultas - 1], tam_capa_salida);

	matrices_biases = (MatrizDevice**)malloc((numero_capas_ocultas + 1) * sizeof(MatrizDevice*));
	for (int i = 0; i < numero_capas_ocultas; i++) { matrices_biases[i] = new MatrizDevice(1, tams_capas_ocultas[i]); }
	matrices_biases[numero_capas_ocultas] = new MatrizDevice(1, tam_capa_salida);

	for (int i = 0; i < numero_capas_ocultas + 1; i++) { 
		matrices_pesos[i]->setAllDataValuesAsRandomNormalDistribution(0.0, 1.0);
		matrices_biases[i]->setAllDataValuesToZero();
	}
}

RedNeuronalSecuencialRobusta::~RedNeuronalSecuencialRobusta() {
	if(matrices_pesos != NULL){
		for (int i = 0; i < numero_capas_ocultas + 1; i++) {
			delete matrices_pesos[i];
			matrices_pesos[i] = 0;
		}
		free(matrices_pesos);
		matrices_pesos = NULL;
	}

	if (matrices_biases != NULL) {
		for (int i = 0; i < numero_capas_ocultas + 1; i++) {
			delete matrices_biases[i];
			matrices_biases[i] = 0;
		}
		free(matrices_biases);
		matrices_biases = NULL;
	}
}

int RedNeuronalSecuencialRobusta::getTamCapaEntrada() {
	return tam_capa_entrada;
}

int RedNeuronalSecuencialRobusta::getTamCapaSalida() {
	return tam_capa_salida;
}

int RedNeuronalSecuencialRobusta::getNumeroCapasOcultas() {
	return numero_capas_ocultas;
}

int* RedNeuronalSecuencialRobusta::getTamsCapasOcultas() {
	return tams_capas_ocultas;
}

int RedNeuronalSecuencialRobusta::getFuncionesCapasOcultas() {
	return funciones_capas_ocultas;
}

int RedNeuronalSecuencialRobusta::getFuncionCapaSalida() {
	return funcion_capa_salida;
}

void RedNeuronalSecuencialRobusta::mostrarBiasesYPesos() {
	for (int i = 0; i < numero_capas_ocultas + 1; i++) {
		printf("\ncapa %d\n", i + 1);
		matrices_biases[i]->show("bias");
		matrices_pesos[i]->show("weights");
	}
}

void RedNeuronalSecuencialRobusta::copiarBiasesYPesos(MatrizDevice** biases, MatrizDevice** pesos) {
	for (int i = 0; i < numero_capas_ocultas + 1; i++) {
		matrices_biases[i]->copyFromMatrizDevice(biases[i]);
		matrices_pesos[i]->copyFromMatrizDevice(pesos[i]);
	}
}

MatrizDevice* RedNeuronalSecuencialRobusta::forwardPropagation(MatrizDevice* input) {

	MatrizDevice* rant = new MatrizDevice(input->getNumRows(), tams_capas_ocultas[0]);
	MatrizDevice* ract;

	float* mr = new float[input->getNumRows() * matrices_pesos[0]->getNumCols()];
	float* vr = matrices_biases[0]->getDataHost();

	printf("\nPRUEBA EN LA CAPA DE ENTRADA:\n");

	printf("\nproducto de matrices:\n");

	multiplyMatrices(
		input->getDataHost(), input->getNumRows(), input->getNumCols(),
		matrices_pesos[0]->getDataHost(), matrices_pesos[0]->getNumRows(), matrices_pesos[0]->getNumCols(),
		mr, input->getNumRows(), matrices_pesos[0]->getNumCols()
	);
	input->copyMatrixProductToMatrix(matrices_pesos[0], rant);

	checkDifferences(mr, rant->getDataHost(), input->getNumRows(), matrices_pesos[0]->getNumCols());

	printf("\nsuma con vector bias:\n");

	sumarACadaFilaMatrizVector(mr, vr, input->getNumRows(), matrices_pesos[0]->getNumCols());
	rant->sumEachRowVector(matrices_biases[0]);

	checkDifferences(mr, rant->getDataHost(), input->getNumRows(), matrices_pesos[0]->getNumCols());

	printf("\naplicar funcion:\n");

	aplicarFuncionSigmoide(mr, input->getNumRows(), matrices_pesos[0]->getNumCols());
	rant->applyFunction(funciones_capas_ocultas);

	checkDifferences(mr, rant->getDataHost(), input->getNumRows(), matrices_pesos[0]->getNumCols());

	delete vr;
	delete mr;

	for (int i = 1; i < numero_capas_ocultas; i++) {
		ract = new MatrizDevice(input->getNumRows(), tams_capas_ocultas[i]);

		mr = new float[input->getNumRows() * tams_capas_ocultas[i]];
		vr = matrices_biases[i]->getDataHost();

		printf("\nPRUEBA EN LA CAPA OCULTA %d:\n", i);

		printf("\nproducto de matrices:\n");

		multiplyMatrices(
			rant->getDataHost(), rant->getNumRows(), rant->getNumCols(),
			matrices_pesos[i]->getDataHost(), matrices_pesos[i]->getNumRows(), matrices_pesos[i]->getNumCols(),
			mr, rant->getNumRows(), matrices_pesos[i]->getNumCols()
		);
		rant->copyMatrixProductToMatrix(matrices_pesos[i], ract);
		delete rant;

		checkDifferences(mr, ract->getDataHost(), input->getNumRows(), matrices_pesos[i]->getNumCols());

		rant = ract;

		printf("\nsuma con vector bias:\n");

		sumarACadaFilaMatrizVector(mr, vr, input->getNumRows(), matrices_pesos[i]->getNumCols());
		rant->sumEachRowVector(matrices_biases[i]);

		checkDifferences(mr, rant->getDataHost(), input->getNumRows(), matrices_pesos[i]->getNumCols());

		printf("\naplicar funcion:\n");

		rant->applyFunction(funciones_capas_ocultas);
		aplicarFuncionSigmoide(mr, input->getNumRows(), matrices_pesos[i]->getNumCols());

		checkDifferences(mr, rant->getDataHost(), input->getNumRows(), matrices_pesos[i]->getNumCols());

		delete mr;
		delete vr;
	}

	ract = new MatrizDevice(input->getNumRows(), tam_capa_salida);

	mr = new float[input->getNumRows() * tams_capas_ocultas[numero_capas_ocultas]];
	vr = matrices_biases[numero_capas_ocultas]->getDataHost();

	printf("\nPRUEBA EN LA CAPA DE SALIDA\n");

	printf("\nproducto de matrices:\n");

	multiplyMatrices(
		rant->getDataHost(), rant->getNumRows(), rant->getNumCols(),
		matrices_pesos[numero_capas_ocultas]->getDataHost(), matrices_pesos[numero_capas_ocultas]->getNumRows(), matrices_pesos[numero_capas_ocultas]->getNumCols(),
		mr, rant->getNumRows(), matrices_pesos[numero_capas_ocultas]->getNumCols()
	);
	rant->copyMatrixProductToMatrix(matrices_pesos[numero_capas_ocultas], ract);
	delete rant;

	checkDifferences(mr, ract->getDataHost(), input->getNumRows(), matrices_pesos[numero_capas_ocultas]->getNumCols());

	printf("\nsuma con vector bias:\n");

	sumarACadaFilaMatrizVector(mr, vr, input->getNumRows(), matrices_pesos[numero_capas_ocultas]->getNumCols());
	ract->sumEachRowVector(matrices_biases[numero_capas_ocultas]);

	checkDifferences(mr, ract->getDataHost(), input->getNumRows(), matrices_pesos[numero_capas_ocultas]->getNumCols());

	printf("\naplicar funcion:\n");

	ract->applyFunction(funcion_capa_salida);
	aplicarFuncionSigmoide(mr, input->getNumRows(), matrices_pesos[numero_capas_ocultas]->getNumCols());

	checkDifferences(mr, ract->getDataHost(), input->getNumRows(), matrices_pesos[numero_capas_ocultas]->getNumCols());

	delete mr;
	delete vr;

	return ract;
}

MatrizDevice* RedNeuronalSecuencialRobusta::forwardPropagationTrainning(MatrizDevice* input) {
	return NULL;
}