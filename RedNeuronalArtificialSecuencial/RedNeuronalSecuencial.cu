#include "RedNeuronalSecuencial.cuh"

void imprimirVectorIntPorPantalla(char* texto_mostrar, float vector[], int inicio, int fin) {
	printf("\n%s [ ", texto_mostrar);
	for (int i = inicio; i < fin; i++) {
		printf("%.8f", vector[i]);
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

float vmax(float a, float b) {
	return a > b ? a : b;
}

dim3 dim3Ceil(float x, float y) {
	return dim3((int)ceil(x), (int)ceil(y));
}

const void aplicarFuncion(int id, float* zl, float* al, int nfilas, int ncolumnas) {
	if (id == 0) { aplicarFuncionSigmoideCadaElementoMatriz << < dim3Ceil(nfilas / (float)32, ncolumnas / (float)32), dim3(32, 32) >> > (zl, al, nfilas, ncolumnas); }
	else if (id == 1) { aplicarFuncionTahnCadaElementoMatriz << < dim3Ceil(nfilas / (float)32, ncolumnas / (float)32), dim3(32, 32) >> > (zl, al, nfilas, ncolumnas); }
	else if (id == 2) { aplicarFuncionCosenoEspecialCadaElementoMatriz << < dim3Ceil(nfilas / (float)32, ncolumnas / (float)32), dim3(32, 32) >> > (zl, al, nfilas, ncolumnas); }
	else if (id == 3) { aplicarFuncionPReluCadaElementoMatriz << < dim3Ceil(nfilas / (float)32, ncolumnas / (float)32), dim3(32, 32) >> > (zl, al, nfilas, ncolumnas); }
	cudaDeviceSynchronize();
}

const void aplicarDerivadaFuncion(int id, float* m, int nfilas, int ncolumnas) {
	if (id == 0) { aplicarDerivadaFuncionSigmoideCadaElementoMatriz << < dim3Ceil(nfilas / (float)32, ncolumnas / (float)32), dim3(32, 32) >> > (m, nfilas, ncolumnas); }
	else if (id == 1){ aplicarDerivadaFuncionTahnCadaElementoMatriz << < dim3Ceil(nfilas / (float)32, ncolumnas / (float)32), dim3(32, 32) >> > (m, nfilas, ncolumnas); }
	else if (id == 2) { aplicarDerivadaFuncionCosenoEspecialCadaElementoMatriz << < dim3Ceil(nfilas / (float)32, ncolumnas / (float)32), dim3(32, 32) >> > (m, nfilas, ncolumnas); }
	else if (id == 3) { aplicarDerivadaFuncionPReluCadaElementoMatriz << < dim3Ceil(nfilas / (float)32, ncolumnas / (float)32), dim3(32, 32) >> > (m, nfilas, ncolumnas); }
	cudaDeviceSynchronize();
}

const void productoMatricesDevice(float* a, float* b, float* c, int m, int n, int p) {
	productoMatrices <<< dim3Ceil((p + 32 - 1) / (float)32, (m + 32 - 1) / (float)32), dim3(32, 32) >> > (a, b, c, m, n, p);
	cudaDeviceSynchronize();
}

//T(idata) = odata
const void matrizTraspuestaDevice(float* odata, float* idata, int m, int n) {
	int dimension = (int)vmax(m, n);
	matrizTraspuesta << < dim3Ceil(dimension / (float)32, dimension / (float)32), dim3(32, 32) >> > (odata, idata, m, n);
	cudaDeviceSynchronize();
}

void mostarMatrizDevice( char* com, float* p, int m, int n ) {
	float* ph = new float[m * n];
	cudaMemcpy(ph, p, m*n*sizeof(float), cudaMemcpyDeviceToHost);
	imprimirMatrizPorPantalla(com, ph, m, n);
	delete ph;
}

RedNeuronalSecuencial::RedNeuronalSecuencial(int nc, int* dc, int* fc) {
	numero_capas = nc;
	dimensiones_capas = dc;
	funciones_capas = fc;

	cargarEnDevice(true);
}

RedNeuronalSecuencial::RedNeuronalSecuencial(const char* nombre_archivo) {
	
	unsigned int nbytes = 0;
	
	char* cargar = leerArchivoYCerrar(nombre_archivo, &nbytes);

	unsigned int nnumeros = nbytes / 4;

	float* array = (float*) cargar;

	numero_capas = ((int*) array)[0];

	unsigned int offset = 1;

	dimensiones_capas = new int[numero_capas];
	funciones_capas = new int[numero_capas-1];

	for (int i = 0; i < numero_capas; i++) {
		dimensiones_capas[i] = ((int*)array)[offset];
		offset += 1;
	}

	for (int i = 0; i < numero_capas - 1; i++) {
		funciones_capas[i] = ((int*)array)[offset];
		offset += 1;
	}

	GestorPunteroPunteroFloatHost gestor_host_bias_vectors(numero_capas - 1, getCopiaDimensionesCapasRed());
	GestorPunteroPunteroFloatHost gestor_host_weight_matrices(numero_capas - 1, getCopiaDimensionesMatricesRed());

	float** host_bias_vectors = gestor_host_bias_vectors.getPunteroPunteroHost();
	float** host_weight_matrices = gestor_host_weight_matrices.getPunteroPunteroHost();

	for (int i = 1; i < numero_capas; i++) {
		for (int j = 0; j < dimensiones_capas[i]; j++) {
			host_bias_vectors[i - 1][j] = ((float*)array)[offset];
			offset += 1;
		}
	}

	for (int i = 0; i < numero_capas - 1; i++) {
		for (int j = 0; j < dimensiones_capas[i] * dimensiones_capas[i + 1]; j++) {
			host_weight_matrices[i][j] = ((float*)array)[offset];
			offset += 1;
		}
	}

	cargarEnDevice(false);
	copiarPesosHostDevice(host_weight_matrices, host_bias_vectors);
}

RedNeuronalSecuencial::~RedNeuronalSecuencial() {

	if (device_bias_vectors != NULL) { delete device_bias_vectors; device_bias_vectors = NULL; }
	if (device_weight_matrices != NULL) { delete device_weight_matrices; device_weight_matrices = NULL; }

	if (device_forward_zl != NULL) { delete device_forward_zl; device_forward_zl = NULL; }
	if (device_forward_al != NULL) { delete device_forward_al; device_forward_al = NULL; }

	if (device_err_bias_m != NULL) { delete device_err_bias_m; device_err_bias_m = NULL; }
	if (device_err_weight_m != NULL) { delete device_err_weight_m; device_err_weight_m = NULL; }
	if (device_err_bias_v != NULL) { delete device_err_bias_v; device_err_bias_v = NULL; }
	if (device_err_weight_v != NULL) { delete device_err_weight_v; device_err_weight_v = NULL; }

	if (device_batch_input != NULL) { cudaFree(device_batch_input); device_batch_input = NULL; }
	if (device_batch_output != NULL) { cudaFree(device_batch_output); device_batch_output = NULL; }
	if (temp_matr_traspose != NULL) { cudaFree(temp_matr_traspose); temp_matr_traspose = NULL; }

	if (dimensiones_capas != NULL) { free(dimensiones_capas); dimensiones_capas = NULL; }
	if (funciones_capas != NULL) { free(funciones_capas); funciones_capas = NULL; }

	cudaDeviceSynchronize();

}

int RedNeuronalSecuencial::getNumeroCapas() {
	return numero_capas;
}

int* RedNeuronalSecuencial::getDimensionesCapas() {
	return dimensiones_capas;
}

int* RedNeuronalSecuencial::getFuncionesCapas() {
	return funciones_capas;
}

int* RedNeuronalSecuencial::getCopiaDimensionesCapasRed() {
	int* copia_dimensiones_capas = new int[numero_capas - 1];
	for (int i = 0; i < numero_capas - 1; i++) { copia_dimensiones_capas[i] = dimensiones_capas[i + 1]; }
	return copia_dimensiones_capas;
}

int* RedNeuronalSecuencial::getCopiaDimensionesMatricesRed() {
	int* dimensiones_matrices = new int[numero_capas - 1];
	for (int i = 0; i < numero_capas - 1; i++) { dimensiones_matrices[i] = dimensiones_capas[i] * dimensiones_capas[i + 1]; }
	return dimensiones_matrices;
}

void RedNeuronalSecuencial::exportarRedComoArchivo(const char* nombre_archivo) {

	GestorPunteroPunteroFloatHost gestor_host_bias_vectors(numero_capas - 1, getCopiaDimensionesCapasRed());
	GestorPunteroPunteroFloatHost gestor_host_weight_matrices(numero_capas - 1, getCopiaDimensionesMatricesRed());

	float** host_bias_vectors = gestor_host_bias_vectors.getPunteroPunteroHost();
	float** host_weight_matrices = gestor_host_weight_matrices.getPunteroPunteroHost();

	device_bias_vectors->copiarDeviceAHost(host_bias_vectors);
	device_weight_matrices->copiarDeviceAHost(host_weight_matrices);

	unsigned int numero = 1 + numero_capas + (numero_capas - 1);
	for (int i = 1; i < numero_capas; i++) { numero += dimensiones_capas[i]; }
	for (int i = 1; i < numero_capas; i++) { numero += dimensiones_capas[i] * dimensiones_capas[i-1]; }

	float* array = (float*) malloc(numero * sizeof(float));
	((int*)array)[0] = numero_capas;

	unsigned int offset = 1;

	for (int i = 0; i < numero_capas; i++) {
		((int*)array)[offset] = dimensiones_capas[i];
		offset += 1;
	}

	for (int i = 0; i < numero_capas - 1; i++) {
		((int*)array)[offset] = funciones_capas[i];
		offset += 1;
	}

	for (int i = 1; i < numero_capas; i++) {
		for (int j = 0; j < dimensiones_capas[i]; j++) {
			array[offset] = host_bias_vectors[i - 1][j];
			offset += 1;
		}
	}

	for (int i = 0; i < numero_capas - 1; i++) {
		for (int j = 0; j < dimensiones_capas[i] * dimensiones_capas[i + 1]; j++) {
			array[offset] = host_weight_matrices[i][j];
			offset += 1;
		}
	}

	char* buffer = (char*) array;

	crearArchivoEscribirYCerrar(nombre_archivo, numero * sizeof(float), buffer);

	cudaFree(array);
	array = NULL;

	//se limpian solos los punteros, ojo también, no limpies host_bias_vectors ni host_weight_matrices en el destructor de esta clase
}

void RedNeuronalSecuencial::copiarPesosHostDevice(float** host_pesos, float** host_biases) {
	device_bias_vectors->copiarHostADevice(host_biases);
	device_weight_matrices->copiarHostADevice(host_pesos);
	cudaDeviceSynchronize();
}

void RedNeuronalSecuencial::cargarEnDevice(bool iniciarValoresBiasesWeights) {

	device_bias_vectors = new GestorPunteroPunteroFloatDevice(numero_capas - 1, getCopiaDimensionesCapasRed());
	device_weight_matrices = new GestorPunteroPunteroFloatDevice(numero_capas - 1, getCopiaDimensionesMatricesRed());

	float** puntero_host_device = device_bias_vectors->getPunteroPunteroHostDevice();
	float** punteros_en_host_de_device_weights = device_weight_matrices->getPunteroPunteroHostDevice();

	if (iniciarValoresBiasesWeights) {
		//de esta manera se ponen todos los valores de los biases a 0's
		for (int i = 0; i < device_bias_vectors->getNumeroElementos(); i++) {
			cudaMemset(puntero_host_device[i], 0, device_bias_vectors->getDimensionesElementos()[i] * sizeof(float));
		}

		//ahora estableceremos los valores aleatorios de los pesos
		curandGenerator_t generador_dnorm = crearGeneradorNumerosAleatoriosEnDistribucionNormal();

		for (int i = 0; i < device_weight_matrices->getNumeroElementos(); i++) {
			generarNumerosAleatoriosEnDistribucionNormal(generador_dnorm, 0.0, 1.0, punteros_en_host_de_device_weights[i], device_weight_matrices->getDimensionesElementos()[i]);
		}

		curandDestroyGenerator(generador_dnorm);
	}

	cudaDeviceSynchronize();

}

float* RedNeuronalSecuencial::propagacionHaciaDelante(int nejemplos, int nvalsentrada, float* matrizejemplos) {

	if (nvalsentrada == dimensiones_capas[0]) {

		int dimension_host = dimensiones_capas[numero_capas - 1];
		int dimension_device = 0;
		for (int i = 0; i < numero_capas; i++) { dimension_device = (int)vmax(dimension_device, dimensiones_capas[i]); }

		float* host_matriz_resultado = (float*)malloc(nejemplos * dimension_host * sizeof(float));

		float* device_matriz_entrada = 0;
		cudaMalloc(&device_matriz_entrada, nejemplos * dimension_device * sizeof(float));
		cudaMemcpy(device_matriz_entrada, matrizejemplos, nejemplos * nvalsentrada * sizeof(float), cudaMemcpyHostToDevice);

		float* device_matriz_resultado = 0;
		cudaMalloc(&device_matriz_resultado, nejemplos * dimension_device * sizeof(float));

		float** host_device_bias_vectors = device_bias_vectors->getPunteroPunteroHostDevice();
		float** host_device_weight_matrices = device_weight_matrices->getPunteroPunteroHostDevice();

		int M = nejemplos;
		int N = 0;
		int P = 0;

		for (int i = 0; i < numero_capas - 1; i++) {
			N = dimensiones_capas[i];
			P = dimensiones_capas[i + 1];

			productoMatricesDevice(device_matriz_entrada, host_device_weight_matrices[i], device_matriz_resultado, M, N, P);

			sumarCadaFilaMatrizVector << < dim3Ceil(M / (float)32, P / (float)32), dim3(32, 32) >> > (device_matriz_resultado, host_device_bias_vectors[i], M, P);
			cudaDeviceSynchronize();

			aplicarFuncion(funciones_capas[i], device_matriz_resultado, device_matriz_resultado, M, P);
			//aplicarFuncionSigmoideCadaElementoMatriz << < dim3Ceil(M / (float)32, P / (float)32), dim3(32, 32) >> > (device_matriz_resultado, device_matriz_resultado, M , P);
			//manageCUDAError(cudaDeviceSynchronize());

			cudaMemcpy(device_matriz_entrada, device_matriz_resultado, M * P * sizeof(float), cudaMemcpyDeviceToDevice);
		}

		cudaMemcpy(host_matriz_resultado, device_matriz_resultado, nejemplos * dimension_host * sizeof(float), cudaMemcpyDeviceToHost);

		cudaFree(device_matriz_resultado);
		device_matriz_resultado = NULL;

		return host_matriz_resultado;
	}
	else {
		printf("El numero de valores de cada ejemplo debe ser igual que el tamanno de la capa de entrada de la red");
		return NULL;
	}
}

int RedNeuronalSecuencial::getMaxTamMatrTempTrans(int batch_size) {
	int* dimensiones_matrices = getCopiaDimensionesMatricesRed();
	int mayor_dimension_matriz = 0;
	for (int i = 0; i < numero_capas - 1; i++) { mayor_dimension_matriz = (int)vmax(mayor_dimension_matriz, dimensiones_matrices[i]); }
	free(dimensiones_matrices);
	dimensiones_matrices = NULL;

	int mayor_tam_capa = 0;
	for (int i = 0; i < numero_capas; i++) { mayor_tam_capa = (int)vmax(mayor_tam_capa, dimensiones_capas[i]); }
	return ((int)vmax(mayor_dimension_matriz, batch_size * mayor_tam_capa));
}

int* RedNeuronalSecuencial::getDimensionesZlAl(int batch_size) {
	int* dimensionesCapas = getCopiaDimensionesCapasRed();
	for (int i = 0; i < numero_capas - 1; i++) { dimensionesCapas[i] = dimensionesCapas[i] * batch_size; }
	return dimensionesCapas;
}

float RedNeuronalSecuencial::calcularFuncionCosteMSE(int batch_size, int nvalsalida, float* vsalida) {

	if (nvalsalida == dimensiones_capas[numero_capas - 1]) {

		float** host_device_al = device_forward_al->getPunteroPunteroHostDevice();

		int os = batch_size * nvalsalida * sizeof(float);
		float* copia_vsalida = 0;
		cudaMalloc(&copia_vsalida, os);

		cudaMemcpy(copia_vsalida, vsalida, os, cudaMemcpyHostToDevice);

		aplicarFuncionCosteMSE << < dim3Ceil(batch_size / (float)32, nvalsalida / (float)32), dim3(32, 32) >> > (batch_size, nvalsalida, host_device_al[numero_capas - 2], copia_vsalida, copia_vsalida);
		cudaDeviceSynchronize();

		float* agrupados = 0;
		cudaMalloc(&agrupados, nvalsalida * sizeof(float));

		cudaMemset(agrupados, 0, nvalsalida * sizeof(float));

		sumarACadaElementoVectorColumnaMatriz <<< dim3Ceil(batch_size / (float)32, nvalsalida / (float)32), dim3(32, 32) >>> (copia_vsalida, agrupados, batch_size, nvalsalida);

		float* resultado = (float*) malloc( nvalsalida * sizeof(float) );
		cudaMemcpy(resultado, agrupados, nvalsalida * sizeof(float), cudaMemcpyDeviceToHost);

		cudaFree(copia_vsalida);
		cudaFree(agrupados);

		float res = 0.0;
		for (int i = 0; i < nvalsalida; i++) { res += resultado[i]; }

		//printf("\nValor de la funcion de coste MSE: %.16f",res/(batch_size*nvalsalida));

		return res;

	}

	return 0.0;
}

void RedNeuronalSecuencial::entrenarRedMSE_SGD(float tapren, int nepocas, int nejemplos, int batch_size, int nvalsentrada, int nvalsalida, float* ventrada, float* vsalida) {

	int ins = batch_size * nvalsentrada * sizeof(float);
	int os = batch_size * nvalsalida * sizeof(float);

	if (nvalsentrada == dimensiones_capas[0] && nvalsalida == dimensiones_capas[numero_capas - 1]) {

		cudaMalloc(&device_batch_input, ins);
		cudaMalloc(&device_batch_output, os);
		cudaMalloc(&temp_matr_traspose, getMaxTamMatrTempTrans(batch_size) * sizeof(float));

		device_forward_zl = new GestorPunteroPunteroFloatDevice(numero_capas - 1, getDimensionesZlAl(batch_size));
		device_forward_al = new GestorPunteroPunteroFloatDevice(numero_capas - 1, getDimensionesZlAl(batch_size));

		device_err_bias_m = new GestorPunteroPunteroFloatDevice(numero_capas - 1, getDimensionesZlAl(batch_size));
		device_err_weight_m = new GestorPunteroPunteroFloatDevice(numero_capas - 1, getCopiaDimensionesMatricesRed());
		//device_err_bias_v = NULL;
		//device_err_weight_v = NULL;

		for (int i = 0; i < nepocas; i++) {
			float error = 0.0;
			int nbatchs = (int)(nejemplos / batch_size);
			int nrelems = nejemplos % batch_size;
			for (int j = 0; j < nbatchs; j++) {
				cudaMemcpy(device_batch_input, ventrada + (batch_size * nvalsentrada * j), ins, cudaMemcpyHostToDevice);
				cudaMemcpy(device_batch_output, vsalida + (batch_size * nvalsalida * j), os, cudaMemcpyHostToDevice);
				cudaDeviceSynchronize();
				propagacionHaciaDelanteEntrenamiento(batch_size, nvalsentrada, ventrada + (batch_size * nvalsentrada * j));
				error += calcularFuncionCosteMSE(batch_size, nvalsalida, vsalida + (batch_size * nvalsalida * j));
				calcularErrorMSE_SGD(batch_size, nvalsalida, vsalida + (batch_size * nvalsalida * j));
				aplicarVectorGradienteSGD(tapren, batch_size);
			}
			//aquí se hace con el resto
			if (nrelems > 0) {
				cudaMemcpy(device_batch_input, ventrada + (batch_size * nvalsentrada * nbatchs), nrelems * nvalsentrada * sizeof(float), cudaMemcpyHostToDevice);
				cudaMemcpy(device_batch_output, vsalida + (batch_size * nvalsalida * nbatchs), nrelems * nvalsalida * sizeof(float), cudaMemcpyHostToDevice);
				cudaDeviceSynchronize();
				propagacionHaciaDelanteEntrenamiento(nrelems, nvalsentrada, ventrada + (batch_size * nvalsentrada * nbatchs));
				error += calcularFuncionCosteMSE(nrelems, nvalsalida, vsalida + (batch_size * nvalsalida * nbatchs));
				calcularErrorMSE_SGD(nrelems, nvalsalida, vsalida + (batch_size * nvalsalida * nbatchs));
				aplicarVectorGradienteSGD(tapren, nrelems);
			}
			if ((i + 1) % 500 == 0) {
				printf("\nError MSE: %.16f | Quedan %d epocas", (float)(error / ((float)(nejemplos * nvalsalida))), nepocas - i - 1);
			}
		}

		/*
		cudaMemcpy(device_batch_input, ventrada, ins, cudaMemcpyHostToDevice);
		cudaMemcpy(device_batch_output, vsalida, os, cudaMemcpyHostToDevice);

		
		for (int i = 0; i < nepocas; i++) {
			float error = 0.0;
			propagacionHaciaDelanteEntrenamiento(batch_size, nvalsentrada, ventrada);
			if ((i + 1) % 500 == 0) { 
				error += calcularFuncionCosteMSE(batch_size, nvalsalida, vsalida); 
				printf("\nError MSE: %.16f | Quedan %d epocas", error/(float)(nejemplos*nvalsalida), nepocas - i - 1);
			}
			calcularErrorMSE_SGD(batch_size, nvalsalida, vsalida);
			aplicarVectorGradienteSGD(tapren, batch_size);
		}
		printf("\n\nValor final de la funcion de coste:");
		propagacionHaciaDelanteEntrenamiento(batch_size, nvalsentrada, ventrada);
		calcularFuncionCosteMSE(batch_size, nvalsalida, vsalida);
		*/

		if (device_batch_input != NULL) { cudaFree(device_batch_input); device_batch_input = NULL; }
		if (device_batch_output != NULL) { cudaFree(device_batch_output); device_batch_output = NULL; }
		if (temp_matr_traspose != NULL) { cudaFree(temp_matr_traspose); temp_matr_traspose = NULL; }

		if (device_forward_zl != NULL) { delete device_forward_zl; device_forward_zl = NULL; }
		if (device_forward_al != NULL) { delete device_forward_al; device_forward_al = NULL; }

		if (device_err_bias_m != NULL) { delete device_err_bias_m; device_err_bias_m = NULL; }
		if (device_err_weight_m != NULL) { delete device_err_weight_m; device_err_weight_m = NULL; }
		//if (device_err_bias_v != NULL) { delete device_err_bias_v; device_err_bias_v = NULL; }
		//if (device_err_weight_v != NULL) { delete device_err_weight_v; device_err_weight_v = NULL; }

	} else {
		printf("El numero de valores de cada ejemplo de entrada y salida debe ser igual que el tamanno de la capa de entrada de la red");
	}

}

void RedNeuronalSecuencial::calcularErrorMSE_SGD(int batch_size, int nvalsalida, float* vsalida) {

	float** host_device_zl = device_forward_zl->getPunteroPunteroHostDevice();
	float** host_device_al = device_forward_al->getPunteroPunteroHostDevice();

	float** host_device_weight_matrices = device_weight_matrices->getPunteroPunteroHostDevice();

	float** host_device_bias_error_vectors = device_err_bias_m->getPunteroPunteroHostDevice();
	float** host_device_weight_error_matrices = device_err_weight_m->getPunteroPunteroHostDevice();

	aplicarDerivadaFuncionPerdidaMSECadaElementoPredY << < dim3Ceil(batch_size / (float)32, nvalsalida / (float)32), dim3(32, 32) >> > (batch_size, nvalsalida, host_device_al[numero_capas - 2], device_batch_output);
	cudaDeviceSynchronize();

	for (int i = numero_capas - 1; i > 0; i--) {

		//error bias actual

		aplicarDerivadaFuncion(funciones_capas[i-1], host_device_zl[i - 1], batch_size, dimensiones_capas[i]);
		//aplicarDerivadaFuncionSigmoideCadaElementoMatriz << < dim3Ceil(batch_size / (float)32, dimensiones_capas[i] / (float)32), dim3(32, 32) >> > (host_device_zl[i - 1], batch_size, dimensiones_capas[i]);
		//manageCUDAError(cudaDeviceSynchronize());

		multiplicarAMatrizAMatrizB << < dim3Ceil(batch_size / (float)32, dimensiones_capas[i] / (float)32), dim3(32, 32) >> > (host_device_al[i - 1], host_device_zl[i - 1], batch_size, dimensiones_capas[i]);
		cudaDeviceSynchronize();

		//error pesos

		if (i > 1) {
			matrizTraspuestaDevice(temp_matr_traspose, host_device_al[i - 2], batch_size, dimensiones_capas[i - 1]);
		} else {
			matrizTraspuestaDevice(temp_matr_traspose, device_batch_input, batch_size, dimensiones_capas[i - 1]);
		}

		productoMatricesDevice(temp_matr_traspose, host_device_al[i - 1], host_device_weight_error_matrices[i - 1], dimensiones_capas[i - 1], batch_size, dimensiones_capas[i]);

		//error bias anterior

		if (i > 1) {
			matrizTraspuestaDevice(temp_matr_traspose, host_device_weight_matrices[i - 1], dimensiones_capas[i - 1], dimensiones_capas[i]);

			productoMatricesDevice(host_device_al[i - 1], temp_matr_traspose, host_device_al[i - 2], batch_size, dimensiones_capas[i], dimensiones_capas[i - 1]);
		}

	}
}

void RedNeuronalSecuencial::aplicarVectorGradienteSGD(float tapren, int batch_size) {

	float factor = -1.0 * (tapren / batch_size);

	float** host_device_bias_vectors = device_bias_vectors->getPunteroPunteroHostDevice();
	float** host_device_weight_matrices = device_weight_matrices->getPunteroPunteroHostDevice();

	float** host_device_al = device_forward_al->getPunteroPunteroHostDevice();
	float** host_device_weight_error_matrices = device_err_weight_m->getPunteroPunteroHostDevice();	

	for (int i = 0; i < numero_capas - 1; i++) {

		dim3 dimension = dim3Ceil(batch_size / (float)32, dimensiones_capas[i + 1] / (float)32);

		multiplicarCadaElementoMatriz << < dimension, dim3(32, 32) >> > (host_device_al[i], factor, batch_size, dimensiones_capas[i + 1]);
		cudaDeviceSynchronize();

		sumarACadaElementoVectorColumnaMatriz << < dimension, dim3(32, 32) >> > (host_device_al[i], host_device_bias_vectors[i], batch_size, dimensiones_capas[i + 1]);
		cudaDeviceSynchronize();

		

		multiplicarCadaElementoMatriz << < dimension, dim3(32, 32) >> > (host_device_weight_error_matrices[i], factor, dimensiones_capas[i], dimensiones_capas[i + 1]);
		cudaDeviceSynchronize();

		sumarAMatrizAMatrizB << < dimension, dim3(32, 32) >> > (host_device_weight_matrices[i], host_device_weight_error_matrices[i], dimensiones_capas[i], dimensiones_capas[i + 1]);
		cudaDeviceSynchronize();
	}
}

void RedNeuronalSecuencial::propagacionHaciaDelanteEntrenamiento(int nejemplos, int nvalsentrada, float* matrizejemplos) {

	float** host_device_bias_vectors = device_bias_vectors->getPunteroPunteroHostDevice();
	float** host_device_weight_matrices = device_weight_matrices->getPunteroPunteroHostDevice();

	float** host_device_zl = device_forward_zl->getPunteroPunteroHostDevice();
	float** host_device_al = device_forward_al->getPunteroPunteroHostDevice();

	int M = nejemplos;
	int N = dimensiones_capas[0];
	int P = dimensiones_capas[1];

	productoMatricesDevice(device_batch_input, host_device_weight_matrices[0], host_device_zl[0], M, N, P);

	dim3 dimension = dim3Ceil(M / (float)32, P / (float)32);
	sumarCadaFilaMatrizVector << < dimension, dim3(32, 32) >> > (host_device_zl[0], host_device_bias_vectors[0], M, P);
	cudaDeviceSynchronize();

	aplicarFuncion(funciones_capas[0], host_device_zl[0], host_device_al[0], M, P);
	//aplicarFuncionSigmoideCadaElementoMatriz << < dimension, dim3(32, 32) >> > (host_device_zl[0], host_device_al[0], M, P);
	//manageCUDAError(cudaDeviceSynchronize());

	for (int i = 1; i < numero_capas - 1; i++) {
		N = dimensiones_capas[i];
		P = dimensiones_capas[i + 1];
		productoMatricesDevice(host_device_al[i - 1], host_device_weight_matrices[i], host_device_zl[i], M, N, P);

		dim3 dimension = dim3Ceil(M / (float)32, P / (float)32);

		sumarCadaFilaMatrizVector << < dimension, dim3(32, 32) >> > (host_device_zl[i], host_device_bias_vectors[i], M, P);
		cudaDeviceSynchronize();

		aplicarFuncion(funciones_capas[i], host_device_zl[i], host_device_al[i], M, P);
		//aplicarFuncionSigmoideCadaElementoMatriz << < dimension, dim3(32, 32) >> > (host_device_zl[i], host_device_al[i], M, P);
		//manageCUDAError(cudaDeviceSynchronize());
	}

}