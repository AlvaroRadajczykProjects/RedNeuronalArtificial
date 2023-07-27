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

const void productoMatricesDevice(float* a, float* b, float* c, int m, int n, int p) {
	productoMatrices <<< dim3Ceil((p + 32 - 1) / (float)32, (m + 32 - 1) / (float)32), dim3(32, 32) >> > (a, b, c, m, n, p);
	manageCUDAError(cudaDeviceSynchronize());
}

//T(idata) = odata
const void matrizTraspuestaDevice(float* odata, float* idata, int m, int n) {
	int dimension = (int)vmax(m, n);
	matrizTraspuesta << < dim3Ceil(dimension / (float)32, dimension / (float)32), dim3(32, 32) >> > (odata, idata, m, n);
	manageCUDAError(cudaDeviceSynchronize());
}

void mostarMatrizDevice( char* com, float* p, int m, int n ) {
	float* ph = new float[m * n];
	manageCUDAError(cudaMemcpy(ph, p, m*n*sizeof(float), cudaMemcpyDeviceToHost));
	manageCUDAError(cudaDeviceSynchronize());
	imprimirMatrizPorPantalla(com, ph, m, n);
	delete ph;
}

RedNeuronalSecuencial::RedNeuronalSecuencial(int nc, int* dc, int* fc) {
	numero_capas = nc;
	dimensiones_capas = dc;
	funciones_capas = fc;

	cargarEnDevice(true);
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

	if (device_batch_input != NULL) { manageCUDAError(cudaFree(device_batch_input)); device_batch_input = NULL; }
	if (device_batch_output != NULL) { manageCUDAError(cudaFree(device_batch_output)); device_batch_output = NULL; }
	if (temp_matr_traspose != NULL) { manageCUDAError(cudaFree(temp_matr_traspose)); temp_matr_traspose = NULL; }

	if (host_bias_vectors != NULL) {
		for (int i = 0; i < numero_capas; i++) {
			if (host_bias_vectors[i] != NULL) { free(host_bias_vectors[i]); host_bias_vectors[i] = NULL; }
		}
		free(host_bias_vectors); host_bias_vectors = NULL;
	}

	if (host_weight_matrices != NULL) {
		for (int i = 0; i < numero_capas; i++) {
			if (host_weight_matrices[i] != NULL) { free(host_weight_matrices[i]); host_weight_matrices[i] = NULL; }
		}
		free(host_weight_matrices); host_weight_matrices = NULL;
	}

	if (dimensiones_capas != NULL) { free(dimensiones_capas); dimensiones_capas = NULL; }
	if (funciones_capas != NULL) { free(funciones_capas); funciones_capas = NULL; }

	manageCUDAError(cudaDeviceSynchronize());

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

void RedNeuronalSecuencial::serializarRedNeuronal(const char* nombre_archivo, bool restablecer_red_device) {

	GestorPunteroPunteroFloatHost gestor_host_bias_vectors(numero_capas - 1, getCopiaDimensionesCapasRed());
	GestorPunteroPunteroFloatHost gestor_host_weight_matrices(numero_capas - 1, getCopiaDimensionesMatricesRed());

	host_bias_vectors = gestor_host_bias_vectors.getPunteroPunteroHost();
	host_weight_matrices = gestor_host_weight_matrices.getPunteroPunteroHost();

	device_bias_vectors->copiarDeviceAHost(host_bias_vectors);
	device_weight_matrices->copiarDeviceAHost(host_weight_matrices);

	//hay que eliminar toda la memoria de device para que no se copie
	if (device_bias_vectors != NULL) { delete device_bias_vectors; device_bias_vectors = NULL; }
	if (device_weight_matrices != NULL) { delete device_weight_matrices; device_weight_matrices = NULL; }

	if (device_forward_zl != NULL) { delete device_forward_zl; device_forward_zl = NULL; }
	if (device_forward_al != NULL) { delete device_forward_al; device_forward_al = NULL; }

	if (device_err_bias_m != NULL) { delete device_err_bias_m; device_err_bias_m = NULL; }
	if (device_err_weight_m != NULL) { delete device_err_weight_m; device_err_weight_m = NULL; }
	if (device_err_bias_v != NULL) { delete device_err_bias_v; device_err_bias_v = NULL; }
	if (device_err_weight_v != NULL) { delete device_err_weight_v; device_err_weight_v = NULL; }

	if (device_batch_input != NULL) { manageCUDAError(cudaFree(device_batch_input)); device_batch_input = NULL; }
	if (device_batch_output != NULL) { manageCUDAError(cudaFree(device_batch_output)); device_batch_output = NULL; }
	if (temp_matr_traspose != NULL) { manageCUDAError(cudaFree(temp_matr_traspose)); temp_matr_traspose = NULL; }

	//serialización

	//aquí restablecemos de host a device, sólo la red, el resto de vectores no
	if (restablecer_red_device) {
		cargarEnDevice(false);
		copiarPesosHostDevice(host_bias_vectors, host_weight_matrices);
	}

	manageCUDAError(cudaDeviceSynchronize());

	//se limpian solos los punteros, ojo también, no limpies host_bias_vectors ni host_weight_matrices en el destructor de esta clase
}

void RedNeuronalSecuencial::copiarPesosHostDevice(float** host_pesos, float** host_biases) {
	device_bias_vectors->copiarHostADevice(host_biases);
	device_weight_matrices->copiarHostADevice(host_pesos);
	manageCUDAError(cudaDeviceSynchronize());
}

void RedNeuronalSecuencial::cargarEnDevice(bool iniciarValoresBiasesWeights) {

	device_bias_vectors = new GestorPunteroPunteroFloatDevice(numero_capas - 1, getCopiaDimensionesCapasRed());
	device_weight_matrices = new GestorPunteroPunteroFloatDevice(numero_capas - 1, getCopiaDimensionesMatricesRed());

	float** puntero_host_device = device_bias_vectors->getPunteroPunteroHostDevice();
	float** punteros_en_host_de_device_weights = device_weight_matrices->getPunteroPunteroHostDevice();

	if (iniciarValoresBiasesWeights) {
		//de esta manera se ponen todos los valores de los biases a 0's
		for (int i = 0; i < device_bias_vectors->getNumeroElementos(); i++) {
			manageCUDAError(cudaMemset(puntero_host_device[i], 0, device_bias_vectors->getDimensionesElementos()[i] * sizeof(float)));
			//printf("%d: ", i);
			//mostarMatrizDevice("bias", puntero_host_device[i], device_bias_vectors->getDimensionesElementos()[i], 1);
		}

		//ahora estableceremos los valores aleatorios de los pesos
		curandGenerator_t generador_dnorm = crearGeneradorNumerosAleatoriosEnDistribucionNormal();

		for (int i = 0; i < device_weight_matrices->getNumeroElementos(); i++) {
			generarNumerosAleatoriosEnDistribucionNormal(generador_dnorm, 0.0, 1.0, punteros_en_host_de_device_weights[i], device_weight_matrices->getDimensionesElementos()[i]);
			//printf("%d: ", i);
			//mostarMatrizDevice("weight", punteros_en_host_de_device_weights[i], dimensiones_capas[i], dimensiones_capas[i+1]);
		}

		curandDestroyGenerator(generador_dnorm);
	}

	manageCUDAError(cudaDeviceSynchronize());

}

float* RedNeuronalSecuencial::propagacionHaciaDelante(int nejemplos, int nvalsentrada, float* matrizejemplos) {

	if (nvalsentrada == dimensiones_capas[0]) {

		int dimension_host = dimensiones_capas[numero_capas - 1];
		int dimension_device = 0;
		for (int i = 0; i < numero_capas; i++) { dimension_device = (int)vmax(dimension_device, dimensiones_capas[i]); }

		float* host_matriz_resultado = (float*)malloc(nejemplos * dimension_host * sizeof(float));

		float* device_matriz_entrada = 0;
		manageCUDAError(cudaMalloc(&device_matriz_entrada, nejemplos * dimension_device * sizeof(float)));
		manageCUDAError(cudaMemcpy(device_matriz_entrada, matrizejemplos, nejemplos * nvalsentrada * sizeof(float), cudaMemcpyHostToDevice));

		float* device_matriz_resultado = 0;
		manageCUDAError(cudaMalloc(&device_matriz_resultado, nejemplos * dimension_device * sizeof(float)));

		float** host_device_bias_vectors = device_bias_vectors->getPunteroPunteroHostDevice();
		float** host_device_weight_matrices = device_weight_matrices->getPunteroPunteroHostDevice();

		int M = nejemplos;
		int N = 0;
		int P = 0;

		manageCUDAError(cudaDeviceSynchronize());

		for (int i = 0; i < numero_capas - 1; i++) {
			N = dimensiones_capas[i];
			P = dimensiones_capas[i + 1];

			productoMatricesDevice(device_matriz_entrada, host_device_weight_matrices[i], device_matriz_resultado, M, N, P);
			//productoMatrices << < dim3(ceil((P + 32 - 1) / (float)32), ceil((M + 32 - 1) / (float)32)), dim3(32, 32) >> > (device_matriz_entrada, host_device_weight_matrices[i], device_matriz_resultado, M, N, P);
			//manageCUDAError(cudaDeviceSynchronize());
			sumarCadaFilaMatrizVector << < dim3Ceil(M / (float)32, P / (float)32), dim3(32, 32) >> > (device_matriz_resultado, host_device_bias_vectors[i], M, P);
			manageCUDAError(cudaDeviceSynchronize());
			aplicarFuncionSigmoideCadaElementoMatriz << < dim3Ceil(M / (float)32, P / (float)32), dim3(32, 32) >> > (device_matriz_resultado, device_matriz_resultado, M , P);
			manageCUDAError(cudaDeviceSynchronize());

			manageCUDAError(cudaMemcpy(device_matriz_entrada, device_matriz_resultado, M * P * sizeof(float), cudaMemcpyDeviceToDevice));
			manageCUDAError(cudaDeviceSynchronize());
		}

		manageCUDAError(cudaMemcpy(host_matriz_resultado, device_matriz_resultado, nejemplos * dimension_host * sizeof(float), cudaMemcpyDeviceToHost));
		manageCUDAError(cudaDeviceSynchronize());

		manageCUDAError(cudaFree(device_matriz_resultado));
		device_matriz_resultado = NULL;

		manageCUDAError(cudaDeviceSynchronize());

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

void RedNeuronalSecuencial::calcularFuncionCosteMSE(int batch_size, int nvalsalida, float* vsalida) {

	float* resultado = new float[1000];

	if (nvalsalida == dimensiones_capas[numero_capas - 1]) {

		float** host_device_al = device_forward_al->getPunteroPunteroHostDevice();

		int os = batch_size * nvalsalida * sizeof(float);
		float* copia_vsalida = 0;
		manageCUDAError(cudaMalloc(&copia_vsalida, os));
		manageCUDAError(cudaDeviceSynchronize());

		manageCUDAError(cudaMemcpy(copia_vsalida, vsalida, os, cudaMemcpyHostToDevice));
		manageCUDAError(cudaDeviceSynchronize());

		//mostarMatrizDevice("antes de calcular coste p", host_device_al[numero_capas - 2], batch_size, nvalsalida);
		//mostarMatrizDevice("antes de calcular coste r", copia_vsalida, batch_size, nvalsalida);

		aplicarFuncionCosteMSE << < dim3Ceil(batch_size / (float)32, nvalsalida / (float)32), dim3(32, 32) >> > (batch_size, nvalsalida, host_device_al[numero_capas - 2], copia_vsalida, copia_vsalida);
		manageCUDAError(cudaDeviceSynchronize());

		//mostarMatrizDevice("despues de calcular coste p", host_device_al[numero_capas - 2], batch_size, nvalsalida);
		//mostarMatrizDevice("despues de calcular coste r", copia_vsalida, batch_size, nvalsalida);

		float* agrupados = 0;
		manageCUDAError(cudaMalloc(&agrupados, nvalsalida * sizeof(float)));
		manageCUDAError(cudaDeviceSynchronize());

		manageCUDAError(cudaMemset(agrupados, 0, nvalsalida * sizeof(float)));
		manageCUDAError(cudaDeviceSynchronize());

		sumarACadaElementoVectorColumnaMatriz <<< dim3Ceil(batch_size / (float)32, nvalsalida / (float)32), dim3(32, 32) >>> (copia_vsalida, agrupados, batch_size, nvalsalida);

		float* resultado = (float*) malloc( nvalsalida * sizeof(float) );
		manageCUDAError(cudaMemcpy(resultado, agrupados, nvalsalida * sizeof(float), cudaMemcpyDeviceToHost));
		manageCUDAError(cudaDeviceSynchronize());

		manageCUDAError(cudaFree(copia_vsalida));
		manageCUDAError(cudaFree(agrupados));

		float res = 0.0;
		for (int i = 0; i < nvalsalida; i++) { res += resultado[i]; }

		printf("\nValor de la funcion de coste MSE: %f",res/(batch_size*nvalsalida));

	}

	delete resultado;
}

void RedNeuronalSecuencial::entrenarRedMSE_SGD(float tapren, int nepocas, int nejemplos, int batch_size, int nvalsentrada, int nvalsalida, float* ventrada, float* vsalida) {

	if (nvalsentrada == dimensiones_capas[0] && nvalsalida == dimensiones_capas[numero_capas - 1]) {

		int ins = batch_size * nvalsentrada * sizeof(float);
		int os = batch_size * nvalsalida * sizeof(float);

		manageCUDAError(cudaMalloc(&device_batch_input, ins));
		manageCUDAError(cudaMalloc(&device_batch_output, os));
		manageCUDAError(cudaMalloc(&temp_matr_traspose, getMaxTamMatrTempTrans(batch_size) * sizeof(float)));
		manageCUDAError(cudaDeviceSynchronize());

		device_forward_zl = new GestorPunteroPunteroFloatDevice(numero_capas - 1, getDimensionesZlAl(batch_size));
		device_forward_al = new GestorPunteroPunteroFloatDevice(numero_capas - 1, getDimensionesZlAl(batch_size));

		device_err_bias_m = new GestorPunteroPunteroFloatDevice(numero_capas - 1, getDimensionesZlAl(batch_size));
		device_err_weight_m = new GestorPunteroPunteroFloatDevice(numero_capas - 1, getCopiaDimensionesMatricesRed());
		//device_err_bias_v = NULL;
		//device_err_weight_v = NULL;

		manageCUDAError(cudaMemcpy(device_batch_input, ventrada, ins, cudaMemcpyHostToDevice));
		manageCUDAError(cudaMemcpy(device_batch_output, vsalida, os, cudaMemcpyHostToDevice));
		manageCUDAError(cudaDeviceSynchronize());

		for (int i = 0; i < nepocas; i++) {
			propagacionHaciaDelanteEntrenamiento(batch_size, nvalsentrada, ventrada);
			if ((i + 1) % 500 == 0) { 
				calcularFuncionCosteMSE(batch_size, nvalsalida, vsalida); 
				printf("\nQuedan %d epocas", nepocas-i-1 );
			}
			calcularErrorMSE_SGD(batch_size, nvalsalida, vsalida);
			aplicarVectorGradienteSGD(tapren, batch_size);
		}
		printf("\n\nValor final de la funcion de coste:");
		propagacionHaciaDelanteEntrenamiento(batch_size, nvalsentrada, ventrada);
		calcularFuncionCosteMSE(batch_size, nvalsalida, vsalida);

		if (device_batch_input != NULL) { manageCUDAError(cudaFree(device_batch_input)); device_batch_input = NULL; }
		if (device_batch_output != NULL) { manageCUDAError(cudaFree(device_batch_output)); device_batch_output = NULL; }
		if (temp_matr_traspose != NULL) { manageCUDAError(cudaFree(temp_matr_traspose)); temp_matr_traspose = NULL; }

		if (device_forward_zl != NULL) { delete device_forward_zl; device_forward_zl = NULL; }
		if (device_forward_al != NULL) { delete device_forward_al; device_forward_al = NULL; }

		if (device_err_bias_m != NULL) { delete device_err_bias_m; device_err_bias_m = NULL; }
		if (device_err_weight_m != NULL) { delete device_err_weight_m; device_err_weight_m = NULL; }
		//if (device_err_bias_v != NULL) { delete device_err_bias_v; device_err_bias_v = NULL; }
		//if (device_err_weight_v != NULL) { delete device_err_weight_v; device_err_weight_v = NULL; }

		manageCUDAError(cudaDeviceSynchronize());

	} else {
		printf("El numero de valores de cada ejemplo de entrada y salida debe ser igual que el tamanno de la capa de entrada de la red");
	}

}

void RedNeuronalSecuencial::calcularErrorMSE_SGD(int batch_size, int nvalsalida, float* vsalida) {

	//printf("\nHORA DE HACER EL TONTITO\n========================\n");

	float** host_device_zl = device_forward_zl->getPunteroPunteroHostDevice();
	float** host_device_al = device_forward_al->getPunteroPunteroHostDevice();

	float** host_device_weight_matrices = device_weight_matrices->getPunteroPunteroHostDevice();

	float** host_device_bias_error_vectors = device_err_bias_m->getPunteroPunteroHostDevice();
	float** host_device_weight_error_matrices = device_err_weight_m->getPunteroPunteroHostDevice();


	aplicarDerivadaFuncionPerdidaMSECadaElementoPredY << < dim3Ceil(batch_size / (float)32, nvalsalida / (float)32), dim3(32, 32) >> > (batch_size, nvalsalida, host_device_al[numero_capas - 2], device_batch_output);
	manageCUDAError(cudaDeviceSynchronize());

	for (int i = numero_capas - 1; i > 0; i--) {

		//error bias actual

		aplicarDerivadaFuncionSigmoideCadaElementoMatriz << < dim3Ceil(batch_size / (float)32, dimensiones_capas[i] / (float)32), dim3(32, 32) >> > (host_device_zl[i - 1], batch_size, dimensiones_capas[i]);
		manageCUDAError(cudaDeviceSynchronize());

		multiplicarAMatrizAMatrizB << < dim3Ceil(batch_size / (float)32, dimensiones_capas[i] / (float)32), dim3(32, 32) >> > (host_device_al[i - 1], host_device_zl[i - 1], batch_size, dimensiones_capas[i]);
		manageCUDAError(cudaDeviceSynchronize());

		//error pesos

		if (i > 1) {
			matrizTraspuestaDevice(temp_matr_traspose, host_device_al[i - 2], batch_size, dimensiones_capas[i - 1]);
			//matrizTraspuesta << < dim3Ceil(batch_size / (float)32, dimensiones_capas[i - 1] / (float)32), dim3(32, 32) >> > (temp_matr_traspose, host_device_al[i - 2], batch_size, dimensiones_capas[i - 1]);
			//manageCUDAError(cudaDeviceSynchronize());
		} else {
			matrizTraspuestaDevice(temp_matr_traspose, device_batch_input, batch_size, dimensiones_capas[i - 1]);
			//matrizTraspuesta << < dim3Ceil(batch_size / (float)32, dimensiones_capas[i] / (float)32), dim3(32, 32) >> > (temp_matr_traspose, device_batch_input, batch_size, dimensiones_capas[i - 1]);
			//manageCUDAError(cudaDeviceSynchronize());
		}

		productoMatricesDevice(temp_matr_traspose, host_device_al[i - 1], host_device_weight_error_matrices[i - 1], dimensiones_capas[i - 1], batch_size, dimensiones_capas[i]);
		//productoMatrices << < dim3(ceil((dimensiones_capas[i] + 32 - 1) / (float)32), ceil((dimensiones_capas[i - 1] + 32 - 1) / (float)32)), dim3(32, 32) >> > (temp_matr_traspose, host_device_al[i-1], host_device_weight_error_matrices[i-1], dimensiones_capas[i - 1], batch_size, dimensiones_capas[i]);
		//manageCUDAError(cudaDeviceSynchronize());

		//error bias anterior

		if (i > 1) {

			//mostarMatrizDevice("aplicar producto en al", host_device_al[i - 1], batch_size, dimensiones_capas[i]);

			//mostarMatrizDevice("al anterior antes producto", host_device_al[i - 2], batch_size, dimensiones_capas[i - 1]);

			//mostarMatrizDevice("pesos en capa actual", host_device_weight_matrices[i - 1], dimensiones_capas[i - 1], dimensiones_capas[i]);

			matrizTraspuestaDevice(temp_matr_traspose, host_device_weight_matrices[i - 1], dimensiones_capas[i - 1], dimensiones_capas[i]);
			//matrizTraspuesta << < dim3(ceil(dimensiones_capas[i - 1] / (float)32), ceil(dimensiones_capas[i] / (float)32)), dim3(32, 32) >> > (temp_matr_traspose, host_device_weight_matrices[i - 1], dimensiones_capas[i - 1], dimensiones_capas[i]);
			//manageCUDAError(cudaDeviceSynchronize());

			productoMatricesDevice(host_device_al[i - 1], temp_matr_traspose, host_device_al[i - 2], batch_size, dimensiones_capas[i], dimensiones_capas[i - 1]);
			//productoMatrices << < dim3(ceil((dimensiones_capas[i - 1] + 32 - 1) / (float)32), ceil((batch_size + 32 - 1) / (float)32)), dim3(32, 32) >> > (host_device_al[i-1], temp_matr_traspose, host_device_al[i - 2], batch_size, dimensiones_capas[i], dimensiones_capas[i - 1]);
			//manageCUDAError(cudaDeviceSynchronize());

			//mostarMatrizDevice("al anterior despues producto", host_device_al[i - 2], batch_size, dimensiones_capas[i-1]);
		}

	}

	/*
	for (int i = numero_capas - 1; i > 0; i--) {

		//error bias actual

		printf("\niteracion %d\n", i);

		mostarMatrizDevice("aplicando funcion en z(L): ", host_device_zl[i - 1], batch_size, dimensiones_capas[i]);
		mostarMatrizDevice("a(L) sin multiplicar: ", host_device_al[i - 1], batch_size, dimensiones_capas[i]);
		

		multiplicarAMatrizAMatrizB << < dim3(ceil(batch_size / (float)32), ceil(dimensiones_capas[i] / (float)32)), dim3(32, 32) >> > (host_device_al[i-1], host_device_zl[i-1], batch_size, dimensiones_capas[i]);
		manageCUDAError(cudaDeviceSynchronize());

		mostarMatrizDevice("a(L) multiplicado: ", host_device_al[i - 1], batch_size, dimensiones_capas[i]);

		//error pesos

		if (i > 1) {
			matrizTraspuesta << < dim3(ceil(batch_size / (float)32), ceil(dimensiones_capas[i-1] / (float)32)), dim3(32, 32) >> > (temp_matr_traspose, host_device_al[i - 2], batch_size, dimensiones_capas[i-1]);
			manageCUDAError(cudaDeviceSynchronize());
		} else {
			matrizTraspuesta << < dim3(ceil(batch_size / (float)32), ceil(dimensiones_capas[i] / (float)32)), dim3(32, 32) >> > (temp_matr_traspose, device_batch_input, batch_size, dimensiones_capas[i-1]);
			manageCUDAError(cudaDeviceSynchronize());
		}

		productoMatricesDevice(temp_matr_traspose, host_device_al[i - 1], host_device_weight_error_matrices[i - 1], dimensiones_capas[i - 1], batch_size, dimensiones_capas[i]);
		//productoMatrices << < dim3(ceil((dimensiones_capas[i] + 32 - 1) / (float)32), ceil((dimensiones_capas[i - 1] + 32 - 1) / (float)32)), dim3(32, 32) >> > (temp_matr_traspose, host_device_al[i-1], host_device_weight_error_matrices[i-1], dimensiones_capas[i - 1], batch_size, dimensiones_capas[i]);
		//manageCUDAError(cudaDeviceSynchronize());

		//error bias anterior

		if (i > 1) {
			matrizTraspuesta << < dim3(ceil(dimensiones_capas[i - 1] / (float)32), ceil(dimensiones_capas[i] / (float)32)), dim3(32, 32) >> > (temp_matr_traspose, host_device_weight_matrices[i - 1], dimensiones_capas[i - 1], dimensiones_capas[i]);
			manageCUDAError(cudaDeviceSynchronize());

			productoMatricesDevice(host_device_al[i - 1], temp_matr_traspose, host_device_al[i - 2], batch_size, dimensiones_capas[i], dimensiones_capas[i - 1]);
			//productoMatrices << < dim3(ceil((dimensiones_capas[i - 1] + 32 - 1) / (float)32), ceil((batch_size + 32 - 1) / (float)32)), dim3(32, 32) >> > (host_device_al[i-1], temp_matr_traspose, host_device_al[i - 2], batch_size, dimensiones_capas[i], dimensiones_capas[i - 1]);
			//manageCUDAError(cudaDeviceSynchronize());
		}
	}

	*/
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
		manageCUDAError(cudaDeviceSynchronize());

		sumarACadaElementoVectorColumnaMatriz << < dimension, dim3(32, 32) >> > (host_device_al[i], host_device_bias_vectors[i], batch_size, dimensiones_capas[i + 1]);
		manageCUDAError(cudaDeviceSynchronize());

		

		multiplicarCadaElementoMatriz << < dimension, dim3(32, 32) >> > (host_device_weight_error_matrices[i], factor, dimensiones_capas[i], dimensiones_capas[i + 1]);
		manageCUDAError(cudaDeviceSynchronize());

		sumarAMatrizAMatrizB << < dimension, dim3(32, 32) >> > (host_device_weight_matrices[i], host_device_weight_error_matrices[i], dimensiones_capas[i], dimensiones_capas[i + 1]);
		manageCUDAError(cudaDeviceSynchronize());
	}

	manageCUDAError(cudaDeviceSynchronize());
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
	//productoMatrices << < dim3(ceil((P + 32 - 1) / (float)32), ceil((M + 32 - 1) / (float)32)), dim3(32, 32) >> > (device_batch_input, host_device_weight_matrices[0], host_device_zl[0], M, N, P);
	//manageCUDAError(cudaDeviceSynchronize());

	dim3 dimension = dim3Ceil(M / (float)32, P / (float)32);
	sumarCadaFilaMatrizVector << < dimension, dim3(32, 32) >> > (host_device_zl[0], host_device_bias_vectors[0], M, P);
	manageCUDAError(cudaDeviceSynchronize());
	aplicarFuncionSigmoideCadaElementoMatriz << < dimension, dim3(32, 32) >> > (host_device_zl[0], host_device_al[0], M, P);
	manageCUDAError(cudaDeviceSynchronize());

	//mostarMatrizDevice("zl en 0", host_device_zl[0], M, P);
	//mostarMatrizDevice("propagacion al en 0", host_device_al[0], M, P);

	for (int i = 1; i < numero_capas - 1; i++) {
		N = dimensiones_capas[i];
		P = dimensiones_capas[i + 1];
		productoMatricesDevice(host_device_al[i - 1], host_device_weight_matrices[i], host_device_zl[i], M, N, P);
		//productoMatrices << < dim3(ceil((P + 32 - 1) / (float)32), ceil((M + 32 - 1) / (float)32)), dim3(32, 32) >> > (host_device_al[i - 1], host_device_weight_matrices[i], host_device_zl[i], M, N, P);
		//manageCUDAError(cudaDeviceSynchronize());
		dim3 dimension = dim3Ceil(M / (float)32, P / (float)32);
		sumarCadaFilaMatrizVector << < dimension, dim3(32, 32) >> > (host_device_zl[i], host_device_bias_vectors[i], M, P);
		manageCUDAError(cudaDeviceSynchronize());
		aplicarFuncionSigmoideCadaElementoMatriz << < dimension, dim3(32, 32) >> > (host_device_zl[i], host_device_al[i], M, P);
		manageCUDAError(cudaDeviceSynchronize());
		//printf("\niteracion -> %d\n",i);
		//mostarMatrizDevice("zl", host_device_zl[i], M, P);
		//mostarMatrizDevice("propagacion al", host_device_al[i], M, P);
	}

	/*
	for (int i = numero_capas - 1; i > 0; i--) {
		printf("al-zl en %d\n",i);
		mostarMatrizDevice("propagacion al", host_device_al[i-1], nejemplos, dimensiones_capas[i]);
		mostarMatrizDevice("propagacion zl", host_device_zl[i - 1], nejemplos, dimensiones_capas[i]);
	}
	*/

}