#include "MatrizDevice.cuh"

void manageError(const char* message) {
	printf("\nMatrizDevice ERROR: %s\n", message);
	exit(EXIT_FAILURE);
}

void imprimirVectorIntPorPantalla(const char* texto_mostrar, float vector[], int inicio, int fin) {
	printf("\n%s [ ", texto_mostrar);
	for (int i = inicio; i < fin; i++) {
		printf("%.15f", vector[i]);
		if (i < fin - 1) { printf(","); }
		printf(" ");
	}
	printf("]");
}

void imprimirMatrizPorPantalla(const char* texto_mostrar, float matriz[], int n_filas, int n_columnas) {
	printf("\n%s\n", texto_mostrar);
	for (int i = 0; i < n_filas; i++) {
		imprimirVectorIntPorPantalla(" ", matriz, i * n_columnas, i * n_columnas + n_columnas);
	}
	printf("\n");
}

float f_max(float a, float b) {
	return a > b ? a : b;
}

int i_max(int a, int b) {
	return a > b ? a : b;
}

dim3 dim3Ceil(float x, float y) {
	return dim3((int)ceil(x), (int)ceil(y));
}

MatrizDevice::MatrizDevice(int nr, int nc) {
	ncols = nc;
	nrows = nr;
	CUDA_CHECK_SYNC( cudaMalloc( &d_data, ncols * nrows * sizeof(float) ) );
}

MatrizDevice::MatrizDevice(int nr, int nc, float* data) {
	ncols = nc;
	nrows = nr;
	CUDA_CHECK_SYNC( cudaMalloc( &d_data, ncols * nrows * sizeof(float) ) );
	setDataFromHost(data);
}

MatrizDevice::~MatrizDevice() {
	CUDA_CHECK_SYNC( cudaFree(d_data) );
}

int MatrizDevice::getNumRows() {
	return nrows;
}

int MatrizDevice::getNumCols() {
	return ncols;
}

float* MatrizDevice::getDeviceDataPointer() {
	return d_data;
}

void MatrizDevice::copyFromMatrizDevice( MatrizDevice* md ) {
	if (ncols != md->ncols || nrows != md->nrows) { manageError("copyFromMatrizDevice -> tried to copy data from other matrix with different dimensionality"); }
	CUDA_CHECK_SYNC(cudaMemcpy(d_data, md->d_data, ncols * nrows * sizeof(float), cudaMemcpyDeviceToDevice));
}

float* MatrizDevice::getDataHost() {
	float* ret = (float*) malloc( ncols * nrows * sizeof(float) );
	CUDA_CHECK_SYNC(cudaMemcpy(ret, d_data, ncols * nrows * sizeof(float), cudaMemcpyDeviceToHost));
	return ret;
}

void MatrizDevice::setDataFromHost( float* data ) {
	CUDA_CHECK_SYNC(cudaMemcpy(d_data, data, ncols * nrows * sizeof(float), cudaMemcpyHostToDevice));
}

void MatrizDevice::show(const char* message) {
	float* ret = getDataHost();
	imprimirMatrizPorPantalla(message, ret, nrows, ncols);
	free(ret);
	ret = NULL;
}

void MatrizDevice::setAllDataValuesToZero() {
	CUDA_CHECK_SYNC( cudaMemset((void*) d_data, 0, ncols * nrows * sizeof(float) ) );
}

void MatrizDevice::setAllDataValuesAsRandomNormalDistribution(float mean, float sdev) {
	curandGenerator_t generador_dnorm = crearGeneradorNumerosAleatoriosEnDistribucionNormal();
	generarNumerosAleatoriosEnDistribucionNormal(generador_dnorm, mean, sdev, d_data, ncols*nrows);
	curandDestroyGenerator(generador_dnorm);
}

void MatrizDevice::applyFunction(int id_func) {
	if (id_func == 0) { aplicarFuncionSigmoideCadaElementoMatriz <<< dim3Ceil(nrows / (float)32, ncols / (float)32), dim3(32, 32) >>> (d_data, d_data, nrows, ncols); }
	else if (id_func == 1) { aplicarFuncionTahnCadaElementoMatriz <<< dim3Ceil(nrows / (float)32, ncols / (float)32), dim3(32, 32) >>> (d_data, d_data, nrows, ncols); }
	else if (id_func == 2) { aplicarFuncionCosenoEspecialCadaElementoMatriz <<< dim3Ceil(nrows / (float)32, ncols / (float)32), dim3(32, 32) >>> (d_data, d_data, nrows, ncols); }
	else if (id_func == 3) { aplicarFuncionPReluCadaElementoMatriz <<< dim3Ceil(nrows / (float)32, ncols / (float)32), dim3(32, 32) >>> (d_data, d_data, nrows, ncols); }
	CUDA_CHECK( cudaDeviceSynchronize() );
}

void MatrizDevice::applyDerivativeFunction(int id_func) {
	if (id_func == 0) { aplicarDerivadaFuncionSigmoideCadaElementoMatriz <<< dim3Ceil(nrows / (float)32, ncols / (float)32), dim3(32, 32) >>> (d_data, nrows, ncols); }
	else if (id_func == 1) { aplicarDerivadaFuncionTahnCadaElementoMatriz <<< dim3Ceil(nrows / (float)32, ncols / (float)32), dim3(32, 32) >>> (d_data, nrows, ncols); }
	else if (id_func == 2) { aplicarDerivadaFuncionCosenoEspecialCadaElementoMatriz <<< dim3Ceil(nrows / (float)32, ncols / (float)32), dim3(32, 32) >> > (d_data, nrows, ncols); }
	else if (id_func == 3) { aplicarDerivadaFuncionPReluCadaElementoMatriz <<< dim3Ceil(nrows / (float)32, ncols / (float)32), dim3(32, 32) >>> (d_data, nrows, ncols); }
	CUDA_CHECK(cudaDeviceSynchronize());
}

void MatrizDevice::transpose(bool flipData) {
	if (flipData) {
		int dimension = i_max(ncols, nrows);
		MatrizDevice* mp = new MatrizDevice(ncols, nrows);
		matrizTraspuesta << < dim3Ceil(dimension / (float)32, dimension / (float)32), dim3(32, 32) >> > (mp->getDeviceDataPointer(), d_data, nrows, ncols);
		CUDA_CHECK(cudaDeviceSynchronize());

		int temp = ncols;
		ncols = nrows;
		nrows = temp;

		copyFromMatrizDevice(mp);
		delete mp;
	} else {
		int temp = ncols;
		ncols = nrows;
		nrows = temp;
	}
}

void MatrizDevice::copyTrasposedToMatrix(MatrizDevice* mp) {
	if (ncols != mp->nrows || nrows != mp->ncols) { manageError("copyTrasposedToMatrix -> dest matrix has no correct trasposed dimensions"); }
	
	int dimension = i_max(ncols, nrows);
	matrizTraspuesta << < dim3Ceil(dimension / (float)32, dimension / (float)32), dim3(32, 32) >> > (mp->getDeviceDataPointer(), d_data, ncols, nrows);
	CUDA_CHECK(cudaDeviceSynchronize());
}

void MatrizDevice::copyMatrixProductToMatrix(MatrizDevice* B, MatrizDevice* C) {
	if (ncols != B->nrows) { manageError("copyMatrixProductToMatrix -> B matrix rows need to be the same as this matrix cols"); }
	if (nrows != C->nrows || B->ncols != C->ncols ) { manageError("copyMatrixProductToMatrix -> C rows must be = A rows and C cols must be = B cols"); }

	int dimension = i_max(nrows, i_max(ncols, C->ncols));

	productoMatrices << < dim3Ceil(dimension / (float)32, dimension / (float)32), dim3(32, 32) >> > (d_data, B->d_data, C->d_data, nrows, ncols, C->ncols);
	CUDA_CHECK(cudaDeviceSynchronize());
}

void MatrizDevice::addMatrixB(MatrizDevice* B) {
	if(nrows != B->nrows || ncols != B->ncols) { manageError("addMatrixB -> A rows must be = B rows and A cols must be = B cols"); }
	sumarAMatrizAMatrizB <<< dim3Ceil(nrows / (float)32, ncols / (float)32), dim3(32, 32) >>> (d_data, B->d_data, nrows, ncols);
	CUDA_CHECK(cudaDeviceSynchronize());
}

void MatrizDevice::multiplyEachElementWithEachElementMatrixB(MatrizDevice* B) {
	if (nrows != B->nrows || ncols != B->ncols) { manageError("multiplyEachElementWithEachElementMatrixB -> A rows must be = B rows and A cols must be = B cols"); }
	multiplicarAMatrizAMatrizB << < dim3Ceil(nrows / (float)32, ncols / (float)32), dim3(32, 32) >> > (d_data, B->d_data, nrows, ncols);
	CUDA_CHECK(cudaDeviceSynchronize());
}

void MatrizDevice::multiplyEachElementWithValue(float val) {
	multiplicarCadaElementoMatriz << < dim3Ceil(nrows / (float)32, ncols / (float)32), dim3(32, 32) >> > (d_data, val, nrows, ncols);
	CUDA_CHECK(cudaDeviceSynchronize());
}

void MatrizDevice::applyMSELostFunction(MatrizDevice* real_y) {
	if (nrows != real_y->nrows || ncols != real_y->ncols) { manageError("applyMSELostFunction -> matrices need to have the same shape"); }
	aplicarDerivadaFuncionPerdidaMSECadaElementoPredY << < dim3Ceil(nrows / (float)32, ncols / (float)32), dim3(32, 32) >> > (nrows, ncols, d_data, real_y->d_data);
	CUDA_CHECK(cudaDeviceSynchronize());
}

void MatrizDevice::applyMSECostFunction(MatrizDevice* real_y, MatrizDevice* result) {
	if (nrows != real_y->nrows || ncols != real_y->ncols) { manageError("applyMSECostFunction -> matrices need to have the same shape"); }
	aplicarFuncionCosteMSE << < dim3Ceil(nrows / (float)32, ncols / (float)32), dim3(32, 32) >> > (nrows, ncols, d_data, real_y->d_data, result->d_data);
	CUDA_CHECK(cudaDeviceSynchronize());
}

void MatrizDevice::copySumAllColumns(MatrizDevice* res) {
	if (res->nrows != 1 || ncols != res->ncols) { manageError("copySumAllColumns -> res rows must be 1 and res cols must be = this matrix cols"); }
	res->setAllDataValuesToZero();
	sumarACadaElementoVectorColumnaMatriz << < dim3Ceil(nrows / (float)32, ncols / (float)32), dim3(32, 32) >> > (d_data, res->d_data, nrows, ncols);
	CUDA_CHECK(cudaDeviceSynchronize());
}

void MatrizDevice::sumEachRowVector(MatrizDevice* vector) {
	if (vector->nrows != 1 || ncols != vector->ncols) { manageError("copySumAllColumns -> vector rows must be 1 and vector cols must be = this matrix cols"); }
	sumarCadaFilaMatrizVector << < dim3Ceil(nrows / (float)32, ncols / (float)32), dim3(32, 32) >> > (d_data, vector->d_data, nrows, ncols);
	CUDA_CHECK(cudaDeviceSynchronize());
}