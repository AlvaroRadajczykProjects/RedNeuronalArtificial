#include "cuda_functions.cuh"

float f_max(float a, float b);
int i_max(int a, int b);

void imprimirVectorIntPorPantalla(const char* texto_mostrar, float vector[], int inicio, int fin);
void imprimirMatrizPorPantalla(const char* texto_mostrar, float matriz[], int n_filas, int n_columnas);

class MatrizDevice {

	private:
		int ncols;
		int nrows;
		float* d_data = NULL;

	public:
		MatrizDevice(int nr, int nc);
		MatrizDevice(int nr, int nc, float* data);
		~MatrizDevice();
		int getNumRows();
		int getNumCols();
		float* getDeviceDataPointer();
		void copyFromMatrizDevice( MatrizDevice* md );
		float* getDataHost();
		void setDataFromHost(float* data);
		void show(const char* message);
		void setAllDataValuesToZero();
		void setAllDataValuesAsRandomNormalDistribution( float mean, float sdev );
		void applyFunction(int id_func);
		void applyDerivativeFunction(int id_func);
		void transpose( bool flipData );
		void copyTrasposedToMatrix(MatrizDevice* mp);
		void copyMatrixProductToMatrix(MatrizDevice* B, MatrizDevice* C);
		void addMatrixB(MatrizDevice* B);
		void multiplyEachElementWithEachElementMatrixB(MatrizDevice* B);
		void multiplyEachElementWithValue(float val);
		void applyMSELostFunction(MatrizDevice* real_y);
		void applyMSECostFunction(MatrizDevice* real_y, MatrizDevice* result);
		void copySumAllColumns(MatrizDevice* res);
		void sumEachRowVector(MatrizDevice* vector);

};