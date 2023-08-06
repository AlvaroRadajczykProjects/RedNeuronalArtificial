#include <stdio.h>
#include <math.h>

#include "RedNeuronalSecuencialRobusta.cuh"

__global__ void suma(float* nums) {
    nums[0] = expf(nums[0] * nums[1]);
}

/*bool compare_float(float x, float y, float epsilon = 0.01f) {
    if (fabs(x - y) < epsilon)
        return true; //they are same
    return false; //they are not same
}

const char* boolToString(bool b) {
    return b ? "true" : "false";
}
*/

/*
int main(int argc, char** argv) {

    float* h_p = new float[2];
    h_p[0] = 3.63442;
    h_p[1] = 1 / (float)3;
    float* d_p = 0;

    float resant = (float)exp(h_p[0] * h_p[1]);

    printf("%.16f %.16f %.16f\n", h_p[0], h_p[1], resant);

    cudaMalloc(&d_p, 2 * sizeof(float));
    cudaMemcpy(d_p, h_p, 2 * sizeof(float), cudaMemcpyHostToDevice);
    suma << < dim3(1, 1), dim3(1, 1) >> > (d_p);
    cudaDeviceSynchronize();
    cudaMemcpy(h_p, d_p, 2 * sizeof(float), cudaMemcpyDeviceToHost);

    printf("%.16f\n", h_p[0]);

    printf("similares?: %s\n", boolToString(compare_float(resant, h_p[0], 0.0001f)));

    return 0;
}
*/