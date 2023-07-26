#include "cuda_runtime.h"
#include "cuda.h"
#include "device_launch_parameters.h"

#include <stdlib.h>
#include <stdio.h>
#include <Windows.h>

#include <stdio.h>

#include <iostream>


int main()
{

    size_t mf, ma;
    cudaMemGetInfo(&mf, &ma);
    std::cout << "free: " << mf << " total: " << ma << std::endl;
    Sleep(10000);
    int* d;
    size_t num = 3504172980;
    cudaMalloc(&d, num);
    cudaMemGetInfo(&mf, &ma);
    std::cout << "free: " << mf << " total: " << ma << std::endl;
    Sleep(10000);
    size_t mem = 7478929792;
    float* p = (float*)malloc(mem);
    for (unsigned long long i = 0; i < mem; i++) {
        p[i] = 0;
    }
    Sleep(10000);

    return 0;
}