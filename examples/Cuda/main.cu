#include <vector>
#include <iostream>

#include "cuda_runtime.h"

__global__ void square(int *array, int n) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < n)
        array[tid] = array[tid] * array[tid];
}

int main() {

    std::vector<int> vec0(2048);
    for (int i = 0; i < vec0.size(); ++i) {
        vec0[i] = i + 2;
    }

    int *cudaVec;

    cudaMalloc(&cudaVec, vec0.size() * sizeof(int));
    cudaMemcpy(cudaVec, vec0.data(), vec0.size() * sizeof(int), cudaMemcpyHostToDevice);

    // Launch the Vector Add CUDA Kernel
    int threadsPerBlock = 1024;
    int blocksPerGrid = (vec0.size() + threadsPerBlock - 1) / threadsPerBlock;
    std::cout << "launching kernel blocksPerGrid " << blocksPerGrid << ", threadsPerBlock " << threadsPerBlock
              << std::endl;
    square<<<blocksPerGrid, threadsPerBlock>>>(cudaVec, vec0.size());

    cudaMemcpy(vec0.data(), cudaVec, vec0.size() * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(cudaVec);

    for (int i = 0; i < 11; ++i) {
        std::cout << vec0[i] << " ";
    }
    std::cout << std::endl;


    return 0;
}
