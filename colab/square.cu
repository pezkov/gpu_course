#include "cuda_runtime.h"

#include <iostream>
#include <vector>


#define CHECK_CUDA_ERROR(call) { 						                            \
    cudaError_t err = call; 							                              \
    if (err != cudaSuccess) { 							                            \
        std::cerr << "Ошибка CUDA (файл: " << __FILE__ << ", строка: "  \
         << __LINE__ << "): " << cudaGetErrorString(err) << std::endl ; \
        return (int32_t)err;							                              \
    } 										                                              \
}

__global__ void square(int *array, int n) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < n)
        array[tid] = array[tid] * array[tid];
}

int32_t square() {

    std::vector<int> vec0(2048);
    for (int i = 0; i < vec0.size(); ++i) {
        vec0[i] = i + 2;
    }

    int *cudaVec;

    CHECK_CUDA_ERROR(cudaMalloc(&cudaVec, vec0.size() * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMemcpy(cudaVec, vec0.data(), vec0.size() * sizeof(int), cudaMemcpyHostToDevice));

    // Launch the Vector Add CUDA Kernel
    int threadsPerBlock = 1024;
    int blocksPerGrid = (vec0.size() + threadsPerBlock - 1) / threadsPerBlock;
    std::cout << "launching kernel blocksPerGrid " << blocksPerGrid << ", threadsPerBlock " << threadsPerBlock
              << std::endl;
    square<<<blocksPerGrid, threadsPerBlock>>>(cudaVec, vec0.size());

    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    CHECK_CUDA_ERROR(cudaMemcpy(vec0.data(), cudaVec, vec0.size() * sizeof(int), cudaMemcpyDeviceToHost));

    CHECK_CUDA_ERROR(cudaFree(cudaVec));

    for (int i = 0; i < 11; ++i) {
        std::cout << vec0[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}


int main() {
    auto err = square();
    if(err){
      return err;
    }

    return 0;
}
