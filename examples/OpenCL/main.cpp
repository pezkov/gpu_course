#ifdef __unix__
#include "CL/opencl.hpp"
#endif
#ifdef _WIN32
#include "CL/cl.hpp"
#endif
#include <iostream>
#include <vector>

int main(){
    // Исходные данные
    std::vector<float> A = {1, 2, 3, 4, 5};
    std::vector<float> B = {10, 20, 30, 40, 50};
    std::vector<float> C(A.size());

    // Получаем платформы и устройства
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    cl::Device device;
    for (const auto &platform: platforms){
        std::vector<cl::Device> devices;
        platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
        if(!devices.empty()){
            device = devices.front();
            break;
        }
    }

    // Создаем контекст и очередь команд
    cl::Context context(device);
    cl::CommandQueue queue(context, device);

    // Выделяем память на GPU
    cl::Buffer bufferA(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * A.size(), A.data());
    cl::Buffer bufferB(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * B.size(), B.data());
    cl::Buffer bufferC(context, CL_MEM_WRITE_ONLY, sizeof(float) * C.size());

    // Компилируем ядро
    const char* kernel_code = R"(
     __kernel void vector_add(__global const float *A, __global const float *B, __global float *C) {
            int id = get_global_id(0);
            C[id] = A[id] + B[id];
        }
    )";
    cl::Program program(context, kernel_code);
    program.build("-cl-std=CL1.2");

    // Создаем ядро и устанавливаем аргументы
    cl::Kernel kernel(program, "vector_add");
    kernel.setArg(0, bufferA);
    kernel.setArg(1, bufferB);
    kernel.setArg(2, bufferC);

    // Запускаем ядро
    queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(A.size()), cl::NullRange);

    // Копируем результат обратно на CPU
    queue.enqueueReadBuffer(bufferC, CL_TRUE, 0, sizeof(float) * C.size(), C.data());

    // Выводим результат
    for (float c : C) {
        std::cout << c << " ";
    }
    return 0;
}