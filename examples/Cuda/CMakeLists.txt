cmake_minimum_required(VERSION 3.26)

if(UNIX)
	set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
endif ()

project(examples LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 17)

find_package(CUDAToolkit )

add_executable(examples main.cu)

target_link_libraries(${PROJECT_NAME} CUDA::cudart CUDA::cuda_driver)
