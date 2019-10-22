cmake_minimum_required(VERSION 3.10)
project(Assignment6)

find_package(CUDA REQUIRED)
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -O3 -lm")

cuda_add_executable(mandel mandel.cu)