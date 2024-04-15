cmake_minimum_required(VERSION 3.14)  # change the version, if asked by compiler
project(mlp)

find_package(CUDA REQUIRED)
add_definitions(-std=c++11)
set(TENSORRT_DIR "/usr/src/tensorrt")
set(CMAKE_CXX_STANDARD 11)
add_definitions(-DAPI_EXPORTS)
option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)

# include and link dirs of cuda for inference
message(STATUS "    libraries: ${CUDA_LIBRARIES}")
message(STATUS "    include path: ${CUDA_INCLUDE_DIRS}")
include_directories(${CUDA_INCLUDE_DIRS})

include_directories(./)
cuda_add_library(gelu_plugin SHARED gelu.cu)
target_link_libraries(gelu_plugin nvinfer cudart)

# create link for executable files
add_executable(mlp  mlp.cpp)

# perform linking with nvinfer libraries
target_link_libraries(mlp nvinfer)
target_link_libraries(mlp gelu_plugin)

# link with cuda libraries for Inference
target_link_libraries(mlp cudart)

add_definitions(-O2 -pthread)

