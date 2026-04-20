cmake_minimum_required(VERSION 3.20)
project(MyProject LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -static-global-template-stub=false")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(CUDAToolkit REQUIRED)

if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    # try to set CMAKE_CUDA_ARCHITECTURES to native card version
    if (CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
        set(CMAKE_CUDA_ARCHITECTURES native)
    else()
        message(STATUS "Running __nvcc_device_query for CMAKE_CUDA_ARCHITECTURES")
        execute_process(COMMAND "__nvcc_device_query"
            RESULT_VARIABLE NVCC_DEVICE_QUERY_RESULT
            OUTPUT_VARIABLE CMAKE_CUDA_ARCHITECTURES
            OUTPUT_STRIP_TRAILING_WHITESPACE)
        if (NVCC_DEVICE_QUERY_RESULT EQUAL 0)
            message(STATUS "CMAKE_CUDA_ARCHITECTURES set to ${CMAKE_CUDA_ARCHITECTURES}")
        else()
            message(WARNING "CMAKE_CUDA_ARCHITECTURES not defined and __nvcc_device_query failed.")
            unset(CMAKE_CUDA_ARCHITECTURES)
        endif()
    endif()
endif()

add_compile_options(-fdiagnostics-color=always -Wall)

set(CMAKE_BUILD_TYPE Debug)

file(GLOB_RECURSE SRC_FILES
    "${CMAKE_SOURCE_DIR}/src/*.cpp"
    "${CMAKE_SOURCE_DIR}/src/*.cu"
    "${CMAKE_SOURCE_DIR}/main.cpp"
)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

add_executable(app ${SRC_FILES})

include_directories(${CMAKE_SOURCE_DIR}/include)