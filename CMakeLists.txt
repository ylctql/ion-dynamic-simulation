cmake_minimum_required(VERSION 3.21)
project(ioncpp LANGUAGES CXX CUDA)

enable_language(CUDA)

project(quantum_sim LANGUAGES CXX)

add_compile_options("$<$<C_COMPILER_ID:MSVC>:/utf-8>")
add_compile_options("$<$<CXX_COMPILER_ID:MSVC>:/utf-8>")

add_library(Eigen INTERFACE)
target_include_directories(Eigen INTERFACE externals/eigen-4.3.0)

find_package(OpenMP QUIET)
if(OpenMP_FOUND)
	message("OpenMP Enabled")
	target_link_libraries(Eigen INTERFACE OpenMP::OpenMP_CXX)
endif(OpenMP_FOUND)

add_library(ioncpp)

set_target_properties(ioncpp PROPERTIES
    CUDA_ARCHITECTURES 86
    CUDA_STANDARD 14
)

# Find CUDA installation
# Try modern CUDAToolkit first (CMake 3.18+)
find_package(CUDAToolkit QUIET)
if(CUDAToolkit_FOUND)
    message(STATUS "Found CUDA Toolkit: ${CUDAToolkit_ROOT_DIR}")
    include_directories(${CUDAToolkit_INCLUDE_DIRS})
    link_directories(${CUDAToolkit_LIBRARY_DIR})
else()
    # Fallback: try legacy CUDA package
    find_package(CUDA QUIET)
    if(CUDA_FOUND)
        message(STATUS "Found CUDA: ${CUDA_TOOLKIT_ROOT_DIR}")
        include_directories(${CUDA_INCLUDE_DIRS})
        link_directories(${CUDA_LIBRARY_DIRS})
    else()
        # Final fallback: try common CUDA paths
        set(CUDA_PATHS
            /usr/local/cuda
            /usr/local/cuda-11.8
            /usr/local/cuda-12.8
            /usr/local/cuda-11
        )
        foreach(CUDA_PATH ${CUDA_PATHS})
            if(EXISTS ${CUDA_PATH}/include/cuda.h)
                message(STATUS "Found CUDA at: ${CUDA_PATH}")
                include_directories(${CUDA_PATH}/include)
                link_directories(${CUDA_PATH}/lib64)
                break()
            endif()
        endforeach()
    endif()
endif()

target_link_libraries(ioncpp PRIVATE cudart)

target_compile_features(ioncpp PRIVATE cxx_std_17)
set_target_properties(ioncpp PROPERTIES CXX_STANDARD 17)
set_target_properties(ioncpp PROPERTIES CXX_STANDARD_REQUIRED ON)
set_target_properties(ioncpp PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_link_libraries(ioncpp PUBLIC Eigen)

add_subdirectory(ioncpp)



add_subdirectory(externals/pybind11)
pybind11_add_module(ionsim ionsim.cpp)
target_compile_features(ionsim PRIVATE cxx_std_17)
set_target_properties(ionsim PROPERTIES CXX_STANDARD 17)
set_target_properties(ionsim PROPERTIES CXX_STANDARD_REQUIRED ON)

if(VERSION_INFO)
	target_compile_definitions(ionsim PRIVATE VERSION_INFO=${VERSION_INFO})
endif(VERSION_INFO)

target_link_libraries(ionsim PRIVATE ioncpp)

option(BUILD_MAIN "Build main for testing purpose" ON)

if(BUILD_MAIN)
	add_executable(main)

	target_sources(main PRIVATE main.cpp)

	target_compile_features(main PRIVATE cxx_std_17)
	set_target_properties(main PROPERTIES CXX_STANDARD 17)
	set_target_properties(main PROPERTIES CXX_STANDARD_REQUIRED ON)

	target_link_libraries(main PRIVATE ioncpp)
endif(BUILD_MAIN)