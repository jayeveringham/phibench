cmake_minimum_required(VERSION 3.18)
project(phicuda LANGUAGES CXX CUDA)

# C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# CUDA setup
find_package(CUDAToolkit 11.0 REQUIRED)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 86)  # RTX 3090 = sm_86

# Options
option(PHI_BUILD_TESTS "Build test suite" ON)
option(PHI_BUILD_PYTHON "Build Python bindings" OFF)
option(PHI_USE_DOUBLE "Use double precision (vs float)" ON)

if(PHI_USE_DOUBLE)
    add_compile_definitions(PHI_REAL_TYPE=double)
else()
    add_compile_definitions(PHI_REAL_TYPE=float)
endif()

# Core CPU library
add_library(phi_core STATIC
    src/core/types.cpp
    src/data/tpm.cpp
    src/data/repertoire.cpp
    src/partition/bipartition.cpp
    src/compute/small_phi.cpp
    src/compute/big_phi.cpp
    src/metrics/emd.cpp
)

target_include_directories(phi_core PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_compile_options(phi_core PRIVATE
    $<$<CXX_COMPILER_ID:GNU>:-Wall -Wextra -O3 -march=native>
    $<$<CXX_COMPILER_ID:Clang>:-Wall -Wextra -O3 -march=native>
)

# CUDA library (Phase 2)
# add_library(phi_cuda STATIC
#     kernels/partition_kernel.cu
#     kernels/repertoire_kernel.cu
#     kernels/emd_kernel.cu
#     kernels/phi_kernel.cu
# )
# target_link_libraries(phi_cuda PUBLIC phi_core CUDA::cudart)
# set_target_properties(phi_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Main executable
add_executable(phicuda src/main.cpp)
target_link_libraries(phicuda PRIVATE phi_core)

# Tests
if(PHI_BUILD_TESTS)
    enable_testing()
    find_package(GTest QUIET)

    if(GTest_FOUND)
        add_executable(phi_tests
            tests/test_types.cpp
            tests/test_tpm.cpp
            tests/test_partition.cpp
            tests/test_repertoire.cpp
            tests/test_emd.cpp
            tests/test_small_phi.cpp
        )
        target_link_libraries(phi_tests PRIVATE phi_core GTest::gtest_main)
        gtest_discover_tests(phi_tests)
    else()
        message(WARNING "GTest not found, tests disabled. Install with: sudo apt install libgtest-dev")
    endif()
endif()

# Python bindings (Phase 4)
if(PHI_BUILD_PYTHON)
    find_package(pybind11 CONFIG REQUIRED)
    pybind11_add_module(phicuda_py python/bindings.cpp)
    target_link_libraries(phicuda_py PRIVATE phi_core)
endif()
