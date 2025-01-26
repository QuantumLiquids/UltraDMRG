#  SPDX-License-Identifier: LGPL-3.0-only
# 
#  Author: Hao-Xin Wang <wanghaoxin1996@gmail.com>
#  Creation Date: Aug, 13th, 2024
#
#  Description: QuantumLiquids/UltraDMRG project. CMake file to compile the benchmark file
# 
# Set MKL compile flags and link flags.

option(QLMPS_USE_GPU "Use GPU" OFF)
if (QLMPS_USE_GPU)
    set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/cmake/Modules/") # find cutensor

    add_definitions(-DUSE_GPU=1)

    # Enable CUDA language support
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)  # Modern CUDA Toolkit detection

    include_directories(${CUDAToolkit_INCLUDE_DIRS})
    message(STATUS "CUDA_INCLUDE_DIRS: '${CUDAToolkit_INCLUDE_DIRS}'")

    # Set CUDA architectures
    set(CMAKE_CUDA_ARCHITECTURES 70 80) # For V100 & A100, adjust if needed
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -lineinfo -Xcompiler=-fPIC")


    find_package(CUTENSOR REQUIRED)

    file(GLOB_RECURSE TEST_SOURCES_SUBDIRS */*.cc)
    file(GLOB TEST_SOURCES_ROOT *.cc)
    set(TEST_SOURCES ${TEST_SOURCES_SUBDIRS} ${TEST_SOURCES_ROOT})
    message(STATUS "TEST_SOURCES" ${TEST_SOURCES})
    # Treat all .cc files in tests/ as CUDA files
    set_source_files_properties(${TEST_SOURCES} PROPERTIES LANGUAGE CUDA)
else ()
    find_path(hptt_INCLUDE_DIR "hptt.h")
    find_library(hptt_LIBRARY "libhptt.a")
    message(STATUS "HPTT Lib : " "${hptt_LIBRARY}")
endif ()
option(QLTEN_USE_OPENBLAS "Use openblas rather mkl" OFF)


if (CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
    set(BLA_VENDOR Intel10_64lp)   #FOR VENDOR CHOICE: https://cmake.org/cmake/help/latest/module/FindBLAS.html
    set(BLAS_INCLUDE_DIR "$ENV{MKLROOT}/include")
elseif (CMAKE_SYSTEM_PROCESSOR MATCHES "arm64") # (e.g., Apple Silicon)
    add_definitions(-DUSE_OPENBLAS)
    set(BLA_VENDOR OpenBLAS)
    set(OpenBLAS_ROOT "/opt/homebrew/opt/openblas/") #work for homebrew installed openblas in MacOSX
    set(Lapack_ROOT "/opt/homebrew/opt/lapack")
    set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} ${OpenBLAS_ROOT} ${Lapack_ROOT}) # For find
    set(BLAS_INCLUDE_DIR "${OpenBLAS_ROOT}/include")
endif ()

set(MATH_LIB_LINK_FLAGS "BLAS::BLAS;LAPACK::LAPACK")

FIND_PACKAGE(BLAS REQUIRED)
FIND_PACKAGE(LAPACK REQUIRED)
find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

option(TEST_ACTUAL_COMBAT "test the actual examples of mps tensors" OFF)

if (TEST_ACTUAL_COMBAT)
    add_definitions(-DACTUALCOMBAT)
endif ()


if (QLTEN_USE_GPU)
    macro(add_unittest
            TEST_NAME TEST_SRC MATH_INCLUDE_DIR LINK_LIBS LINK_LIB_FLAGS INPUT_ARGS)
        add_executable(${TEST_NAME}
                ${TEST_SRC})

        target_include_directories(${TEST_NAME}
                PRIVATE ${MATH_INCLUDE_DIR}
                PRIVATE ${QLTEN_HEADER_PATH}
                PRIVATE ${CUTENSOR_INCLUDE_DIR}
                PRIVATE ${MPI_CXX_HEADER_DIR}
        )
        target_link_libraries(${TEST_NAME}
                PRIVATE CUDA::cublas_static
                PRIVATE CUDA::cusolver_static
                PRIVATE ${CUTENSOR_LIBRARY}
                PRIVATE GTest::GTest GTest::Main OpenMP::OpenMP_CXX
                "${LINK_LIBS}" "${LINK_LIB_FLAGS}"
        ) #remove MPI dependence since GPU code not support MPI upto now

        add_test(
                NAME ${TEST_NAME}
                COMMAND "${TEST_NAME}")

        set_target_properties(${TEST_NAME} PROPERTIES FOLDER tests)
    endmacro()
else ()
    macro(add_unittest
            TEST_NAME TEST_SRC MATH_INCLUDE_DIR LINK_LIBS LINK_LIB_FLAGS INPUT_ARGS)
        add_executable(${TEST_NAME}
                ${TEST_SRC})

        target_include_directories(${TEST_NAME}
                PRIVATE ${MATH_INCLUDE_DIR}
                PRIVATE ${QLTEN_HEADER_PATH}
                PRIVATE ${hptt_INCLUDE_DIR}
                PRIVATE ${MPI_CXX_HEADER_DIR})
        target_link_libraries(${TEST_NAME}
                PRIVATE ${hptt_LIBRARY}
                MPI::MPI_CXX
                PRIVATE GTest::GTest GTest::Main OpenMP::OpenMP_CXX
                "${LINK_LIBS}" "${LINK_LIB_FLAGS}"
        )

        add_test(
                NAME ${TEST_NAME}
                COMMAND "${TEST_NAME}")

        set_target_properties(${TEST_NAME} PROPERTIES FOLDER tests)
    endmacro()
endif ()


add_unittest(ultradmrg_hubbard2d
        "ultradmrg_hubbard2d.cpp"
        "${BLAS_INCLUDE_DIR}" "" "${MATH_LIB_LINK_FLAGS}" ""
)