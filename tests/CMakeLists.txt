#  SPDX-License-Identifier: LGPL-3.0-only
# 
#  Author: Hao-Xin Wang <wanghaoxin1996@gmail.com>
#  Creation Date: Jan, 31th, 2024
#
#  Description: QuantumLiquids/UltraDMRG project. CMake file to control unittest.
# 
# Set MKL compile flags and link flags.

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

FIND_PACKAGE(BLAS REQUIRED)
FIND_PACKAGE(LAPACK REQUIRED)
find_package(MPI REQUIRED)

# Intel MKL multi-thread may conflict with other compiler's OpenMP
# Close OpenMP if using MKL with other compiler
if (NOT BLA_VENDOR STREQUAL "Intel10_64lp" OR CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
    find_package(OpenMP REQUIRED)
    link_libraries(OpenMP::OpenMP_CXX)
endif ()

set(MATH_LIB_LINK_FLAGS "BLAS::BLAS;LAPACK::LAPACK")

option(TEST_ACTUAL_COMBAT "test the actual examples of mps tensors" OFF)

if (TEST_ACTUAL_COMBAT)
    add_definitions(-DACTUALCOMBAT)
endif ()


if (QLMPS_USE_GPU)
    macro(add_unittest
            TEST_NAME TEST_SRC MATH_INCLUDE_DIR LINK_LIBS LINK_LIB_FLAGS INPUT_ARGS)
        add_executable(${TEST_NAME}
                ${TEST_SRC})

        target_include_directories(${TEST_NAME}
                PRIVATE ${MATH_INCLUDE_DIR}
                PRIVATE ${QLMPS_HEADER_PATH}
                PRIVATE ${QLTEN_HEADER_PATH}
                PRIVATE ${CUTENSOR_INCLUDE_DIR}
                PRIVATE ${MPI_CXX_HEADER_DIR}
        )
        target_link_libraries(${TEST_NAME}
                PRIVATE CUDA::cublas_static
                PRIVATE CUDA::cusolver_static
                PRIVATE ${CUTENSOR_LIBRARY}
                PRIVATE GTest::GTest GTest::Main
                "${LINK_LIBS}"
                PRIVATE "${LINK_LIB_FLAGS}"
        ) #remove MPI dependence since GPU code not support MPI upto now

        add_test(
                NAME ${TEST_NAME}
                COMMAND "${TEST_NAME}" "${INPUT_ARGS}"
                WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
        )

        set_target_properties(${TEST_NAME} PROPERTIES FOLDER tests)
    endmacro()
else ()
    macro(add_unittest
            TEST_NAME TEST_SRC MATH_INCLUDE_DIR LINK_LIBS LINK_LIB_FLAGS INPUT_ARGS)
        add_executable(${TEST_NAME}
                ${TEST_SRC})

        target_include_directories(${TEST_NAME}
                PRIVATE ${MATH_INCLUDE_DIR}
                PRIVATE ${QLMPS_HEADER_PATH}
                PRIVATE ${QLTEN_HEADER_PATH}
                PRIVATE ${hptt_INCLUDE_DIR}
                PRIVATE ${MPI_CXX_HEADER_DIR})
        target_link_libraries(${TEST_NAME}
                PRIVATE ${hptt_LIBRARY}
                PRIVATE MPI::MPI_CXX
                PRIVATE GTest::GTest GTest::Main
                "${LINK_LIBS}"
                PRIVATE "${LINK_LIB_FLAGS}"
        )

        add_test(
                NAME ${TEST_NAME}
                COMMAND "${TEST_NAME}" "${INPUT_ARGS}"
                WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
        )

        set_target_properties(${TEST_NAME} PROPERTIES FOLDER tests)
    endmacro()


    macro(add_mpi_unittest
            TEST_NAME TEST_SRC MATH_INCLUDE_DIR LINK_LIBS LINK_LIB_FLAGS PROCESSOR_NUM)
        add_executable(${TEST_NAME}
                ${TEST_SRC})

        target_include_directories(${TEST_NAME}
                PRIVATE ${MATH_INCLUDE_DIR}
                PRIVATE ${QLMPS_HEADER_PATH}
                PRIVATE ${QLTEN_HEADER_PATH}
                PRIVATE ${hptt_INCLUDE_DIR}
                PRIVATE ${MPI_CXX_HEADER_DIR})
        target_link_libraries(${TEST_NAME}
                GTest::GTest GTest::Main
                ${hptt_LIBRARY}
                MPI::MPI_CXX
                "${LINK_LIBS}" "${LINK_LIB_FLAGS}"
        )

        if (NOT (MPIEXEC_MAX_NUMPROCS MATCHES "1"))
            add_test(
                    NAME ${TEST_NAME}
                    COMMAND ${MPIEXEC_EXECUTABLE}
                    ${MPIEXEC_NUMPROC_FLAG} ${PROCESSOR_NUM}
                    $<TARGET_FILE:${TEST_NAME}>)
        endif ()
        set_target_properties(${TEST_NAME} PROPERTIES FOLDER tests)
    endmacro()
endif ()

## Test sites vector.
add_unittest(test_site_vec test_site_vec.cc "${BLAS_INCLUDE_DIR}" "" "${MATH_LIB_LINK_FLAGS}" "")

## Test one-dimensional tensor networks.
# Test DuoVector class
add_unittest(test_duovector test_one_dim_tn/test_duovector.cc "" "" "" "")
# Test TenVec class
add_unittest(test_ten_vec
        "test_one_dim_tn/test_ten_vec.cc"
        "${BLAS_INCLUDE_DIR}" "" "${MATH_LIB_LINK_FLAGS}" ""
)

# Test MPS class
add_unittest(test_finite_mps
        "test_one_dim_tn/test_finite_mps/test_finite_mps.cc"
        "${BLAS_INCLUDE_DIR}" "" "${MATH_LIB_LINK_FLAGS}" ""
)
# Test MPS measurement.
add_unittest(test_finite_mps_measu
        "test_one_dim_tn/test_finite_mps/test_finite_mps_measu.cc"
        "${BLAS_INCLUDE_DIR}" "" "${MATH_LIB_LINK_FLAGS}" ""
)

# Test MPO class
add_unittest(test_mpo
        "test_one_dim_tn/test_mpo.cc"
        "${BLAS_INCLUDE_DIR}" "" "${MATH_LIB_LINK_FLAGS}" ""
)
# Test MPO generation.
add_unittest(test_mpogen_coef_op_alg
        "test_one_dim_tn/test_mpogen/test_mpogen_coef_op_alg.cc"
        "${BLAS_INCLUDE_DIR}" "" "${MATH_LIB_LINK_FLAGS}" ""
)
add_unittest(test_mpogen_fsm
        "test_one_dim_tn/test_mpogen/test_mpogen_fsm.cc"
        "${BLAS_INCLUDE_DIR}" "" "${MATH_LIB_LINK_FLAGS}" ""
)
add_unittest(test_mpogen
        "test_one_dim_tn/test_mpogen/test_mpogen.cc"
        "${BLAS_INCLUDE_DIR}" "" "${MATH_LIB_LINK_FLAGS}" ""
)
add_unittest(test_impo_gen
        "test_one_dim_tn/test_mpogen/test_impogen.cc"
        "${BLAS_INCLUDE_DIR}" "" "${MATH_LIB_LINK_FLAGS}" ""
)


## Test algorithms
## Test simulation case parameters parser.
add_unittest(test_case_params_parser
        test_case_params_parser.cc
        "" "" ""
        "${CMAKE_CURRENT_LIST_DIR}/testdata/test-params.json" "")
# Lanczos solver
add_unittest(test_vmps_lanczos_solver
        "test_algorithm/test_lanczos_solver.cc"
        "${BLAS_INCLUDE_DIR}" "" "${MATH_LIB_LINK_FLAGS}" ""
)

# Lanczos solver
add_unittest(test_expmv_lanczos_solver
        "test_algorithm/test_lanczos_expmv_solver.cc"
        "${BLAS_INCLUDE_DIR}" "" "${MATH_LIB_LINK_FLAGS}" ""
)

# Two-site update finite vMPS
add_unittest(test_two_site_finite_vmps
        "test_algorithm/test_two_site_update_finite_vmps.cc"
        "${BLAS_INCLUDE_DIR}" "" "${MATH_LIB_LINK_FLAGS}" ""
)
# Single-site update finite vMPS
add_unittest(test_single_site_finite_vmps
        "test_algorithm/test_single_site_update_finite_vmps.cc"
        "${BLAS_INCLUDE_DIR}" "" "${MATH_LIB_LINK_FLAGS}" ""
)

# Two-site noised update finite vMPS
add_unittest(test_two_site_finite_vmps_with_noise
        "test_algorithm/test_two_site_noise_update_finite_vmps.cc"
        "${BLAS_INCLUDE_DIR}" "" "${MATH_LIB_LINK_FLAGS}" ""
)
# Two-site update finite DMRG
add_unittest(test_finite_dmrg
        "test_algorithm/test_finite_dmrg.cc"
        "${BLAS_INCLUDE_DIR}" "" "${MATH_LIB_LINK_FLAGS}" ""
)
add_unittest(test_fermion_two_site_update_finite_vmps
        "test_algorithm/test_fermion_two_site_update_finite_vmps.cc"
        "${BLAS_INCLUDE_DIR}" "" "${MATH_LIB_LINK_FLAGS}" ""
)
# Test two site TDVP
add_unittest(test_two_site_finite_tdvp
        "test_algorithm/test_two_site_finite_tdvp.cc"
        "${BLAS_INCLUDE_DIR}" "" "${MATH_LIB_LINK_FLAGS}" ""
)


if (NOT QLMPS_USE_GPU)
    # Test mpi two site update
    add_mpi_unittest(test_mpi_two_site_update_finite_vmps
            "test_mpi_algorithm/test_mpi_two_site_update_finite_vmps.cc"
            "${BLAS_INCLUDE_DIR}" "" "${MATH_LIB_LINK_FLAGS}" "4"
    )

    # Test mpi two site noised update
    add_mpi_unittest(test_mpi_two_site_update_noised_finite_vmps
            "test_mpi_algorithm/test_mpi_two_site_update_noised_finite_vmps.cc"
            "${BLAS_INCLUDE_DIR}" "" "${MATH_LIB_LINK_FLAGS}" "4"
    )

    # Test mpi two site TDVP
    add_mpi_unittest(test_mpi_two_site_finite_tdvp
            "test_mpi_algorithm/test_mpi_two_site_finite_tdvp.cc"
            "${BLAS_INCLUDE_DIR}" "" "${MATH_LIB_LINK_FLAGS}" "4"
    )

    # Test mpi two site DMRG
    add_mpi_unittest(test_mpi_finite_dmrg
            "test_mpi_algorithm/test_mpi_dmrg.cc"
            "${BLAS_INCLUDE_DIR}" "" "${MATH_LIB_LINK_FLAGS}" "4"
    )
endif ()