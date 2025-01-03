#  SPDX-License-Identifier: LGPL-3.0-only
# 
#  Author: Hao-Xin Wang <wanghaoxin1996@gmail.com>
#  Creation Date: Jan, 31th, 2024
#
#  Description: QuantumLiquids/UltraDMRG project. CMake file to control unittest.
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
elseif ()
    find_path(hptt_INCLUDE_DIR "hptt.h")
    find_library(hptt_LIBRARY "libhptt.a")
endif ()
option(QLTEN_USE_OPENBLAS "Use openblas rather mkl" OFF)


if (APPLE AND CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
    if (CMAKE_CXX_COMPILER_ID MATCHES "Intel")
        set(MATH_LIB_COMPILE_FLAGS "-I$ENV{MKLROOT}/include")
        #Need test
        set(MATH_LIB_LINK_FLAGS $ENV{MKLROOT}/lib/libmkl_intel_lp64.a $ENV{MKLROOT}/lib/libmkl_intel_thread.a $ENV{MKLROOT}/lib/libmkl_core.a -liomp5 -lpthread -lm -ldl)
    endif ()
    if (CMAKE_CXX_COMPILER_ID MATCHES "GNU")
        set(MATH_LIB_COMPILE_FLAGS -m64 -I$ENV{MKLROOT}/include)
        # May not work
        set(MATH_LIB_LINK_FLAGS $ENV{MKLROOT}/lib/libmkl_intel_lp64.a $ENV{MKLROOT}/lib/libmkl_intel_thread.a $ENV{MKLROOT}/lib/libmkl_core.a -L$ENV{MKLROOT}/lib -L$ENV{CMPLR_ROOT}/mac/compiler/lib/ -liomp5 -lpthread -lm -ldl)
    endif ()
    if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        set(MATH_LIB_COMPILE_FLAGS -m64 -I$ENV{MKLROOT}/include)
        # Note as of Intel oneAPI 2021.2, "source /opt/intel/oneapi/compiler/latest/env/vars.sh"
        set(MATH_LIB_LINK_FLAGS $ENV{MKLROOT}/lib/libmkl_intel_lp64.a $ENV{MKLROOT}/lib/libmkl_intel_thread.a $ENV{MKLROOT}/lib/libmkl_core.a -L$ENV{MKLROOT}/lib -L$ENV{CMPLR_ROOT}/mac/compiler/lib/ -Wl, -rpath $ENV{CMPLR_ROOT}/mac/compiler/lib/libiomp5.dylib -liomp5 -lpthread -lm -ldl)
    endif ()
elseif (APPLE)
    #ARM
    add_definitions(-DUSE_OPENBLAS)
    set(BLA_VENDOR OpenBLAS)
    #        FIND_PACKAGE(BLAS REQUIRED)
    #        FIND_PACKAGE(LAPACK REQUIRED)
    set(OpenBLAS_ROOT "/opt/homebrew/opt/openblas/")
    set(Lapack_ROOT "/opt/homebrew/opt/lapack")
    set(OpenBLAS_INCLUDE_DIRS "${OpenBLAS_ROOT}/include")
    set(OpenBLAS_LIBRARIES "${OpenBLAS_ROOT}/lib/libblas.dylib")
    set(MATH_LIB_COMPILE_FLAGS -I${OpenBLAS_INCLUDE_DIRS} -pthread)
    set(MATH_LIB_LINK_FLAGS ${OpenBLAS_LIBRARIES} ${OpenBLAS_ROOT}/lib/liblapack.dylib -lm -lpthread -ldl -fopenmp -lclapack)
elseif (UNIX)
    #UNIX include APPLE, but we except it here
    if (CMAKE_CXX_COMPILER_ID MATCHES "Intel")
        set(MATH_LIB_COMPILE_FLAGS "-I$ENV{MKLROOT}/include")
        set(MATH_LIB_LINK_FLAGS -Wl,--start-group $ENV{MKLROOT}/lib/intel64/libmkl_intel_lp64.a $ENV{MKLROOT}/lib/intel64/libmkl_intel_thread.a $ENV{MKLROOT}/lib/intel64/libmkl_core.a -Wl,--end-group -liomp5 -lpthread -lm -ldl)
    endif ()
    if (CMAKE_CXX_COMPILER_ID MATCHES "GNU")
        set(MATH_LIB_COMPILE_FLAGS -m64 -I$ENV{MKLROOT}/include)
        # Link the Intel's OpenMP library to avoid performance issue when the library calls the MKL's gesdd function.
        set(MATH_LIB_LINK_FLAGS -Wl,--start-group $ENV{MKLROOT}/lib/intel64/libmkl_intel_lp64.a $ENV{MKLROOT}/lib/intel64/libmkl_intel_thread.a $ENV{MKLROOT}/lib/intel64/libmkl_core.a -Wl,--end-group -L$ENV{MKLROOT}/lib/intel64 -liomp5 -lpthread -lm -ldl)
    endif ()
    if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        set(MATH_LIB_COMPILE_FLAGS -m64 -I$ENV{MKLROOT}/include)
        set(MATH_LIB_LINK_FLAGS -Wl,--start-group $ENV{MKLROOT}/lib/intel64/libmkl_intel_lp64.a $ENV{MKLROOT}/lib/intel64/libmkl_intel_thread.a $ENV{MKLROOT}/lib/intel64/libmkl_core.a -Wl,--end-group -L$ENV{MKLROOT}/lib/intel64 -liomp5 -lpthread -lm -ldl)
    endif ()
endif ()


find_package(Boost
        1.74.0
        REQUIRED
        COMPONENTS serialization mpi)

find_package(MPI REQUIRED)

option(TEST_ACTUAL_COMBAT "test the actual examples of mps tensors" OFF)

if (TEST_ACTUAL_COMBAT)
    add_definitions(-DACTUALCOMBAT)
endif ()

#set omp flag
if (CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
    set(OMP_FLAGS -qopenmp)
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(OMP_FLAGS -fopenmp)
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    set(OMP_FLAGS -fopenmp)
endif ()

if (QLMPS_USE_GPU)
    macro(add_unittest
            TEST_NAME TEST_SRC CFLAGS LINK_LIBS LINK_LIB_FLAGS INPUT_ARGS)
        add_executable(${TEST_NAME}
                ${TEST_SRC})

        target_compile_options(${TEST_NAME}
                PRIVATE ${CFLAGS}
                PRIVATE ${OMP_FLAGS}
        )

        target_include_directories(${TEST_NAME}
                PRIVATE ${QLMPS_HEADER_PATH}
                PRIVATE ${QLTEN_HEADER_PATH}
                PRIVATE ${CUTENSOR_INCLUDE_DIR}
                PRIVATE ${Boost_INCLUDE_DIR}
                PRIVATE ${MPI_CXX_HEADER_DIR}
        )
        target_link_libraries(${TEST_NAME}
                PRIVATE CUDA::cublas_static
                PRIVATE CUDA::cusolver_static
                PRIVATE ${CUTENSOR_LIBRARY}
                PRIVATE GTest::GTest GTest::Main
                "${LINK_LIBS}" "${LINK_LIB_FLAGS}"
        ) #remove MPI and boost dependence since GPU code not support MPI upto now


        add_test(
                NAME ${TEST_NAME}
                COMMAND "${TEST_NAME}" "${INPUT_ARGS}"
                WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
        )

        set_target_properties(${TEST_NAME} PROPERTIES FOLDER tests)
    endmacro()
else ()
    macro(add_unittest
            TEST_NAME TEST_SRC CFLAGS LINK_LIBS LINK_LIB_FLAGS INPUT_ARGS)
        add_executable(${TEST_NAME}
                ${TEST_SRC})

        target_compile_options(${TEST_NAME}
                PRIVATE ${CFLAGS}
                PRIVATE ${OMP_FLAGS}
        )

        target_include_directories(${TEST_NAME}
                PRIVATE ${QLMPS_HEADER_PATH}
                PRIVATE ${QLMPS_TENSOR_LIB_HEADER_PATH}
                PRIVATE ${hptt_INCLUDE_DIR}
                PRIVATE ${Boost_INCLUDE_DIR}
                PRIVATE ${MPI_CXX_HEADER_DIR})
        target_link_libraries(${TEST_NAME}
                GTest::GTest GTest::Main
                ${hptt_LIBRARY}
                Boost::serialization
                Boost::mpi
                ${MPI_CXX_LINK_FLAGS}
                ${MPI_mpi_LIBRARY}
                ${LINK_LIBS} "${LINK_LIB_FLAGS}")

        add_test(
                NAME ${TEST_NAME}
                COMMAND "${TEST_NAME}" "${INPUT_ARGS}"
                WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
        )

        set_target_properties(${TEST_NAME} PROPERTIES FOLDER tests)
    endmacro()


    macro(add_mpi_unittest
            TEST_NAME TEST_SRC CFLAGS LINK_LIBS LINK_LIB_FLAGS PROCESSOR_NUM)
        add_executable(${TEST_NAME}
                ${TEST_SRC})

        target_compile_options(${TEST_NAME}
                PRIVATE ${CFLAGS}
                PRIVATE ${OMP_FLAGS}
        )
        target_include_directories(${TEST_NAME}
                PRIVATE ${QLMPS_HEADER_PATH}
                PRIVATE ${QLMPS_TENSOR_LIB_HEADER_PATH}
                PRIVATE ${hptt_INCLUDE_DIR}
                PRIVATE ${Boost_INCLUDE_DIR}
                PRIVATE ${MPI_CXX_HEADER_DIR})
        target_link_libraries(${TEST_NAME}
                GTest::GTest GTest::Main
                ${hptt_LIBRARY}
                Boost::serialization
                Boost::mpi
                ${MPI_CXX_LINK_FLAGS}
                ${MPI_mpi_LIBRARY}
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


    macro(compile_unittest
            TEST_NAME TEST_SRC CFLAGS LINK_LIBS LINK_LIB_FLAGS)
        add_executable(${TEST_NAME}
                ${TEST_SRC})

        target_compile_options(${TEST_NAME}
                PRIVATE ${CFLAGS}
                PRIVATE ${OMP_FLAGS}
        )
        target_include_directories(${TEST_NAME}
                PRIVATE ${QLMPS_HEADER_PATH}
                PRIVATE ${QLMPS_TENSOR_LIB_HEADER_PATH}
                PRIVATE ${hptt_INCLUDE_DIR}
                PRIVATE ${Boost_INCLUDE_DIR}
                PRIVATE ${MPI_CXX_HEADER_DIR})
        target_link_libraries(${TEST_NAME}
                GTest::GTest GTest::Main
                ${hptt_LIBRARY}
                Boost::serialization
                Boost::mpi
                ${MPI_CXX_LINK_FLAGS}
                ${MPI_mpi_LIBRARY}
                "${LINK_LIBS}" "${LINK_LIB_FLAGS}"
        )
        set_target_properties(${TEST_NAME} PROPERTIES FOLDER tests)
    endmacro()
endif ()

## Test sites vector.
add_unittest(test_site_vec test_site_vec.cc "${MATH_LIB_COMPILE_FLAGS}" "" "${MATH_LIB_LINK_FLAGS}" "")

## Test one-dimensional tensor networks.
# Test DuoVector class
add_unittest(test_duovector test_one_dim_tn/test_duovector.cc "" "" "" "")
# Test TenVec class
add_unittest(test_ten_vec
        "test_one_dim_tn/test_ten_vec.cc"
        "${MATH_LIB_COMPILE_FLAGS}" "" "${MATH_LIB_LINK_FLAGS}" ""
)

# Test MPS class
add_unittest(test_finite_mps
        "test_one_dim_tn/test_finite_mps/test_finite_mps.cc"
        "${MATH_LIB_COMPILE_FLAGS}" "" "${MATH_LIB_LINK_FLAGS}" ""
)
# Test MPS measurement.
add_unittest(test_finite_mps_measu
        "test_one_dim_tn/test_finite_mps/test_finite_mps_measu.cc"
        "${MATH_LIB_COMPILE_FLAGS}" "" "${MATH_LIB_LINK_FLAGS}" ""
)

# Test MPO class
add_unittest(test_mpo
        "test_one_dim_tn/test_mpo.cc"
        "${MATH_LIB_COMPILE_FLAGS}" "" "${MATH_LIB_LINK_FLAGS}" ""
)
# Test MPO generation.
add_unittest(test_mpogen_coef_op_alg
        "test_one_dim_tn/test_mpogen/test_mpogen_coef_op_alg.cc"
        "${MATH_LIB_COMPILE_FLAGS}" "" "${MATH_LIB_LINK_FLAGS}" ""
)
add_unittest(test_mpogen_fsm
        "test_one_dim_tn/test_mpogen/test_mpogen_fsm.cc"
        "${MATH_LIB_COMPILE_FLAGS}" "" "${MATH_LIB_LINK_FLAGS}" ""
)
add_unittest(test_mpogen
        "test_one_dim_tn/test_mpogen/test_mpogen.cc"
        "${MATH_LIB_COMPILE_FLAGS}" "" "${MATH_LIB_LINK_FLAGS}" ""
)
add_unittest(test_impo_gen
        "test_one_dim_tn/test_mpogen/test_impogen.cc"
        "${MATH_LIB_COMPILE_FLAGS}" "" "${MATH_LIB_LINK_FLAGS}" ""
)


## Test algorithms
## Test simulation case parameters parser.
add_unittest(test_case_params_parser
        test_case_params_parser.cc
        "${MATH_LIB_COMPILE_FLAGS}" "" "${MATH_LIB_LINK_FLAGS}"
        "${CMAKE_CURRENT_LIST_DIR}/testdata/test-params.json" "")
# Lanczos solver
add_unittest(test_vmps_lanczos_solver
        "test_algorithm/test_lanczos_solver.cc"
        "${MATH_LIB_COMPILE_FLAGS}" "" "${MATH_LIB_LINK_FLAGS}" ""
)

# Lanczos solver
add_unittest(test_expmv_lanczos_solver
        "test_algorithm/test_lanczos_expmv_solver.cc"
        "${MATH_LIB_COMPILE_FLAGS}" "" "${MATH_LIB_LINK_FLAGS}" ""
)

# Two-site update finite vMPS
add_unittest(test_two_site_finite_vmps
        "test_algorithm/test_two_site_update_finite_vmps.cc"
        "${MATH_LIB_COMPILE_FLAGS}" "" "${MATH_LIB_LINK_FLAGS}" ""
)
# Single-site update finite vMPS
add_unittest(test_single_site_finite_vmps
        "test_algorithm/test_single_site_update_finite_vmps.cc"
        "${MATH_LIB_COMPILE_FLAGS}" "" "${MATH_LIB_LINK_FLAGS}" ""
)

# Two-site noised update finite vMPS
add_unittest(test_two_site_finite_vmps_with_noise
        "test_algorithm/test_two_site_noise_update_finite_vmps.cc"
        "${MATH_LIB_COMPILE_FLAGS}" "" "${MATH_LIB_LINK_FLAGS}" ""
)
# Two-site update finite DMRG
add_unittest(test_finite_dmrg
        "test_algorithm/test_finite_dmrg.cc"
        "${MATH_LIB_COMPILE_FLAGS}" "" "${MATH_LIB_LINK_FLAGS}" ""
)
add_unittest(test_fermion_two_site_update_finite_vmps
        "test_algorithm/test_fermion_two_site_update_finite_vmps.cc"
        "${MATH_LIB_COMPILE_FLAGS}" "" "${MATH_LIB_LINK_FLAGS}" ""
)
# Test two site TDVP
add_unittest(test_two_site_finite_tdvp
        "test_algorithm/test_two_site_finite_tdvp.cc"
        "${MATH_LIB_COMPILE_FLAGS}" "" "${MATH_LIB_LINK_FLAGS}" ""
)


if (NOT QLMPS_USE_GPU)
    # Test mpi two site update
    add_mpi_unittest(test_mpi_two_site_update_finite_vmps
            "test_mpi_algorithm/test_mpi_two_site_update_finite_vmps.cc"
            "${MATH_LIB_COMPILE_FLAGS}" "" "${MATH_LIB_LINK_FLAGS}" "4"
    )

    # Test mpi two site noised update
    add_mpi_unittest(test_mpi_two_site_update_noised_finite_vmps
            "test_mpi_algorithm/test_mpi_two_site_update_noised_finite_vmps.cc"
            "${MATH_LIB_COMPILE_FLAGS}" "" "${MATH_LIB_LINK_FLAGS}" "4"
    )

    # Test mpi two site TDVP
    add_mpi_unittest(test_mpi_two_site_finite_tdvp
            "test_mpi_algorithm/test_mpi_two_site_finite_tdvp.cc"
            "${MATH_LIB_COMPILE_FLAGS}" "" "${MATH_LIB_LINK_FLAGS}" "4"
    )

    # Test mpi two site DMRG
    add_mpi_unittest(test_mpi_finite_dmrg
            "test_mpi_algorithm/test_mpi_dmrg.cc"
            "${MATH_LIB_COMPILE_FLAGS}" "" "${MATH_LIB_LINK_FLAGS}" "4"
    )
endif ()