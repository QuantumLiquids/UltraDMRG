#  SPDX-License-Identifier: LGPL-3.0-only
# 
#  Author: Hao-Xin Wang <wanghaoxin1996@gmail.com>
#  Creation Date: Aug, 13th, 2024
#
#  Description: QuantumLiquids/UltraDMRG project. CMake file to compile the benchmark file
# 
# Set MKL compile flags and link flags.

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
            PRIVATE ${MPI_CXX_HEADER_DIR})
    target_link_libraries(${TEST_NAME}
            ${hptt_LIBRARY}
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
            PRIVATE ${MPI_CXX_HEADER_DIR})
    target_link_libraries(${TEST_NAME}
            ${hptt_LIBRARY}
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
            PRIVATE ${MPI_CXX_HEADER_DIR})
    target_link_libraries(${TEST_NAME}
            GTest::GTest GTest::Main
            ${hptt_LIBRARY}
            ${MPI_CXX_LINK_FLAGS}
            ${MPI_mpi_LIBRARY}
            "${LINK_LIBS}" "${LINK_LIB_FLAGS}"
    )
    set_target_properties(${TEST_NAME} PROPERTIES FOLDER tests)
endmacro()


add_unittest(ultradmrg_hubbard2d
        "ultradmrg_hubbard2d.cpp"
        "${MATH_LIB_COMPILE_FLAGS}" "" "${MATH_LIB_LINK_FLAGS}" ""
)