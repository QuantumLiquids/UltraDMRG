#  SPDX-License-Identifier: LGPL-3.0-only
#
#  Author: Hao-Xin Wang <wanghaoxin1996@gmail.com>
#  Creation Date: 2019-05-12 10:15
#
#  Description: QuantumLiquids/UltraDMRG project. CMake file to control unittest.
#
# Set MKL compile flags and link flags.

if (APPLE)
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
        1.76.0
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


macro(add_profiler_case
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
            PRIVATE ${PROFILER_INCLUDE_DIR}
            PRIVATE ${hptt_INCLUDE_DIR}
            PRIVATE ${Boost_INCLUDE_DIR}
            PRIVATE ${MPI_CXX_HEADER_DIR})
    target_link_libraries(${TEST_NAME}
            ${LIBPROFILER_LIBRARY}
            ${hptt_LIBRARY}
            Boost::serialization
            Boost::mpi
            ${MPI_CXX_LINK_FLAGS}
            ${MPI_mpi_LIBRARY}
            "${LINK_LIBS}" "${LINK_LIB_FLAGS}"
    )
    add_test(
            NAME ${TEST_NAME}
            COMMAND "${TEST_NAME}" "${INPUT_ARGS}"
            WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
    )

    set_target_properties(${TEST_NAME} PROPERTIES FOLDER tests)
endmacro()


# Test single processor matrix multiply vector performance
add_profiler_case(single_processor_lanczos_performance_test
        "../profiler/single_processor_lanczos_performance_test.cc"
        "${MATH_LIB_COMPILE_FLAGS}" "" "${MATH_LIB_LINK_FLAGS}" "--thread_nums=1,2"
)


## Test single processor svd performance
#compile_unittest(single_processor_svd_performance_test
#  "test_mpi_algorithm/single_processor_svd_performance_test.cc"
#  "${MATH_LIB_COMPILE_FLAGS}" "" "${MATH_LIB_LINK_FLAGS}"
#)
#above test include nested omp so just neglect now