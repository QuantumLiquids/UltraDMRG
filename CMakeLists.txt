#  SPDX-License-Identifier: LGPL-3.0-only
# 
#  Author: Hao-Xin Wang <wanghaoxin1996@gmail.com>
#  Creation Date: Jan, 31th, 2024
#  
#  Description: QuantumLiquids/UltraDMRG project. Main CMake file.
# 
cmake_minimum_required(VERSION 3.12)


project("QuantumLiquids/UltraDMRG"
        DESCRIPTION "A large-scale matrix product state(MPS) algorithms library"
        LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

# Build type control.
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif ()


# Build output control.
if (${CMAKE_BUILD_TYPE} MATCHES Debug)
    set(CMAKE_VERBOSE_MAKEFILE ON)
endif ()

option(QLMPS_USE_GPU "USE CUDA." OFF)
option(QLMPS_BUILD_UNITTEST "Build unittests for UltraDMRG." OFF)
option(QLMPS_BUILD_PROFILER "Build profiler test cases for UltraDMRG." OFF)
option(QLMPS_BUILD_BENCHMARK "Build benchmark on the speed of 2D Hubbard DMRG." OFF)

# Advanced options.
option(QLMPS_MPI_TIMING_MODE "MPI Timing mode." OFF)
option(QLMPS_TIMING_MODE "Timing mode." OFF)
option(QLMPS_MPI_TIMING_MODE "MPI Timing mode." OFF)


if (QLMPS_TIMING_MODE)
    add_definitions(-DQLMPS_TIMING_MODE)
endif ()
if (QLMPS_MPI_TIMING_MODE)
    add_definitions(-DQLMPS_MPI_TIMING_MODE)
endif ()

# Compilation and linking control.
add_definitions(-Wall -g)

# Installation path
set(QLMPS_TOOL_SCRIPT_DEST "${CMAKE_INSTALL_PREFIX}/bin")
set(QLMPS_INCLUDE_DEST "${CMAKE_INSTALL_PREFIX}/include")


# Build(Install) library.
set(QLMPS_HEADER_PATH "${PROJECT_SOURCE_DIR}/include")
set(QLMPS_HEADER_DIR ${QLMPS_HEADER_PATH}/qlmps)
install(DIRECTORY ${QLMPS_HEADER_DIR}
        DESTINATION ${QLMPS_INCLUDE_DEST})


# Build tool scripts
add_subdirectory(script)

# Build unittests.
if (QLMPS_BUILD_UNITTEST)
    find_path(QLTEN_HEADER_PATH "qlten")
    enable_testing()
    find_package(GTest REQUIRED)
    add_subdirectory(tests tests)
endif ()

if (QLMPS_BUILD_PROFILER)
    find_path(QLTEN_HEADER_PATH "qlten")
    find_path(hptt_INCLUDE_DIR "hptt.h")
    find_library(hptt_LIBRARY "libhptt.a")
    enable_testing()
    find_path(PROFILER_INCLUDE_DIR "gperftools")
    find_library(LIBPROFILER_LIBRARY "libprofiler.a")
    add_subdirectory(profiler profiler)
endif ()

if (QLMPS_BUILD_BENCHMARK)
    find_path(QLTEN_HEADER_PATH "qlten")
    find_path(hptt_INCLUDE_DIR "hptt.h")
    find_library(hptt_LIBRARY "libhptt.a")
    enable_testing()
    add_subdirectory(benchmark benchmark)
endif ()