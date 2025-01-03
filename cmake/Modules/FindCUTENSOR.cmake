#  SPDX-License-Identifier: LGPL-3.0-only
# 
#  Author: Hao-Xin Wang <wanghaoxin1996@gmail.com>
#  Creation Date: 2024-11-28
#  
#  Description: QuantumLiquids/tensor project. CMake module to find cutensor library.
# 

# Set default CUTENSOR_ROOT to CMAKE_INSTALL_PREFIX if not already defined
if (NOT DEFINED CUTENSOR_ROOT)
    set(CUTENSOR_ROOT ${CMAKE_INSTALL_PREFIX})
endif ()

find_path(CUTENSOR_INCLUDE_DIR
        NAMES cutensor.h
        PATHS ${CUTENSOR_ROOT}/include
        DOC "Path to cuTENSOR include directory"
)

# Dynamically find the versioned libcutensor directory
file(GLOB VERSIONED_LIB_DIRS "${CUTENSOR_ROOT}/lib64/libcutensor/*")

# Filter to directories that contain valid libraries
set(CUTENSOR_LIB_PATHS "")
foreach(version_dir ${VERSIONED_LIB_DIRS})
    if(EXISTS "${version_dir}/libcutensor.so" OR EXISTS "${version_dir}/libcutensorMg.so")
        list(APPEND CUTENSOR_LIB_PATHS "${version_dir}")
    endif()
endforeach()

# Fallback: Add default lib64 path in case no version directories are detected
if(NOT CUTENSOR_LIB_PATHS)
    list(APPEND CUTENSOR_LIB_PATHS "${CUTENSOR_ROOT}/lib64/")
endif()

# Locate the main library
find_library(CUTENSOR_LIBRARY
        NAMES cutensor
        PATHS ${CUTENSOR_LIB_PATHS}
        DOC "Path to cuTENSOR main library"
)

# Locate the auxiliary library (cutensorMg if available)
find_library(CUTENSOR_MG_LIBRARY
        NAMES cutensorMg
        PATHS ${CUTENSOR_LIB_PATHS}
        DOC "Path to cuTENSOR Mg library"
)

# Mark CUTENSOR as found if libraries and include directory are available
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(CUTENSOR
        REQUIRED_VARS CUTENSOR_INCLUDE_DIR CUTENSOR_LIBRARY
        VERSION_VAR CUTENSOR_VERSION
)

# Provide imported targets for easier use in CMake
if(CUTENSOR_FOUND)
    add_library(CUTENSOR::CUTENSOR SHARED IMPORTED
            ../../include/qlten/framework/mem_ops.h
            ../../tests/test_utility/hp_numeric.h)
    set_target_properties(CUTENSOR::CUTENSOR PROPERTIES
            IMPORTED_LOCATION ${CUTENSOR_LIBRARY}
            INTERFACE_INCLUDE_DIRECTORIES ${CUTENSOR_INCLUDE_DIR}
    )

    if(CUTENSOR_MG_LIBRARY)
        add_library(CUTENSOR::CUTENSOR_MG SHARED IMPORTED)
        set_target_properties(CUTENSOR::CUTENSOR_MG PROPERTIES
                IMPORTED_LOCATION ${CUTENSOR_MG_LIBRARY}
                INTERFACE_INCLUDE_DIRECTORIES ${CUTENSOR_INCLUDE_DIR}
        )
    endif()
endif()

# Provide useful messages
if(CUTENSOR_FOUND)
    message(STATUS "Found CUTENSOR: ${CUTENSOR_LIBRARY}")
    message(STATUS "CUTENSOR Mg library (if available): ${CUTENSOR_MG_LIBRARY}")
    message(STATUS "CUTENSOR include directory: ${CUTENSOR_INCLUDE_DIR}")
else()
    message(WARNING "cuTENSOR not found. Set CUTENSOR_ROOT to the cuTENSOR installation directory.")
endif()

# Make CUTENSOR available for the main CMakeLists.txt
mark_as_advanced(CUTENSOR_INCLUDE_DIR CUTENSOR_LIBRARY)
