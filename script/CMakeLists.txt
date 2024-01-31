#  SPDX-License-Identifier: LGPL-3.0-only
#
#  Author: Hao-Xin Wang <wanghaoxin1996@gmail.com>
#  Creation Date: 2019-05-12 10:15
#
#  Description: QuantumLiquids/UltraDMRG project. CMake file to control tool scripts.
#
# Set MKL compile flags and link flags.
set(QLMPS_TOOL_SCRIPT_PATH "${PROJECT_SOURCE_DIR}/script")
set(QLMPS_TOOL_SCRIPTS
        ${QLMPS_TOOL_SCRIPT_PATH}/mps_timing_analyser.py
        ${QLMPS_TOOL_SCRIPT_PATH}/mps_log_simplifier.sh)


install(
        FILES ${QLMPS_TOOL_SCRIPTS}
        DESTINATION "${QLMPS_TOOL_SCRIPT_DEST}")
