// SPDX-License-Identifier: LGPL-3.0-only

/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2020-07-28 15:45
* 
* Description: QuantumLiquids/UltraDMRG project. Constant declarations.
*/

/**
@file consts.h
@brief Constant declarations.
*/
#ifndef QLMPS_CONSTS_H
#define QLMPS_CONSTS_H

#include <string>     // string
#include <vector>     // vector
#include <cmath>

#ifdef USE_GPU
#include <cuda/std/complex>
#include <cuda_runtime.h>
#else
#include <complex>
#endif

namespace qlmps {

/// JSON object name of the simulation case parameter parsed by @link qlmps::CaseParamsParserBasic `CaseParamsParser` @endlink.
const std::string kCaseParamsJsonObjName = "CaseParams";

const std::string kMpsPath = "mps";
const std::string kMpoPath = "mpo";
const std::string kRuntimeTempPath = ".temp";
const std::string kEnvFileBaseName = "env";
const std::string kMpsTenBaseName = "mps_ten";
const std::string kMpoTenBaseName = "mpo";
const std::string kOpFileBaseName = "op";

const int kLanczEnergyOutputPrecision = 16;

const std::vector<size_t> kNullUintVec;
const std::vector<std::vector<size_t>> kNullUintVecVec;

#ifdef USE_GPU
__device__ __host__
inline cuda::std::complex<double> complex_exp(const cuda::std::complex<double>& z) {
    double x = z.real();
    double y = z.imag();
    double e_x = exp(x); // Exponential part for real value
    return cuda::std::complex<double>(e_x * cos(y), e_x * sin(y));
}
#else
inline std::complex<double> complex_exp(const std::complex<double> &z) {
  return std::exp(z);
}
#endif
} /* qlmps */
#endif /* ifndef QLMPS_CONSTS_H */
