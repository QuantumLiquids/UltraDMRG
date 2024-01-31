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
} /* qlmps */
#endif /* ifndef QLMPS_CONSTS_H */
