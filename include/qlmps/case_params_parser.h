// SPDX-License-Identifier: LGPL-3.0-only

/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2020-07-28 16:24
* 
* Description: QuantumLiquids/UltraDMRG project. Simulation case parameters parser.
*/

/**
@file case_params_parser.h
@brief Simulation case parameters parser.
*/
#ifndef QLMPS_CASE_PARAMS_PARSER_H
#define QLMPS_CASE_PARAMS_PARSER_H

#include "qlmps/consts.h"

#include <iostream>
#include <fstream>                                  // ifstream

#include "qlmps/third_party/nlohmann/json.hpp"     // json

namespace qlmps {

/**
Basic simulation case parameter parser.

@since version 0.0.0
*/
class CaseParamsParserBasic {
 public:
  using json = nlohmann::json;

  /**
  Create simulation case parameters parser. Read the input file as a JSON file
  and parse the contained simulation case parameters JSON object.

  @param file Path of the to be parsed file. For example, `argv[1]`.

  @since version 0.0.0
  */
  CaseParamsParserBasic(
      const char *file
  ) {
    std::ifstream ifs(file);
    json raw_json;
    ifs >> raw_json;
    ifs.close();
    if (raw_json.find(kCaseParamsJsonObjName) != raw_json.end()) {
      case_params_ = raw_json[kCaseParamsJsonObjName];
    } else {
      std::cout << kCaseParamsJsonObjName
                << " object not found, exit!"
                << std::endl;
      exit(1);
    }
  }

  /// Parse a int parameter.
  int ParseInt(
      const std::string &item     ///< Parameter key.
  ) {
    return case_params_[item].get<int>();
  }

  /// Parse a float parameter.
  double ParseDouble(
      const std::string &item     ///< Parameter key.
  ) {
    return case_params_[item].get<double>();
  }

  /// Parse a char parameter.
  char ParseChar(
      const std::string &item     ///< Parameter key.
  ) {
    auto char_str = case_params_[item].get<std::string>();
    return char_str.at(0);
  }

  /// Parse a std::string parameter.
  std::string ParseStr(
      const std::string &item     ///< Parameter key.
  ) {
    return case_params_[item].get<std::string>();
  }

  /// Parse a bool parameter.
  bool ParseBool(
      const std::string &item     ///< Parameter key.
  ) {
    return case_params_[item].get<bool>();
  }

  /// Parse a std::vector<int> parameter.
  std::vector<int> ParseIntVec(
      const std::string &item
  ) {
    return case_params_[item].get<std::vector<int>>();
  }

  /// Parse a std::vector<int> parameter.
  std::vector<size_t> ParseSizeTVec(
      const std::string &item
  ) {
    return case_params_[item].get<std::vector<size_t>>();
  }

  /// Parse a std::vector<double> parameter.
  std::vector<double> ParseDoubleVec(
      const std::string &item    ///< Parameter key.
  ) {
    return case_params_[item].get<std::vector<double>>();
  }
 private:
  json case_params_;
};
} /* qlmps */
#endif /* ifndef QLMPS_CASE_PARAMS_PARSER_H */
