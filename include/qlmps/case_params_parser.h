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
    source_file_path_ = file != nullptr ? std::string(file) : std::string();
    if (!ifs.good()) {
      std::cerr << "Failed to open input JSON file: '" << source_file_path_ << "'" << std::endl;
      exit(1);
    }
    json raw_json;
    ifs >> raw_json;
    ifs.close();
    if (raw_json.find(kCaseParamsJsonObjName) != raw_json.end()) {
      case_params_ = raw_json[kCaseParamsJsonObjName];
    } else {
      std::cerr << "JSON object '" << kCaseParamsJsonObjName
                << "' not found in file '" << source_file_path_ << "'" << std::endl;
      exit(1);
    }
  }

  /// Parse a int parameter.
  int ParseInt(
      const std::string &item     ///< Parameter key.
  ) {
    return StrictParse<int>(item, "integer");
  }

  /// Parse a float parameter.
  double ParseDouble(
      const std::string &item     ///< Parameter key.
  ) {
    return StrictParse<double>(item, "number");
  }

  /// Parse a char parameter.
  char ParseChar(
      const std::string &item     ///< Parameter key.
  ) {
    auto char_str = StrictParse<std::string>(item, "string");
    if (char_str.empty()) return '\0';
    return char_str.at(0);
  }

  /// Parse a std::string parameter.
  std::string ParseStr(
      const std::string &item     ///< Parameter key.
  ) {
    return StrictParse<std::string>(item, "string");
  }

  /// Parse a bool parameter.
  bool ParseBool(
      const std::string &item     ///< Parameter key.
  ) {
    return StrictParse<bool>(item, "boolean");
  }

  /// Parse a std::vector<int> parameter.
  std::vector<int> ParseIntVec(
      const std::string &item
  ) {
    return StrictParse<std::vector<int>>(item, "array(integer)");
  }

  /// Parse a std::vector<int> parameter.
  std::vector<size_t> ParseSizeTVec(
      const std::string &item
  ) {
    return StrictParse<std::vector<size_t>>(item, "array(unsigned integer)");
  }

  /// Parse a std::vector<double> parameter.
  std::vector<double> ParseDoubleVec(
      const std::string &item    ///< Parameter key.
  ) {
    return StrictParse<std::vector<double>>(item, "array(number)");
  }

  /// Check if a parameter exists.
  bool Has(
      const std::string &item
  ) const {
    return case_params_.find(item) != case_params_.end();
  }

  // Parse with default (only used when key is missing)
  int ParseIntOr(
      const std::string &item,
      int default_value
  ) {
    if (!Has(item)) return default_value;
    return ParseInt(item);
  }

  double ParseDoubleOr(
      const std::string &item,
      double default_value
  ) {
    if (!Has(item)) return default_value;
    return ParseDouble(item);
  }

  char ParseCharOr(
      const std::string &item,
      char default_value
  ) {
    if (!Has(item)) return default_value;
    return ParseChar(item);
  }

  std::string ParseStrOr(
      const std::string &item,
      const std::string &default_value
  ) {
    if (!Has(item)) return default_value;
    return ParseStr(item);
  }

  bool ParseBoolOr(
      const std::string &item,
      bool default_value
  ) {
    if (!Has(item)) return default_value;
    return ParseBool(item);
  }

  std::vector<int> ParseIntVecOr(
      const std::string &item,
      const std::vector<int> &default_value
  ) {
    if (!Has(item)) return default_value;
    return ParseIntVec(item);
  }

  std::vector<size_t> ParseSizeTVecOr(
      const std::string &item,
      const std::vector<size_t> &default_value
  ) {
    if (!Has(item)) return default_value;
    return ParseSizeTVec(item);
  }

  std::vector<double> ParseDoubleVecOr(
      const std::string &item,
      const std::vector<double> &default_value
  ) {
    if (!Has(item)) return default_value;
    return ParseDoubleVec(item);
  }

  // TryParse APIs (do not exit; return false on failure)
  bool TryParseInt(const std::string &item, int &out_value) const {
    return TryParseStrict<int>(item, out_value);
  }

  bool TryParseDouble(const std::string &item, double &out_value) const {
    return TryParseStrict<double>(item, out_value);
  }

  bool TryParseChar(const std::string &item, char &out_value) const {
    std::string tmp;
    if (!TryParseStrict<std::string>(item, tmp)) return false;
    out_value = tmp.empty() ? '\0' : tmp.at(0);
    return true;
  }

  bool TryParseStr(const std::string &item, std::string &out_value) const {
    return TryParseStrict<std::string>(item, out_value);
  }

  bool TryParseBool(const std::string &item, bool &out_value) const {
    return TryParseStrict<bool>(item, out_value);
  }

  bool TryParseIntVec(const std::string &item, std::vector<int> &out_value) const {
    return TryParseStrict<std::vector<int>>(item, out_value);
  }

  bool TryParseSizeTVec(const std::string &item, std::vector<size_t> &out_value) const {
    // Manually validate to reject floats and negative numbers
    if (case_params_.find(item) == case_params_.end()) return false;
    try {
      const json &value = case_params_.at(item);
      if (!value.is_array()) return false;
      std::vector<size_t> parsed;
      parsed.reserve(value.size());
      for (const auto &el : value) {
        if (el.is_number_unsigned()) {
          parsed.push_back(el.get<size_t>());
        } else if (el.is_number_integer()) {
          long long v = el.get<long long>();
          if (v < 0) return false;
          parsed.push_back(static_cast<size_t>(v));
        } else {
          // Reject floating numbers and non-number types
          return false;
        }
      }
      out_value = std::move(parsed);
      return true;
    } catch (...) {
      return false;
    }
  }

  bool TryParseDoubleVec(const std::string &item, std::vector<double> &out_value) const {
    return TryParseStrict<std::vector<double>>(item, out_value);
  }
 private:
  template <typename T>
  T StrictParse(const std::string &item, const char *expected_type_name) const {
    if (case_params_.find(item) == case_params_.end()) {
      ReportAndExitMissing(item, expected_type_name);
    }
    try {
      return case_params_.at(item).get<T>();
    } catch (const std::exception &e) {
      const char *actual = case_params_.at(item).type_name();
      ReportAndExitMissingOrInvalid(item, expected_type_name, actual);
    }
    return T();
  }

  template <typename T>
  bool TryParseStrict(const std::string &item, T &out_value) const {
    if (case_params_.find(item) == case_params_.end()) return false;
    try {
      out_value = case_params_.at(item).get<T>();
      return true;
    } catch (...) {
      return false;
    }
  }

  void ReportAndExitMissing(const std::string &item, const char *expected) const {
    std::cerr << "CaseParams parse error in file '" << source_file_path_ << "':" << std::endl
              << "  key '" << item << "' is missing (expected " << expected << ")" << std::endl;
    exit(1);
  }

  void ReportAndExitMissingOrInvalid(const std::string &item,
                                     const char *expected,
                                     const char *actual) const {
    std::cerr << "CaseParams parse error in file '" << source_file_path_ << "':" << std::endl
              << "  key '" << item << "' has invalid type" << std::endl
              << "    expected: " << expected << std::endl
              << "    actual:   " << actual << std::endl;
    exit(1);
  }

  json case_params_;
  std::string source_file_path_;
};
} /* qlmps */
#endif /* ifndef QLMPS_CASE_PARAMS_PARSER_H */
