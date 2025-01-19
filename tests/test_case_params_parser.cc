// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2019-06-04 20:20
* 
* Description: QuantumLiquids/MPS project. Unittests for simulation case parameters parser.
*/

#include "gtest/gtest.h"
#include "qlmps/case_params_parser.h"

using namespace qlmps;

// Input arguments.
char *json_file;

// Custom case parameter parser.
struct CustomCaseParamsParser : public CaseParamsParserBasic {
  CustomCaseParamsParser(const char *f) : CaseParamsParserBasic(f) {
    case_int = ParseInt("Int");
    case_double = ParseDouble("Double");
    case_char = ParseChar("Char");
    case_str = ParseStr("String");
    case_bool = ParseBool("Boolean");
    case_int_vec = ParseIntVec("IntVec");
    case_size_t_vec = ParseSizeTVec("SizeTVec");
    case_double_vec = ParseDoubleVec("DoubleVec");
  }

  int case_int;
  double case_double;
  char case_char;
  std::string case_str;
  bool case_bool;
  std::vector<int> case_int_vec;
  std::vector<size_t> case_size_t_vec;
  std::vector<double> case_double_vec;
};

TEST(TestCaseParamsParser, Case1) {
  CustomCaseParamsParser params(json_file);
  EXPECT_EQ(params.case_int, 1);
  EXPECT_DOUBLE_EQ(params.case_double, 2.33);
  EXPECT_EQ(params.case_char, 'c');
  EXPECT_EQ(params.case_str, "string");
  EXPECT_EQ(params.case_bool, false);
  EXPECT_EQ(params.case_size_t_vec, std::vector<size_t>({4, 7, 10}));
  EXPECT_EQ(params.case_int_vec, std::vector<int>({1, -1, 3}));
  EXPECT_EQ(params.case_double_vec, std::vector<double>({0.1, 0.02, 0.003}));
}

int main(int argc, char *argv[]) {
  testing::InitGoogleTest(&argc, argv);
  json_file = argv[1];
  return RUN_ALL_TESTS();
}
