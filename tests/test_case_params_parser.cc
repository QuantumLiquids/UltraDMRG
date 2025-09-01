// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: UltraDMRG Maintainers
* 
* Description: Comprehensive tests for simulation case parameters parser.
*/

#include "gtest/gtest.h"
#include "qlmps/case_params_parser.h"

using namespace qlmps;

// Input arguments: ok_json, wrong_types_json
static const char *json_ok_file;
static const char *json_wrong_types_file;

TEST(TestCaseParamsParser, Combined) {
  // OK data: defaults, try-parse, and strict parse via derived class
  {
    CaseParamsParserBasic parser(json_ok_file);

    // Has
    EXPECT_TRUE(parser.Has("Int"));
    EXPECT_FALSE(parser.Has("MissingInt"));

    // Defaults when missing
    EXPECT_EQ(parser.ParseIntOr("MissingInt", 42), 42);
    EXPECT_TRUE(parser.ParseBoolOr("MissingBool", true));
    EXPECT_EQ(parser.ParseStrOr("MissingStr", std::string("def")), std::string("def"));
    std::vector<int> def_iv = {9, 8};
    EXPECT_EQ(parser.ParseIntVecOr("MissingIntVec", def_iv), def_iv);

    // TryParse success
    int int_val; double dbl_val; char ch_val; std::string str_val; bool bool_val;
    EXPECT_TRUE(parser.TryParseInt("Int", int_val));
    EXPECT_EQ(int_val, 1);
    EXPECT_TRUE(parser.TryParseDouble("Double", dbl_val));
    EXPECT_DOUBLE_EQ(dbl_val, 2.33);
    EXPECT_TRUE(parser.TryParseChar("Char", ch_val));
    EXPECT_EQ(ch_val, 'c');
    EXPECT_TRUE(parser.TryParseStr("String", str_val));
    EXPECT_EQ(str_val, std::string("string"));
    EXPECT_TRUE(parser.TryParseBool("Boolean", bool_val));
    EXPECT_FALSE(bool_val);

    std::vector<int> iv_val; std::vector<size_t> sv_val; std::vector<double> dv_val;
    EXPECT_TRUE(parser.TryParseIntVec("IntVec", iv_val));
    EXPECT_EQ(iv_val, std::vector<int>({1, -1, 3}));
    EXPECT_TRUE(parser.TryParseSizeTVec("SizeTVec", sv_val));
    EXPECT_EQ(sv_val, std::vector<size_t>({4, 7, 10}));
    EXPECT_TRUE(parser.TryParseDoubleVec("DoubleVec", dv_val));
    EXPECT_EQ(dv_val, std::vector<double>({0.1, 0.02, 0.003}));

    // Missing keys -> TryParse false
    int miss_int;
    EXPECT_FALSE(parser.TryParseInt("MissingInt", miss_int));

    // Backward compatibility: strict Parse in a derived parser
    struct CustomCaseParamsParser : public CaseParamsParserBasic {
      using CaseParamsParserBasic::CaseParamsParserBasic;
      int case_int; double case_double; char case_char; std::string case_str; bool case_bool;
      std::vector<int> case_int_vec; std::vector<size_t> case_size_t_vec; std::vector<double> case_double_vec;
      void Init() {
        case_int = ParseInt("Int");
        case_double = ParseDouble("Double");
        case_char = ParseChar("Char");
        case_str = ParseStr("String");
        case_bool = ParseBool("Boolean");
        case_int_vec = ParseIntVec("IntVec");
        case_size_t_vec = ParseSizeTVec("SizeTVec");
        case_double_vec = ParseDoubleVec("DoubleVec");
      }
    };
    CustomCaseParamsParser params(json_ok_file);
    params.Init();
    EXPECT_EQ(params.case_int, 1);
    EXPECT_DOUBLE_EQ(params.case_double, 2.33);
    EXPECT_EQ(params.case_char, 'c');
    EXPECT_EQ(params.case_str, "string");
    EXPECT_EQ(params.case_bool, false);
    EXPECT_EQ(params.case_size_t_vec, std::vector<size_t>({4, 7, 10}));
    EXPECT_EQ(params.case_int_vec, std::vector<int>({1, -1, 3}));
    EXPECT_EQ(params.case_double_vec, std::vector<double>({0.1, 0.02, 0.003}));
  }

  // Wrong types: TryParse returns false, except empty string for Char -> '\0'
  {
    CaseParamsParserBasic parser(json_wrong_types_file);
    int i; double d; char c; std::string s; bool b;
    std::vector<int> iv; std::vector<size_t> sv; std::vector<double> dv;

    EXPECT_FALSE(parser.TryParseInt("Int", i));
    EXPECT_FALSE(parser.TryParseDouble("Double", d));
    EXPECT_TRUE(parser.TryParseChar("Char", c));
    EXPECT_EQ(c, '\0');
    EXPECT_FALSE(parser.TryParseStr("String", s));
    EXPECT_FALSE(parser.TryParseBool("Boolean", b));
    EXPECT_FALSE(parser.TryParseIntVec("IntVec", iv));
    EXPECT_FALSE(parser.TryParseSizeTVec("SizeTVec", sv));
    EXPECT_FALSE(parser.TryParseDoubleVec("DoubleVec", dv));
  }
}

int main(int argc, char *argv[]) {
  testing::InitGoogleTest(&argc, argv);
  // Expect two paths passed in
  json_ok_file = argv[1];
  json_wrong_types_file = argv[2];
  return RUN_ALL_TESTS();
}


