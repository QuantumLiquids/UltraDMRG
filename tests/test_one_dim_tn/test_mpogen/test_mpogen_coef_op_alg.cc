// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2019-10-29 15:34
* 
* Description: QuantumLiquids/UltraDMRG project. Unittests for algebra of MPO's coefficient and operator.
*/

#include <vector>
#include "gtest/gtest.h"
#include "qlmps/one_dim_tn/mpo/mpogen/symb_alg/coef_op_alg.h"

const BasisOpLabel kIdOpLabel = 0;                 // Coefficient label for identity id.
const OpRepr kIdOpRepr = OpRepr(kIdOpLabel);  // Operator representation for identity operator.

// Helpers.
const size_t kMaxTermNum = 5;

std::vector<long> RandVec(size_t size) {
  std::vector<long> rand_vec;
  for (size_t i = 0; i < size; ++i) {
    rand_vec.push_back(rand());
  }
  return rand_vec;
}

std::vector<long> InverseVec(const std::vector<long> &v) {
  std::vector<long> inv_v = v;
  std::reverse(inv_v.begin(), inv_v.end());
  return inv_v;
}

std::vector<CoefRepr> GenCoefReprVec(
    const std::vector<CNumberLabel> &coef_labels) {
  std::vector<CoefRepr> coef_reprs;
  for (auto label : coef_labels) { coef_reprs.push_back(CoefRepr(label)); }
  return coef_reprs;
}

CoefRepr RandCoefRepr(void) {
  size_t term_num = rand() % kMaxTermNum + 1;
  std::vector<CNumberLabel> coef_labels;
  for (size_t i = 0; i < term_num; ++i) {
    coef_labels.push_back(rand());
  }
  return CoefRepr(coef_labels);
}

void RandSetSparCoefReprMatElem(SparCoefReprMat &coef_repr_mat) {
  auto x = rand() % coef_repr_mat.rows;
  auto y = rand() % coef_repr_mat.cols;
  coef_repr_mat.SetElem(x, y, RandCoefRepr());
}

void RandFillSparCoefReprMat(
    SparCoefReprMat &coef_repr_mat, const size_t filling) {
  auto rows = coef_repr_mat.rows;
  auto cols = coef_repr_mat.cols;
  size_t nonull_elem_num = (rows * cols) / filling;
  if (nonull_elem_num == 0) { nonull_elem_num = 1; }
  for (size_t i = 0; i < nonull_elem_num; ++i) {
    RandSetSparCoefReprMatElem(coef_repr_mat);
  }
}

OpRepr RandOpRepr(void) {
  size_t term_num = rand() % kMaxTermNum + 1;
  std::vector<CoefRepr> coef_reprs;
  std::vector<BasisOpLabel> op_labels;
  for (size_t i = 0; i < term_num; ++i) {
    coef_reprs.push_back(RandCoefRepr());
    op_labels.push_back(rand());
  }
  return OpRepr(coef_reprs, op_labels);
}

void RandSetSparOpReprMatElem(SparOpReprMat &op_repr_mat) {
  auto x = rand() % op_repr_mat.rows;
  auto y = rand() % op_repr_mat.cols;
  op_repr_mat.SetElem(x, y, RandOpRepr());
}

void RandFillSparOpReprMat(SparOpReprMat &op_repr_mat, const size_t filling) {
  auto rows = op_repr_mat.rows;
  auto cols = op_repr_mat.cols;
  size_t nonull_elem_num = (rows * cols) / filling;
  if (nonull_elem_num == 0) { nonull_elem_num = 1; }
  for (size_t i = 0; i < nonull_elem_num; ++i) {
    RandSetSparOpReprMatElem(op_repr_mat);
  }
}


// Testing representation of coefficient.
TEST(TestCoefRepr, Initialization) {
  CoefRepr null_coef_repr;
  EXPECT_EQ(null_coef_repr.GetCoefLabelList(), std::vector<CNumberLabel>());

  CoefRepr id_coef_repr(kIdCoefLabel);
  std::vector<CNumberLabel> id_coef_label_list = {kIdCoefLabel};
  EXPECT_EQ(id_coef_repr.GetCoefLabelList(), id_coef_label_list);

  auto rand_coef_labels = RandVec(5);
  CoefRepr rand_coef_repr(rand_coef_labels);
  EXPECT_EQ(rand_coef_repr.GetCoefLabelList(), rand_coef_labels);
}

void RunTestCoefReprEquivalentCase(size_t size) {
  auto rand_coef_labels1 = RandVec(size);
  auto rand_coef_labels1_inv = InverseVec(rand_coef_labels1);
  CoefRepr coef_repr1a(rand_coef_labels1);
  CoefRepr coef_repr1b(rand_coef_labels1_inv);
  EXPECT_EQ(coef_repr1a, coef_repr1b);
  if (size != 0) {
    auto rand_coef_labels2 = RandVec(size);
    CoefRepr coef_repr2(rand_coef_labels2);
    EXPECT_NE(coef_repr1a, coef_repr2);
  }
}

TEST(TestCoefRepr, Equivalent) {
  RunTestCoefReprEquivalentCase(0);
  RunTestCoefReprEquivalentCase(1);
  RunTestCoefReprEquivalentCase(3);
  RunTestCoefReprEquivalentCase(5);
}

void RunTestCoefReprAddCase(size_t lhs_size, size_t rhs_size) {
  auto lhs_rand_coef_labels = RandVec(lhs_size);
  auto rhs_rand_coef_labels = RandVec(rhs_size);
  std::vector<CNumberLabel> added_rand_coef_labels;
  added_rand_coef_labels.reserve(lhs_size + rhs_size);
  added_rand_coef_labels.insert(
      added_rand_coef_labels.end(),
      lhs_rand_coef_labels.begin(), lhs_rand_coef_labels.end());
  added_rand_coef_labels.insert(
      added_rand_coef_labels.end(),
      rhs_rand_coef_labels.begin(), rhs_rand_coef_labels.end());
  CoefRepr lhs_coef_repr(lhs_rand_coef_labels);
  CoefRepr rhs_coef_repr(rhs_rand_coef_labels);
  CoefRepr added_coef_repr(added_rand_coef_labels);
  EXPECT_EQ(lhs_coef_repr + rhs_coef_repr, added_coef_repr);
  EXPECT_EQ(rhs_coef_repr + lhs_coef_repr, added_coef_repr);
}

TEST(TestCoefRepr, Add) {
  RunTestCoefReprAddCase(0, 0);
  RunTestCoefReprAddCase(1, 0);
  RunTestCoefReprAddCase(0, 1);
  RunTestCoefReprAddCase(1, 1);
  RunTestCoefReprAddCase(2, 1);
  RunTestCoefReprAddCase(1, 2);
  RunTestCoefReprAddCase(3, 3);
  RunTestCoefReprAddCase(5, 3);
  RunTestCoefReprAddCase(3, 5);
  RunTestCoefReprAddCase(5, 5);
}


// Testing representation of operator.
TEST(TestOpRepr, Initialization) {
  OpRepr null_op_repr;
  EXPECT_EQ(null_op_repr.GetCoefReprList(), std::vector<CoefRepr>());
  EXPECT_EQ(null_op_repr.GetOpLabelList(), std::vector<BasisOpLabel>());

  BasisOpLabel rand_op_label = rand();
  OpRepr nocoef_op_repr(rand_op_label);
  std::vector<CoefRepr> nocoef_op_coef_repr_list = {kIdCoefRepr};
  std::vector<BasisOpLabel> nocoef_op_op_label_list = {rand_op_label};
  EXPECT_EQ(nocoef_op_repr.GetCoefReprList(), nocoef_op_coef_repr_list);
  EXPECT_EQ(nocoef_op_repr.GetOpLabelList(), nocoef_op_op_label_list);

  CoefRepr rand_coef_repr(rand());
  std::vector<CoefRepr> op_coef_repr_list = {rand_coef_repr};
  auto coef_op_op_label_list = nocoef_op_op_label_list;
  OpRepr coef_op_repr(rand_coef_repr, rand_op_label);
  EXPECT_EQ(coef_op_repr.GetCoefReprList(), op_coef_repr_list);
  EXPECT_EQ(coef_op_repr.GetOpLabelList(), coef_op_op_label_list);

  size_t size = 5;
  std::vector<CoefRepr> rand_coef_reprs;
  std::vector<BasisOpLabel> rand_op_labels;
  for (size_t i = 0; i < size; ++i) {
    rand_coef_reprs.push_back(CoefRepr(rand()));
    rand_op_labels.push_back(rand());
  }
  OpRepr op_repr(rand_coef_reprs, rand_op_labels);
  auto geted_coef_repr_list = op_repr.GetCoefReprList();
  EXPECT_TRUE(std::is_permutation(geted_coef_repr_list.begin(), geted_coef_repr_list.end(),
                                  rand_coef_reprs.begin(), rand_coef_reprs.end()));
  // EXPECT_EQ(op_repr.GetCoefReprList(), rand_coef_reprs);
  std::sort(rand_op_labels.begin(), rand_op_labels.end());
  EXPECT_EQ(op_repr.GetOpLabelList(), rand_op_labels);

  auto coef1 = RandCoefRepr();
  auto coef2 = RandCoefRepr();
  BasisOpLabel op_label1 = rand() + 1;
  OpRepr op1({coef1, coef2}, {op_label1, op_label1});
  EXPECT_EQ(op1.GetCoefReprList(), CoefReprVec({coef1 + coef2}));
  EXPECT_EQ(op1.GetOpLabelList(), std::vector<BasisOpLabel>({op_label1}));

  BasisOpLabel op_label2 = rand() + 1;
  OpRepr op2(std::vector<BasisOpLabel>({op_label1, op_label1, op_label2}));
  EXPECT_EQ(
      op2.GetCoefReprList(),
      CoefReprVec({kIdCoefRepr + kIdCoefRepr, kIdCoefRepr}));
  EXPECT_EQ(op2.GetOpLabelList(), std::vector<BasisOpLabel>({op_label1, op_label2}));
}

void RunTestOpReprEquivalentCase(size_t size) {
  auto rand_vec1a = RandVec(size);
  auto rand_vec1b = RandVec(size);
  auto rand_vec1a_inv = InverseVec(rand_vec1a);
  auto rand_vec1b_inv = InverseVec(rand_vec1b);
  std::vector<CoefRepr> coef_list1 = GenCoefReprVec(rand_vec1a);
  std::vector<CoefRepr> coef_list1_inv = GenCoefReprVec(rand_vec1a_inv);
  OpRepr op_repr1a(coef_list1, rand_vec1b);
  OpRepr op_repr1b(coef_list1_inv, rand_vec1b_inv);
  EXPECT_EQ(op_repr1a, op_repr1a);
  EXPECT_EQ(op_repr1a, op_repr1b);
  if (size != 0) {
    auto rand_vec2a = RandVec(size);
    auto rand_vec2b = RandVec(size);
    std::vector<CoefRepr> coef_list2 = GenCoefReprVec(rand_vec2a);
    OpRepr op_repr2(coef_list2, rand_vec2b);
    EXPECT_NE(op_repr2, op_repr1a);
  }
}

TEST(TestOpRepr, TestOpReprEquivalent) {
  RunTestOpReprEquivalentCase(0);
  RunTestOpReprEquivalentCase(1);
  RunTestOpReprEquivalentCase(3);
  RunTestOpReprEquivalentCase(5);
}

void RunTestOpReprAddCase1(size_t lhs_size, size_t rhs_size) {
  auto lhs_rand_coef_reprs = GenCoefReprVec(RandVec(lhs_size));
  auto rhs_rand_coef_reprs = GenCoefReprVec(RandVec(rhs_size));
  auto lhs_rand_op_labels = RandVec(lhs_size);
  auto rhs_rand_op_labels = RandVec(rhs_size);
  std::vector<CoefRepr> added_rand_coef_reprs;
  added_rand_coef_reprs.reserve(lhs_size + rhs_size);
  added_rand_coef_reprs.insert(
      added_rand_coef_reprs.end(),
      lhs_rand_coef_reprs.begin(), lhs_rand_coef_reprs.end());
  added_rand_coef_reprs.insert(
      added_rand_coef_reprs.end(),
      rhs_rand_coef_reprs.begin(), rhs_rand_coef_reprs.end());
  std::vector<BasisOpLabel> added_rand_op_labels;
  added_rand_op_labels.reserve(lhs_size + rhs_size);
  added_rand_op_labels.insert(
      added_rand_op_labels.end(),
      lhs_rand_op_labels.begin(), lhs_rand_op_labels.end());
  added_rand_op_labels.insert(
      added_rand_op_labels.end(),
      rhs_rand_op_labels.begin(), rhs_rand_op_labels.end());
  OpRepr lhs_op_repr(lhs_rand_coef_reprs, lhs_rand_op_labels);
  OpRepr rhs_op_repr(rhs_rand_coef_reprs, rhs_rand_op_labels);
  OpRepr added_op_repr(added_rand_coef_reprs, added_rand_op_labels);
  EXPECT_EQ(lhs_op_repr + rhs_op_repr, added_op_repr);
  EXPECT_EQ(rhs_op_repr + lhs_op_repr, added_op_repr);
}

void RunTestOpReprAddCase2(void) {
  auto coef1 = RandCoefRepr();
  auto coef2 = RandCoefRepr();
  auto op_label1 = rand() + 1;
  OpRepr op1(coef1, op_label1);
  OpRepr op2(coef2, op_label1);
  OpRepr op3(coef1 + coef2, op_label1);
  EXPECT_EQ(op1 + op2, op3);
  auto coef3 = RandCoefRepr();
  auto op_label2 = rand() + 1;
  OpRepr op4({coef2, coef3}, {op_label1, op_label2});
  OpRepr op5({(coef1 + coef2), coef3}, {op_label1, op_label2});
  EXPECT_EQ(op4 + op1, op5);
}

TEST(TestOpRepr, TestOpReprAdd) {
  RunTestOpReprAddCase1(0, 0);
  RunTestOpReprAddCase1(1, 0);
  RunTestOpReprAddCase1(0, 1);
  RunTestOpReprAddCase1(1, 1);
  RunTestOpReprAddCase1(2, 1);
  RunTestOpReprAddCase1(1, 2);
  RunTestOpReprAddCase1(3, 3);
  RunTestOpReprAddCase1(5, 3);
  RunTestOpReprAddCase1(3, 5);
  RunTestOpReprAddCase1(5, 5);

  RunTestOpReprAddCase2();
}

void RunTestSparCoefReprMatInitializationCase(size_t row_num, size_t col_num) {
  SparCoefReprMat spar_mat(row_num, col_num);
  EXPECT_EQ(spar_mat.rows, row_num);
  EXPECT_EQ(spar_mat.cols, col_num);
  EXPECT_TRUE(spar_mat.data.empty());
  auto size = row_num * col_num;
  auto indexes = spar_mat.indexes;
  EXPECT_EQ(indexes.size(), size);
  for (size_t i = 0; i < size; ++i) { EXPECT_EQ(indexes[i], -1); }
}

TEST(TestSparCoefReprMat, Initialization) {
  SparCoefReprMat null_coef_repr_mat;
  EXPECT_EQ(null_coef_repr_mat.rows, 0);
  EXPECT_EQ(null_coef_repr_mat.cols, 0);
  EXPECT_TRUE(null_coef_repr_mat.data.empty());
  EXPECT_TRUE(null_coef_repr_mat.indexes.empty());

  RunTestSparCoefReprMatInitializationCase(1, 1);
  RunTestSparCoefReprMatInitializationCase(5, 1);
  RunTestSparCoefReprMatInitializationCase(1, 5);
  RunTestSparCoefReprMatInitializationCase(5, 3);
  RunTestSparCoefReprMatInitializationCase(3, 5);
  RunTestSparCoefReprMatInitializationCase(5, 5);
}

void RunTestSparCoefReprMatElemGetterAndSetterCase(
    size_t row_num, size_t col_num) {
  auto size = row_num * col_num;
  SparCoefReprMat spar_mat;
  if (size == 0) {
    spar_mat = SparCoefReprMat();
  } else {
    spar_mat = SparCoefReprMat(row_num, col_num);
  }
  auto null_coef_repr = CoefRepr();
  for (size_t x = 0; x < row_num; ++x) {
    for (size_t y = 0; y < col_num; ++y) {
      EXPECT_EQ(spar_mat(x, y), null_coef_repr);
    }
  }
  if (size > 0) {
    auto x1 = rand() % row_num;
    auto y1 = rand() % col_num;
    auto coef1 = RandCoefRepr();
    spar_mat.SetElem(x1, y1, coef1);
    EXPECT_EQ(spar_mat(x1, y1), coef1);
    auto coef2 = RandCoefRepr();
    spar_mat.SetElem(x1, y1, coef2);
    EXPECT_EQ(spar_mat(x1, y1), coef2);
    if (size > 1) {
      auto x2 = rand() % row_num;
      auto y2 = rand() % col_num;
      auto coef3 = RandCoefRepr();
      spar_mat.SetElem(x2, y2, coef3);
      EXPECT_EQ(spar_mat(x2, y2), coef3);
      auto coef4 = RandCoefRepr();
      spar_mat.SetElem(x2, y2, coef4);
      EXPECT_EQ(spar_mat(x2, y2), coef4);
    }
  }
}

TEST(TestSparCoefReprMat, ElemGetterAndSetter) {
  RunTestSparCoefReprMatElemGetterAndSetterCase(0, 0);
  RunTestSparCoefReprMatElemGetterAndSetterCase(1, 1);
  RunTestSparCoefReprMatElemGetterAndSetterCase(5, 1);
  RunTestSparCoefReprMatElemGetterAndSetterCase(1, 5);
  RunTestSparCoefReprMatElemGetterAndSetterCase(5, 3);
  RunTestSparCoefReprMatElemGetterAndSetterCase(3, 5);
  RunTestSparCoefReprMatElemGetterAndSetterCase(5, 5);
  RunTestSparCoefReprMatElemGetterAndSetterCase(20, 20);
}

void RunTestSparCoefReprMatRowAndColGetter(
    const size_t row_num, const size_t col_num) {
  auto size = row_num * col_num;
  SparCoefReprMat spar_mat(row_num, col_num);
  auto null_coef_repr = CoefRepr();
  std::vector<CoefRepr> null_row(col_num, null_coef_repr);
  std::vector<CoefRepr> null_col(row_num, null_coef_repr);
  for (size_t row_idx = 0; row_idx < row_num; ++row_idx) {
    EXPECT_EQ(spar_mat.GetRow(row_idx), null_row);
  }
  for (size_t col_idx = 0; col_idx < col_num; ++col_idx) {
    EXPECT_EQ(spar_mat.GetCol(col_idx), null_col);
  }
  if (size > 0) {
    auto x = rand() % row_num;
    auto y = rand() % col_num;
    CoefRepr coef(rand());
    spar_mat.SetElem(x, y, coef);
    auto x_row = spar_mat.GetRow(x);
    for (size_t i = 0; i < col_num; ++i) {
      if (i == y) {
        EXPECT_EQ(x_row[i], coef);
      } else {
        EXPECT_EQ(x_row[i], null_coef_repr);
      }
    }
    auto y_col = spar_mat.GetCol(y);
    for (size_t i = 0; i < row_num; ++i) {
      if (i == x) {
        EXPECT_EQ(y_col[i], coef);
      } else {
        EXPECT_EQ(y_col[i], null_coef_repr);
      }
    }
  }
}

TEST(TestSparCoefReprMat, RowAndColGetter) {
  RunTestSparCoefReprMatRowAndColGetter(0, 0);
  RunTestSparCoefReprMatRowAndColGetter(1, 1);
  RunTestSparCoefReprMatRowAndColGetter(5, 1);
  RunTestSparCoefReprMatRowAndColGetter(1, 5);
  RunTestSparCoefReprMatRowAndColGetter(3, 5);
  RunTestSparCoefReprMatRowAndColGetter(5, 3);
  RunTestSparCoefReprMatRowAndColGetter(5, 5);
}

void RunTestSparCoefReprMatRemoveRowAndColCase(
    const size_t row_num, const size_t col_num) {
  SparCoefReprMat spar_mat(row_num, col_num);
  auto x = rand() % row_num;
  auto y = rand() % col_num;
  spar_mat.SetElem(x, y, RandCoefRepr());

  auto spar_mat_to_rmv_row = spar_mat;
  spar_mat_to_rmv_row.RemoveRow(x);
  if (row_num > 1) {
    auto new_rows = row_num - 1;
    EXPECT_EQ(spar_mat_to_rmv_row.rows, new_rows);
    EXPECT_EQ(spar_mat_to_rmv_row.cols, col_num);
    auto new_size = new_rows * col_num;
    EXPECT_EQ(spar_mat_to_rmv_row.indexes, std::vector<long>(new_size, -1));
  } else {
    EXPECT_EQ(spar_mat_to_rmv_row.rows, 0);
    EXPECT_EQ(spar_mat_to_rmv_row.cols, 0);
    EXPECT_TRUE(spar_mat_to_rmv_row.indexes.empty());
  }

  auto spar_mat_to_rmv_col = spar_mat;
  spar_mat_to_rmv_col.RemoveCol(y);
  if (col_num > 1) {
    auto new_cols = col_num - 1;
    EXPECT_EQ(spar_mat_to_rmv_col.rows, row_num);
    EXPECT_EQ(spar_mat_to_rmv_col.cols, new_cols);
    auto new_size = row_num * new_cols;
    EXPECT_EQ(spar_mat_to_rmv_col.indexes, std::vector<long>(new_size, -1));
  } else {
    EXPECT_EQ(spar_mat_to_rmv_col.rows, 0);
    EXPECT_EQ(spar_mat_to_rmv_col.cols, 0);
    EXPECT_TRUE(spar_mat_to_rmv_col.indexes.empty());
  }
}

TEST(TestSparCoefReprMat, RemoveRowAndCol) {
  RunTestSparCoefReprMatRemoveRowAndColCase(1, 1);
  RunTestSparCoefReprMatRemoveRowAndColCase(1, 5);
  RunTestSparCoefReprMatRemoveRowAndColCase(5, 1);
  RunTestSparCoefReprMatRemoveRowAndColCase(3, 5);
  RunTestSparCoefReprMatRemoveRowAndColCase(5, 3);
  RunTestSparCoefReprMatRemoveRowAndColCase(5, 5);
  RunTestSparCoefReprMatRemoveRowAndColCase(100, 100);
}

void RunTestSparCoefReprMatSwapTwoRowsAndColsCase(
    const size_t row_num, const size_t col_num) {
  SparCoefReprMat spar_mat(row_num, col_num);
  auto x = rand() % row_num;
  auto y = rand() % col_num;
  spar_mat.SetElem(x, y, RandCoefRepr());

  auto row_idx2 = rand() % row_num;
  auto row1 = spar_mat.GetRow(x);
  auto row2 = spar_mat.GetRow(row_idx2);
  auto spar_mat_to_swap_rows = spar_mat;
  spar_mat_to_swap_rows.SwapTwoRows(x, row_idx2);
  EXPECT_EQ(spar_mat_to_swap_rows.GetRow(x), row2);
  EXPECT_EQ(spar_mat_to_swap_rows.GetRow(row_idx2), row1);

  auto col_idx2 = rand() % col_num;
  auto col1 = spar_mat.GetCol(y);
  auto col2 = spar_mat.GetCol(col_idx2);
  auto spar_mat_to_swap_cols = spar_mat;
  spar_mat_to_swap_cols.SwapTwoCols(y, col_idx2);
  EXPECT_EQ(spar_mat_to_swap_cols.GetCol(y), col2);
  EXPECT_EQ(spar_mat_to_swap_cols.GetCol(col_idx2), col1);
}

TEST(TestSparCoefReprMat, SwapTwoRowsAndCols) {
  RunTestSparCoefReprMatSwapTwoRowsAndColsCase(1, 1);
  RunTestSparCoefReprMatSwapTwoRowsAndColsCase(1, 5);
  RunTestSparCoefReprMatSwapTwoRowsAndColsCase(5, 1);
  RunTestSparCoefReprMatSwapTwoRowsAndColsCase(3, 5);
  RunTestSparCoefReprMatSwapTwoRowsAndColsCase(5, 3);
  RunTestSparCoefReprMatSwapTwoRowsAndColsCase(5, 5);
  RunTestSparCoefReprMatSwapTwoRowsAndColsCase(20, 20);
}

void RunTestSparCoefReprMatTransposeRowsAndCols(
    const std::vector<size_t> &transposed_row_idxs,
    const std::vector<size_t> &transposed_col_idxs) {
  auto row_num = transposed_row_idxs.size();
  auto col_num = transposed_col_idxs.size();
  SparCoefReprMat spar_mat(row_num, col_num);
  RandFillSparCoefReprMat(spar_mat, 3);

  auto spar_mat_to_tsps_rows = spar_mat;
  spar_mat_to_tsps_rows.TransposeRows(transposed_row_idxs);
  for (size_t i = 0; i < row_num; ++i) {
    EXPECT_EQ(
        spar_mat_to_tsps_rows.GetRow(i),
        spar_mat.GetRow(transposed_row_idxs[i]));
  }

  auto spar_mat_to_tsps_cols = spar_mat;
  spar_mat_to_tsps_cols.TransposeCols(transposed_col_idxs);
  for (size_t i = 0; i < col_num; ++i) {
    EXPECT_EQ(
        spar_mat_to_tsps_cols.GetCol(i),
        spar_mat.GetCol(transposed_col_idxs[i]));
  }
}

TEST(TestSparCoefReprMat, TransposeRowsAndCols) {
  RunTestSparCoefReprMatTransposeRowsAndCols({0, 1}, {0, 1});
  RunTestSparCoefReprMatTransposeRowsAndCols({1, 0}, {0, 1});
  RunTestSparCoefReprMatTransposeRowsAndCols({0, 1}, {1, 0});
  RunTestSparCoefReprMatTransposeRowsAndCols({1, 0}, {1, 0});
  RunTestSparCoefReprMatTransposeRowsAndCols({2, 1, 0}, {4, 3, 1, 0, 2});
  RunTestSparCoefReprMatTransposeRowsAndCols({4, 3, 1, 0, 2}, {1, 0, 2});
}

void RunTestSparOpReprMatInitializationCase(size_t row_num, size_t col_num) {
  SparOpReprMat spar_mat(row_num, col_num);
  EXPECT_EQ(spar_mat.rows, row_num);
  EXPECT_EQ(spar_mat.cols, col_num);
  EXPECT_TRUE(spar_mat.data.empty());
  auto size = row_num * col_num;
  auto indexes = spar_mat.indexes;
  EXPECT_EQ(indexes.size(), size);
  for (size_t i = 0; i < size; ++i) { EXPECT_EQ(indexes[i], -1); }
}

TEST(TestSparOpReprMat, Initialization) {
  SparOpReprMat null_op_repr_mat;
  EXPECT_EQ(null_op_repr_mat.rows, 0);
  EXPECT_EQ(null_op_repr_mat.cols, 0);
  EXPECT_TRUE(null_op_repr_mat.data.empty());
  EXPECT_TRUE(null_op_repr_mat.indexes.empty());

  RunTestSparOpReprMatInitializationCase(1, 1);
  RunTestSparOpReprMatInitializationCase(5, 1);
  RunTestSparOpReprMatInitializationCase(1, 5);
  RunTestSparOpReprMatInitializationCase(5, 3);
  RunTestSparOpReprMatInitializationCase(3, 5);
  RunTestSparOpReprMatInitializationCase(5, 5);
}

void RunTestSparOpReprMatSortRowsAndColsCase(size_t row_num, size_t col_num) {
  SparOpReprMat spar_mat(row_num, col_num);
  RandFillSparOpReprMat(spar_mat, 3);

  spar_mat.SortRows();
  std::vector<size_t> row_nonull_elem_nums(row_num, 0);
  for (size_t x = 0; x < row_num; ++x) {
    auto row = spar_mat.GetRow(x);
    for (auto &elem : row) {
      if (elem != kNullOpRepr) {
        row_nonull_elem_nums[x]++;
      }
    }
  }
  if (row_num != 1) {
    for (size_t x = 1; x < row_num; ++x) {
      EXPECT_TRUE(row_nonull_elem_nums[x - 1] <= row_nonull_elem_nums[x]);
    }
  }

  spar_mat.SortCols();
  std::vector<size_t> col_nonull_elem_nums(col_num, 0);
  for (size_t y = 0; y < col_num; ++y) {
    auto col = spar_mat.GetCol(y);
    for (auto &elem : col) {
      if (elem != kNullOpRepr) {
        col_nonull_elem_nums[y]++;
      }
    }
  }
  if (col_num != 1) {
    for (size_t y = 1; y < col_num; ++y) {
      EXPECT_TRUE(col_nonull_elem_nums[y - 1] <= col_nonull_elem_nums[y]);
    }
  }
}

TEST(TestSparOpReprMat, TestSparOpReprMatSortRowsAndCols) {
  RunTestSparOpReprMatSortRowsAndColsCase(1, 1);
  RunTestSparOpReprMatSortRowsAndColsCase(1, 5);
  RunTestSparOpReprMatSortRowsAndColsCase(5, 1);
  RunTestSparOpReprMatSortRowsAndColsCase(3, 5);
  RunTestSparOpReprMatSortRowsAndColsCase(5, 3);
  RunTestSparOpReprMatSortRowsAndColsCase(5, 5);
  RunTestSparOpReprMatSortRowsAndColsCase(20, 20);
}

TEST(TestSparOpReprMat, TestSparOpReprMatCalcRowAndColCoefs) {
  auto coef1 = RandCoefRepr();
  auto coef2 = RandCoefRepr();
  auto coef3 = RandCoefRepr();
  SparOpReprMat spar_mat(5, 5);
  spar_mat.SetElem(0, 4, OpRepr(1));
  spar_mat.SetElem(1, 1, OpRepr({coef1, coef1}, {2, 3}));
  spar_mat.SetElem(1, 3, OpRepr(coef2, 4));
  spar_mat.SetElem(3, 1, OpRepr(coef1, 5));
  spar_mat.SetElem(3, 3, OpRepr(coef1, 6));
  spar_mat.SetElem(4, 2, OpRepr(coef3, 7));

  EXPECT_EQ(spar_mat.TryCatchRowCommonDivisorCoef(0), kIdCoefRepr);
  EXPECT_EQ(spar_mat.TryCatchRowCommonDivisorCoef(1), kIdCoefRepr);
  EXPECT_EQ(spar_mat.TryCatchRowCommonDivisorCoef(2), kNullCoefRepr);
  EXPECT_EQ(spar_mat.TryCatchRowCommonDivisorCoef(3), coef1);
  EXPECT_EQ(spar_mat.TryCatchRowCommonDivisorCoef(4), coef3);
  EXPECT_EQ(spar_mat.TryCatchColCommonDivisorCoef(0), kNullCoefRepr);
  EXPECT_EQ(spar_mat.TryCatchColCommonDivisorCoef(1), coef1);
  EXPECT_EQ(spar_mat.TryCatchColCommonDivisorCoef(2), coef3);
  EXPECT_EQ(spar_mat.TryCatchColCommonDivisorCoef(3), kIdCoefRepr);
  EXPECT_EQ(spar_mat.TryCatchColCommonDivisorCoef(4), kIdCoefRepr);
}

void RunTestSparOpReprMatRowLinCmbCase1(void) {
  SparOpReprMat spar_mat(1, 1);
  auto cmb = spar_mat.CalcRowLinCmb(0);
  EXPECT_EQ(cmb, CoefReprVec({}));

  auto op_repr1 = RandOpRepr();
  spar_mat.SetElem(0, 0, op_repr1);
  cmb = spar_mat.CalcRowLinCmb(0);
  EXPECT_EQ(cmb, CoefReprVec({}));
}

void RunTestSparOpReprMatRowLinCmbCase2(void) {
  SparOpReprMat spar_mat(2, 2);
  auto cmb = spar_mat.CalcRowLinCmb(0);
  EXPECT_EQ(cmb, CoefReprVec({}));
  cmb = spar_mat.CalcRowLinCmb(1);
  EXPECT_EQ(cmb, CoefReprVec({kIdCoefRepr}));

  auto op_repr1 = OpRepr(1);
  spar_mat.SetElem(0, 0, op_repr1);
  cmb = spar_mat.CalcRowLinCmb(1);
  EXPECT_EQ(cmb, CoefReprVec({kNullCoefRepr}));
  spar_mat.SetElem(1, 0, op_repr1);
  cmb = spar_mat.CalcRowLinCmb(1);
  EXPECT_EQ(cmb, CoefReprVec({kIdCoefRepr}));

  auto op_repr2 = OpRepr(1, 1);
  spar_mat.SetElem(1, 0, op_repr2);
  cmb = spar_mat.CalcRowLinCmb(1);
  EXPECT_EQ(cmb, CoefReprVec({CoefRepr(1)}));

  auto op_repr3 = OpRepr(2);
  spar_mat.SetElem(0, 1, op_repr3);
  cmb = spar_mat.CalcRowLinCmb(1);
  EXPECT_EQ(cmb, CoefReprVec({kNullCoefRepr}));

  auto op_repr4 = OpRepr(1, 2);
  spar_mat.SetElem(1, 1, op_repr4);
  cmb = spar_mat.CalcRowLinCmb(1);
  EXPECT_EQ(cmb, CoefReprVec({CoefRepr(1)}));

  auto op_repr5 = OpRepr(2, 2);
  spar_mat.SetElem(1, 1, op_repr5);
  cmb = spar_mat.CalcRowLinCmb(1);
  EXPECT_EQ(cmb, CoefReprVec({kNullCoefRepr}));
}

void RunTestSparOpReprMatRowLinCmbCase3(void) {
  SparOpReprMat spar_mat(3, 2);
  auto cmb = spar_mat.CalcRowLinCmb(2);
  EXPECT_EQ(cmb, CoefReprVec({kIdCoefRepr, kNullCoefRepr}));

  auto op_repr1 = OpRepr(1);
  auto op_repr2 = OpRepr(2);
  spar_mat.SetElem(0, 0, op_repr1);
  spar_mat.SetElem(1, 1, op_repr2);
  cmb = spar_mat.CalcRowLinCmb(2);
  EXPECT_EQ(cmb, CoefReprVec({kNullCoefRepr, kNullCoefRepr}));

  spar_mat.SetElem(2, 0, op_repr1);
  cmb = spar_mat.CalcRowLinCmb(2);
  EXPECT_EQ(cmb, CoefReprVec({kIdCoefRepr, kNullCoefRepr}));

  spar_mat.SetElem(2, 1, op_repr2);
  cmb = spar_mat.CalcRowLinCmb(2);
  EXPECT_EQ(cmb, CoefReprVec({kIdCoefRepr, kIdCoefRepr}));

  spar_mat.SetElem(2, 0, OpRepr(1, 1));
  cmb = spar_mat.CalcRowLinCmb(2);
  EXPECT_EQ(cmb, CoefReprVec({CoefRepr(1), kIdCoefRepr}));

  spar_mat.SetElem(2, 1, OpRepr(2, 2));
  cmb = spar_mat.CalcRowLinCmb(2);
  EXPECT_EQ(cmb, CoefReprVec({CoefRepr(1), CoefRepr(2)}));
}

void RunTestSparOpReprMatColLinCmbCase1(void) {
  SparOpReprMat spar_mat(1, 1);
  auto cmb = spar_mat.CalcColLinCmb(0);
  EXPECT_EQ(cmb, CoefReprVec({}));

  auto op_repr1 = RandOpRepr();
  spar_mat.SetElem(0, 0, op_repr1);
  cmb = spar_mat.CalcColLinCmb(0);
  EXPECT_EQ(cmb, CoefReprVec({}));
}

void RunTestSparOpReprMatColLinCmbCase2(void) {
  SparOpReprMat spar_mat(2, 2);
  auto cmb = spar_mat.CalcColLinCmb(0);
  EXPECT_EQ(cmb, CoefReprVec({}));
  cmb = spar_mat.CalcColLinCmb(1);
  EXPECT_EQ(cmb, CoefReprVec({kNullCoefRepr}));

  auto op_repr1 = OpRepr(1);
  spar_mat.SetElem(0, 0, op_repr1);
  cmb = spar_mat.CalcColLinCmb(1);
  EXPECT_EQ(cmb, CoefReprVec({kNullCoefRepr}));
  spar_mat.SetElem(0, 1, op_repr1);
  cmb = spar_mat.CalcColLinCmb(1);
  EXPECT_EQ(cmb, CoefReprVec({kIdCoefRepr}));

  auto op_repr2 = OpRepr(1, 1);
  spar_mat.SetElem(0, 1, op_repr2);
  cmb = spar_mat.CalcColLinCmb(1);
  EXPECT_EQ(cmb, CoefReprVec({CoefRepr(1)}));

  auto op_repr3 = OpRepr(2);
  spar_mat.SetElem(1, 0, op_repr3);
  cmb = spar_mat.CalcColLinCmb(1);
  EXPECT_EQ(cmb, CoefReprVec({kNullCoefRepr}));

  auto op_repr4 = OpRepr(1, 2);
  spar_mat.SetElem(1, 1, op_repr4);
  cmb = spar_mat.CalcColLinCmb(1);
  EXPECT_EQ(cmb, CoefReprVec({CoefRepr(1)}));

  auto op_repr5 = OpRepr(2, 2);
  spar_mat.SetElem(1, 1, op_repr5);
  cmb = spar_mat.CalcColLinCmb(1);
  EXPECT_EQ(cmb, CoefReprVec({kNullCoefRepr}));
}

void RunTestSparOpReprMatColLinCmbCase3(void) {
  SparOpReprMat spar_mat(2, 3);
  auto cmb = spar_mat.CalcColLinCmb(2);
  EXPECT_EQ(cmb, CoefReprVec({kNullCoefRepr, kNullCoefRepr}));

  auto op_repr1 = OpRepr(1);
  auto op_repr2 = OpRepr(2);
  spar_mat.SetElem(0, 0, op_repr1);
  spar_mat.SetElem(1, 1, op_repr2);
  cmb = spar_mat.CalcColLinCmb(2);
  EXPECT_EQ(cmb, CoefReprVec({kNullCoefRepr, kNullCoefRepr}));

  spar_mat.SetElem(0, 2, op_repr1);
  cmb = spar_mat.CalcColLinCmb(2);
  EXPECT_EQ(cmb, CoefReprVec({kIdCoefRepr, kNullCoefRepr}));

  spar_mat.SetElem(1, 2, op_repr2);
  cmb = spar_mat.CalcColLinCmb(2);
  EXPECT_EQ(cmb, CoefReprVec({kIdCoefRepr, kIdCoefRepr}));

  spar_mat.SetElem(0, 2, OpRepr(1, 1));
  cmb = spar_mat.CalcColLinCmb(2);
  EXPECT_EQ(cmb, CoefReprVec({CoefRepr(1), kIdCoefRepr}));

  spar_mat.SetElem(1, 2, OpRepr(2, 2));
  cmb = spar_mat.CalcColLinCmb(2);
  EXPECT_EQ(cmb, CoefReprVec({CoefRepr(1), CoefRepr(2)}));
}

TEST(TestSparOpReprMat, TestSparOpReprMatRowAndColLinCmb) {
  RunTestSparOpReprMatRowLinCmbCase1();
  RunTestSparOpReprMatRowLinCmbCase2();
  RunTestSparOpReprMatRowLinCmbCase3();
  RunTestSparOpReprMatColLinCmbCase1();
  RunTestSparOpReprMatColLinCmbCase2();
  RunTestSparOpReprMatColLinCmbCase3();
}

void RunTestSparCoefReprMatSparOpReprMatIncompleteMultiCase1(void) {
  SparCoefReprMat coef_mat;
  SparOpReprMat op_mat;
  auto res = SparCoefReprMatSparOpReprMatIncompleteMulti(coef_mat, op_mat);
  EXPECT_EQ(res, SparOpReprMat());
}

void RunTestSparCoefReprMatSparOpReprMatIncompleteMultiCase2(void) {
  SparCoefReprMat coef_mat(1, 1);
  SparOpReprMat op_mat(1, 1);
  auto res = SparCoefReprMatSparOpReprMatIncompleteMulti(coef_mat, op_mat);
  EXPECT_EQ(res, SparOpReprMat(1, 1));

  CoefRepr coef1(1);
  OpRepr op1(1);
  OpRepr op2(1, 1);
  coef_mat.SetElem(0, 0, coef1);
  op_mat.SetElem(0, 0, op1);
  SparOpReprMat bchmk(1, 1);
  bchmk.SetElem(0, 0, op2);
  res = SparCoefReprMatSparOpReprMatIncompleteMulti(coef_mat, op_mat);
  EXPECT_EQ(res, bchmk);
}

void RunTestSparCoefReprMatSparOpReprMatIncompleteMultiCase3(void) {
  CoefRepr j(1);
  CoefRepr jp(2);
  BasisOpLabel s(1);

  SparCoefReprMat coef_mat1(2, 4);
  SparOpReprMat op_mat1(4, 6);
  SparOpReprMat bchmk1(2, 6);
  coef_mat1.SetElem(0, 0, kIdCoefRepr);
  coef_mat1.SetElem(1, 1, j);
  coef_mat1.SetElem(1, 2, j);
  coef_mat1.SetElem(1, 3, jp);
  op_mat1.SetElem(0, 0, kIdOpRepr);
  op_mat1.SetElem(2, 1, kIdOpRepr);
  op_mat1.SetElem(3, 2, kIdOpRepr);
  op_mat1.SetElem(0, 3, OpRepr(j, s));
  op_mat1.SetElem(0, 4, OpRepr(jp, s));
  op_mat1.SetElem(1, 5, OpRepr(s));
  bchmk1.SetElem(0, 0, kIdOpRepr);
  bchmk1.SetElem(1, 1, OpRepr(j, kIdOpLabel));
  bchmk1.SetElem(1, 2, OpRepr(jp, kIdOpLabel));
  bchmk1.SetElem(0, 3, OpRepr(j, s));
  bchmk1.SetElem(0, 4, OpRepr(jp, s));
  bchmk1.SetElem(1, 5, OpRepr(j, s));
  auto res1 = SparCoefReprMatSparOpReprMatIncompleteMulti(coef_mat1, op_mat1);
  EXPECT_EQ(res1, bchmk1);

  SparCoefReprMat coef_mat2(4, 6);
  SparOpReprMat op_mat2(6, 4);
  SparOpReprMat bchmk2(4, 4);
  coef_mat2.SetElem(0, 0, kIdCoefRepr);
  coef_mat2.SetElem(1, 1, j);
  coef_mat2.SetElem(1, 2, jp);
  coef_mat2.SetElem(2, 3, j);
  coef_mat2.SetElem(2, 4, jp);
  coef_mat2.SetElem(3, 5, j);
  op_mat2.SetElem(2, 0, kIdOpRepr);
  op_mat2.SetElem(3, 1, kIdOpRepr);
  op_mat2.SetElem(0, 2, OpRepr(j, s));
  op_mat2.SetElem(1, 3, OpRepr(s));
  op_mat2.SetElem(4, 3, OpRepr(s));
  op_mat2.SetElem(5, 3, kIdOpRepr);
  bchmk2.SetElem(1, 0, OpRepr(jp, kIdOpLabel));
  bchmk2.SetElem(2, 1, OpRepr(j, kIdOpLabel));
  bchmk2.SetElem(0, 2, OpRepr(j, s));
  bchmk2.SetElem(1, 3, OpRepr(j, s));
  bchmk2.SetElem(2, 3, OpRepr(jp, s));
  bchmk2.SetElem(3, 3, OpRepr(j, kIdOpLabel));
  auto res2 = SparCoefReprMatSparOpReprMatIncompleteMulti(coef_mat2, op_mat2);
  EXPECT_EQ(res2, bchmk2);
}

void RunTestSparCoefReprMatSparOpReprMatIncompleteMultiCase4(void) {
  CoefRepr j(1);
  CoefRepr k(2);
  BasisOpLabel sx(1);
  BasisOpLabel sy(2);
  BasisOpLabel sz(3);

  SparCoefReprMat coef_mat1(4, 5);
  SparOpReprMat op_mat1(5, 6);
  SparOpReprMat bchmk1(4, 6);
  coef_mat1.SetElem(0, 0, kIdCoefRepr);
  coef_mat1.SetElem(1, 1, j);
  coef_mat1.SetElem(2, 2, j);
  coef_mat1.SetElem(3, 3, j);
  coef_mat1.SetElem(1, 4, k);
  op_mat1.SetElem(0, 0, OpRepr(j, sx));
  op_mat1.SetElem(0, 1, OpRepr(j, sy));
  op_mat1.SetElem(0, 2, OpRepr(j, sz));
  op_mat1.SetElem(4, 3, OpRepr(sx));
  op_mat1.SetElem(0, 4, OpRepr(k, sz));
  op_mat1.SetElem(1, 5, OpRepr(sx));
  op_mat1.SetElem(2, 5, OpRepr(sy));
  op_mat1.SetElem(3, 5, OpRepr(sz));
  bchmk1.SetElem(0, 0, OpRepr(j, sx));
  bchmk1.SetElem(0, 1, OpRepr(j, sy));
  bchmk1.SetElem(0, 2, OpRepr(j, sz));
  bchmk1.SetElem(1, 3, OpRepr(k, sx));
  bchmk1.SetElem(0, 4, OpRepr(k, sz));
  bchmk1.SetElem(1, 5, OpRepr(j, sx));
  bchmk1.SetElem(2, 5, OpRepr(j, sy));
  bchmk1.SetElem(3, 5, OpRepr(j, sz));
  auto res1 = SparCoefReprMatSparOpReprMatIncompleteMulti(coef_mat1, op_mat1);
  EXPECT_EQ(res1, bchmk1);

  SparCoefReprMat coef_mat2(5, 6);
  SparOpReprMat op_mat2(6, 1);
  SparOpReprMat bchmk2(5, 1);
  coef_mat2.SetElem(0, 0, j);
  coef_mat2.SetElem(1, 1, j);
  coef_mat2.SetElem(2, 2, j);
  coef_mat2.SetElem(3, 3, k);
  coef_mat2.SetElem(2, 4, k);
  coef_mat2.SetElem(4, 5, j);
  op_mat2.SetElem(0, 0, OpRepr(sx));
  op_mat2.SetElem(1, 0, OpRepr(sy));
  op_mat2.SetElem(2, 0, OpRepr(sz));
  op_mat2.SetElem(3, 0, kIdOpRepr);
  op_mat2.SetElem(4, 0, OpRepr(sz));
  op_mat2.SetElem(5, 0, kIdOpRepr);
  bchmk2.SetElem(0, 0, OpRepr(j, sx));
  bchmk2.SetElem(1, 0, OpRepr(j, sy));
  bchmk2.SetElem(2, 0, OpRepr((j + k), sz));
  bchmk2.SetElem(3, 0, OpRepr(k, kIdOpLabel));
  bchmk2.SetElem(4, 0, OpRepr(j, kIdOpLabel));
  auto res2 = SparCoefReprMatSparOpReprMatIncompleteMulti(coef_mat2, op_mat2);
  EXPECT_EQ(res2, bchmk2);
}

TEST(TestSparOpReprMat, TestSparCoefReprMatSparOpReprMatIncompleteMulti) {
  RunTestSparCoefReprMatSparOpReprMatIncompleteMultiCase1();
  RunTestSparCoefReprMatSparOpReprMatIncompleteMultiCase2();
  RunTestSparCoefReprMatSparOpReprMatIncompleteMultiCase3();
  RunTestSparCoefReprMatSparOpReprMatIncompleteMultiCase4();
}

void RunTestSparOpReprMatSparCoefReprMatIncompleteMultiCase1(void) {
  SparCoefReprMat coef_mat;
  SparOpReprMat op_mat;
  auto res = SparOpReprMatSparCoefReprMatIncompleteMulti(op_mat, coef_mat);
  EXPECT_EQ(res, SparOpReprMat());
}

void RunTestSparOpReprMatSparCoefReprMatIncompleteMultiCase2(void) {
  SparCoefReprMat coef_mat(1, 1);
  SparOpReprMat op_mat(1, 1);
  auto res = SparOpReprMatSparCoefReprMatIncompleteMulti(op_mat, coef_mat);
  EXPECT_EQ(res, SparOpReprMat(1, 1));

  CoefRepr coef1(1);
  OpRepr op1(1);
  OpRepr op2(1, 1);
  coef_mat.SetElem(0, 0, coef1);
  op_mat.SetElem(0, 0, op1);
  SparOpReprMat bchmk(1, 1);
  bchmk.SetElem(0, 0, op2);
  res = SparOpReprMatSparCoefReprMatIncompleteMulti(op_mat, coef_mat);
  EXPECT_EQ(res, bchmk);
}

void RunTestSparOpReprMatSparCoefReprMatIncompleteMultiCase3(void) {
  CoefRepr j(1);
  CoefRepr jp(2);
  BasisOpLabel s(1);

  SparOpReprMat op_mat(4, 4);
  SparCoefReprMat coef_mat(4, 2);
  SparOpReprMat bchmk(4, 2);
  op_mat.SetElem(0, 2, OpRepr(s));
  op_mat.SetElem(1, 0, kIdOpRepr);
  op_mat.SetElem(1, 3, OpRepr(j, s));
  op_mat.SetElem(2, 1, kIdOpRepr);
  op_mat.SetElem(2, 3, OpRepr(jp, s));
  op_mat.SetElem(3, 3, OpRepr(j, kIdOpLabel));
  coef_mat.SetElem(0, 0, jp);
  coef_mat.SetElem(1, 0, j);
  coef_mat.SetElem(2, 0, j);
  coef_mat.SetElem(3, 1, kIdCoefRepr);
  bchmk.SetElem(0, 0, OpRepr(j, s));
  bchmk.SetElem(1, 0, OpRepr(jp, kIdOpLabel));
  bchmk.SetElem(1, 1, OpRepr(j, s));
  bchmk.SetElem(2, 0, OpRepr(j, kIdOpLabel));
  bchmk.SetElem(2, 1, OpRepr(jp, s));
  bchmk.SetElem(3, 1, OpRepr(j, kIdOpLabel));
  auto res = SparOpReprMatSparCoefReprMatIncompleteMulti(op_mat, coef_mat);
  EXPECT_EQ(res, bchmk);
}

void RunTestSparOpReprMatSparCoefReprMatIncompleteMultiCase4(void) {
  CoefRepr j(1);
  CoefRepr k(2);
  BasisOpLabel sx(1);
  BasisOpLabel sy(2);
  BasisOpLabel sz(3);

  SparOpReprMat op_mat(4, 5);
  SparCoefReprMat coef_mat(5, 4);
  SparOpReprMat bchmk(4, 4);
  op_mat.SetElem(0, 0, OpRepr(sx));
  op_mat.SetElem(0, 1, OpRepr(sy));
  op_mat.SetElem(0, 2, OpRepr(sz));
  op_mat.SetElem(1, 3, OpRepr(sx));
  op_mat.SetElem(1, 4, OpRepr(sx));
  op_mat.SetElem(2, 4, OpRepr(sy));
  op_mat.SetElem(3, 4, OpRepr(sz));
  coef_mat.SetElem(0, 0, j);
  coef_mat.SetElem(1, 1, j);
  coef_mat.SetElem(2, 2, j + k);
  coef_mat.SetElem(3, 3, k);
  coef_mat.SetElem(4, 3, j);
  bchmk.SetElem(0, 0, OpRepr(j, sx));
  bchmk.SetElem(0, 1, OpRepr(j, sy));
  bchmk.SetElem(0, 2, OpRepr(j + k, sz));
  bchmk.SetElem(1, 3, OpRepr(k + j, sx));
  bchmk.SetElem(2, 3, OpRepr(j, sy));
  bchmk.SetElem(3, 3, OpRepr(j, sz));
  auto res = SparOpReprMatSparCoefReprMatIncompleteMulti(op_mat, coef_mat);
  EXPECT_EQ(res, bchmk);
}

TEST(TestSparOpReprMat, TestSparOpReprMatSparCoefReprMatIncompleteMulti) {
  RunTestSparOpReprMatSparCoefReprMatIncompleteMultiCase1();
  RunTestSparOpReprMatSparCoefReprMatIncompleteMultiCase2();
  RunTestSparOpReprMatSparCoefReprMatIncompleteMultiCase3();
  RunTestSparOpReprMatSparCoefReprMatIncompleteMultiCase4();
}

void RunTestSparOpReprMatRowCompresserCase1(void) {
  SparOpReprMat m1(1, 1), m2(1, 1);
  SparOpReprMatRowCompresser(m1, m2);
  SparOpReprMat bchmk_m1(1, 1), bchmk_m2(1, 1);
  EXPECT_EQ(m1, bchmk_m1);
  EXPECT_EQ(m2, bchmk_m2);

  m1.SetElem(0, 0, OpRepr(1));
  m2.SetElem(0, 0, OpRepr(2));
  bchmk_m1 = m1;
  bchmk_m2 = m2;
  SparOpReprMatRowCompresser(m1, m2);
  EXPECT_EQ(m1, bchmk_m1);
  EXPECT_EQ(m2, bchmk_m2);
}

void RunTestSparOpReprMatRowCompresserCase2(void) {
  OpRepr a(1), b(2), c(3), d(4);
  SparOpReprMat m1(2, 1), m2(1, 2);
  m1.SetElem(0, 0, c);
  m1.SetElem(1, 0, d);
  m2.SetElem(0, 0, a);
  m2.SetElem(0, 1, b);
  auto bchmk_m1 = m1;
  auto bchmk_m2 = m2;
  SparOpReprMatRowCompresser(m1, m2);
  EXPECT_EQ(m1, bchmk_m1);
  EXPECT_EQ(m2, bchmk_m2);
}

void RunTestSparOpReprMatRowCompresserCase3(void) {
  OpRepr a(1), b(2);
  SparOpReprMat m1(2, 1);
  SparOpReprMat m2(1, 2);
  m1.SetElem(0, 0, kIdOpRepr);
  m1.SetElem(1, 0, kIdOpRepr);
  m2.SetElem(0, 0, a);
  m2.SetElem(0, 1, b);
  SparOpReprMat bchmk_m1(1, 1);
  SparOpReprMat bchmk_m2(1, 1);
  bchmk_m1.SetElem(0, 0, kIdOpRepr);
  bchmk_m2.SetElem(0, 0, a + b);
  SparOpReprMatRowCompresser(m1, m2);
  EXPECT_EQ(m1, bchmk_m1);
  EXPECT_EQ(m2, bchmk_m2);
}

void RunTestSparOpReprMatRowCompresserCase4(void) {
  OpRepr a(1), b(2), c(3);
  SparOpReprMat m1(2, 1);
  SparOpReprMat m2(1, 2);
  m1.SetElem(0, 0, c);
  m1.SetElem(1, 0, c);
  m2.SetElem(0, 0, a);
  m2.SetElem(0, 1, b);
  SparOpReprMat bchmk_m1(1, 1);
  SparOpReprMat bchmk_m2(1, 1);
  bchmk_m1.SetElem(0, 0, c);
  bchmk_m2.SetElem(0, 0, a + b);
  SparOpReprMatRowCompresser(m1, m2);
  EXPECT_EQ(m1, bchmk_m1);
  EXPECT_EQ(m2, bchmk_m2);
}

void RunTestSparOpReprMatRowCompresserCase5(void) {
  OpRepr a(1), b(2), c(3), d(4), e(5);
  SparOpReprMat m1(3, 1);
  SparOpReprMat m2(1, 3);
  m1.SetElem(0, 0, d);
  m1.SetElem(1, 0, d);
  m1.SetElem(2, 0, e);
  m2.SetElem(0, 0, a);
  m2.SetElem(0, 1, b);
  m2.SetElem(0, 2, c);
  SparOpReprMat bchmk_m1(2, 1);
  SparOpReprMat bchmk_m2(1, 2);
  bchmk_m1.SetElem(0, 0, d);
  bchmk_m1.SetElem(1, 0, e);
  bchmk_m2.SetElem(0, 0, a + b);
  bchmk_m2.SetElem(0, 1, c);
  SparOpReprMatRowCompresser(m1, m2);
  EXPECT_EQ(m1, bchmk_m1);
  EXPECT_EQ(m2, bchmk_m2);
}

void RunTestSparOpReprMatRowCompresserCase6(void) {
  OpRepr a(1), b(2), c(3), d(4), e(5);
  SparOpReprMat m1(3, 1);
  SparOpReprMat m2(1, 3);
  m1.SetElem(0, 0, d);
  m1.SetElem(1, 0, e);
  m1.SetElem(2, 0, e);
  m2.SetElem(0, 0, a);
  m2.SetElem(0, 1, b);
  m2.SetElem(0, 2, c);
  SparOpReprMat bchmk_m1(2, 1);
  SparOpReprMat bchmk_m2(1, 2);
  bchmk_m1.SetElem(0, 0, d);
  bchmk_m1.SetElem(1, 0, e);
  bchmk_m2.SetElem(0, 0, a);
  bchmk_m2.SetElem(0, 1, b + c);
  SparOpReprMatRowCompresser(m1, m2);
  EXPECT_EQ(m1, bchmk_m1);
  EXPECT_EQ(m2, bchmk_m2);
}

void RunTestSparOpReprMatRowCompresserCase7(void) {
  OpRepr a(1), b(2), c(3), d(4), e(5);
  SparOpReprMat m1(3, 1), m2(1, 3);
  m1.SetElem(0, 0, d);
  m1.SetElem(1, 0, e);
  m1.SetElem(2, 0, d);
  m2.SetElem(0, 0, a);
  m2.SetElem(0, 1, b);
  m2.SetElem(0, 2, c);
  SparOpReprMat bchmk_m1(2, 1), bchmk_m2(1, 2);
  bchmk_m1.SetElem(0, 0, d);
  bchmk_m1.SetElem(1, 0, e);
  bchmk_m2.SetElem(0, 0, a + c);
  bchmk_m2.SetElem(0, 1, b);
  SparOpReprMatRowCompresser(m1, m2);
  EXPECT_EQ(m1, bchmk_m1);
  EXPECT_EQ(m2, bchmk_m2);
}

void RunTestSparOpReprMatRowCompresserCase8(void) {
  OpRepr a(1), b(2), c(3), d(4), e(5);
  SparOpReprMat m1(3, 2), m2(1, 3);
  m1.SetElem(0, 0, d);
  m1.SetElem(1, 1, e);
  m1.SetElem(2, 0, d);
  m1.SetElem(2, 1, e);
  m2.SetElem(0, 0, a);
  m2.SetElem(0, 1, b);
  m2.SetElem(0, 2, c);
  SparOpReprMat bchmk_m1(2, 2), bchmk_m2(1, 2);
  bchmk_m1.SetElem(0, 0, d);
  bchmk_m1.SetElem(1, 1, e);
  bchmk_m2.SetElem(0, 0, a + c);
  bchmk_m2.SetElem(0, 1, b + c);
  SparOpReprMatRowCompresser(m1, m2);
  EXPECT_EQ(m1, bchmk_m1);
  EXPECT_EQ(m2, bchmk_m2);
}

void RunTestSparOpReprMatRowCompresserCase9(void) {
  OpRepr a(1), b(2), c(3), d(4), e(5);
  SparOpReprMat m1(3, 2), m2(1, 3);
  m1.SetElem(0, 0, d);
  m1.SetElem(0, 1, e);
  m1.SetElem(1, 0, d);
  m1.SetElem(2, 1, e);
  m2.SetElem(0, 0, a);
  m2.SetElem(0, 1, b);
  m2.SetElem(0, 2, c);
  SparOpReprMat bchmk_m1(2, 2), bchmk_m2(1, 2);
  bchmk_m1.SetElem(0, 0, d);
  bchmk_m1.SetElem(1, 1, e);
  bchmk_m2.SetElem(0, 0, a + b);
  bchmk_m2.SetElem(0, 1, a + c);
  SparOpReprMatRowCompresser(m1, m2);
  EXPECT_EQ(m1, bchmk_m1);
  EXPECT_EQ(m2, bchmk_m2);
}

void RunTestSparOpReprMatRowCompresserCase10(void) {
  OpRepr a(1), b(2), c(3), d(4), e(5), alpha_d(1, 4), beta_e(2, 5);
  SparOpReprMat m1(3, 2), m2(1, 3);
  m1.SetElem(0, 0, alpha_d);
  m1.SetElem(0, 1, beta_e);
  m1.SetElem(1, 0, d);
  m1.SetElem(2, 1, e);
  m2.SetElem(0, 0, a);
  m2.SetElem(0, 1, b);
  m2.SetElem(0, 2, c);
  SparOpReprMat bchmk_m1(2, 2), bchmk_m2(1, 2);
  OpRepr alpha_a(1, 1), beta_a(2, 1);
  bchmk_m1.SetElem(0, 0, d);
  bchmk_m1.SetElem(1, 1, e);
  bchmk_m2.SetElem(0, 0, alpha_a + b);
  bchmk_m2.SetElem(0, 1, beta_a + c);
  SparOpReprMatRowCompresser(m1, m2);
  EXPECT_EQ(m1, bchmk_m1);
  EXPECT_EQ(m2, bchmk_m2);
}

void RunTestSparOpReprMatRowCompresserCase11(void) {
  CoefRepr j(1), k(2);
  BasisOpLabel sx = 1, sy = 2, sz = 3;
  SparOpReprMat m1(5, 1), m2(4, 5);
  m1.SetElem(0, 0, OpRepr(j, sx));
  m1.SetElem(1, 0, OpRepr(j, sy));
  m1.SetElem(2, 0, OpRepr(j + k, sz));
  m1.SetElem(3, 0, OpRepr(k, kIdOpLabel));
  m1.SetElem(4, 0, OpRepr(j, kIdOpLabel));
  m2.SetElem(0, 0, OpRepr(sx));
  m2.SetElem(0, 1, OpRepr(sy));
  m2.SetElem(0, 2, OpRepr(sz));
  m2.SetElem(1, 3, OpRepr(sx));
  m2.SetElem(1, 4, OpRepr(sx));
  m2.SetElem(2, 4, OpRepr(sy));
  m2.SetElem(3, 4, OpRepr(sz));
  SparOpReprMat bchmk_m1(4, 1), bchmk_m2(4, 4);
  bchmk_m1.SetElem(0, 0, OpRepr(sx));
  bchmk_m1.SetElem(1, 0, OpRepr(sy));
  bchmk_m1.SetElem(2, 0, OpRepr(sz));
  bchmk_m1.SetElem(3, 0, kIdOpRepr);
  bchmk_m2.SetElem(0, 0, OpRepr(j, sx));
  bchmk_m2.SetElem(0, 1, OpRepr(j, sy));
  bchmk_m2.SetElem(0, 2, OpRepr(j + k, sz));
  bchmk_m2.SetElem(1, 3, OpRepr(j + k, sx));
  bchmk_m2.SetElem(2, 3, OpRepr(j, sy));
  bchmk_m2.SetElem(3, 3, OpRepr(j, sz));
  SparOpReprMatRowCompresser(m1, m2);
  EXPECT_EQ(m1, bchmk_m1);
  EXPECT_EQ(m2, bchmk_m2);
}

void RunTestSparOpReprMatRowCompresserCase12(void) {
  OpRepr s(1);
  SparOpReprMat m1(2, 2), m2(2, 2);
  m1.SetElem(0, 1, kIdOpRepr);
  m1.SetElem(1, 0, s);
  m1.SetElem(1, 1, kIdOpRepr);
  m2.SetElem(0, 0, kIdOpRepr);
  m2.SetElem(1, 1, s);
  auto bchmk_m1 = m1;
  auto bchmk_m2 = m2;
  SparOpReprMatRowCompresser(m1, m2);
  EXPECT_EQ(m1, bchmk_m1);
  EXPECT_EQ(m2, bchmk_m2);
}

TEST(TestSparOpReprMat, TestSparOpReprMatRowCompresser) {
  RunTestSparOpReprMatRowCompresserCase1();
  RunTestSparOpReprMatRowCompresserCase2();
  RunTestSparOpReprMatRowCompresserCase3();
  RunTestSparOpReprMatRowCompresserCase4();
  RunTestSparOpReprMatRowCompresserCase5();
  RunTestSparOpReprMatRowCompresserCase6();
  RunTestSparOpReprMatRowCompresserCase7();
  RunTestSparOpReprMatRowCompresserCase8();
  RunTestSparOpReprMatRowCompresserCase9();
  RunTestSparOpReprMatRowCompresserCase10();
  RunTestSparOpReprMatRowCompresserCase11();
  RunTestSparOpReprMatRowCompresserCase12();
}

void RunTestSparOpReprMatColCompresserCase1(void) {
  SparOpReprMat m1(1, 1), m2(1, 1);
  SparOpReprMatColCompresser(m1, m2);
  SparOpReprMat bchmk_m1(1, 1), bchmk_m2(1, 1);
  EXPECT_EQ(m1, bchmk_m1);
  EXPECT_EQ(m2, bchmk_m2);

  m1.SetElem(0, 0, OpRepr(1));
  m2.SetElem(0, 0, kIdOpRepr);
  bchmk_m1 = m1;
  bchmk_m2 = m2;
  SparOpReprMatColCompresser(m1, m2);
  EXPECT_EQ(m1, bchmk_m1);
  EXPECT_EQ(m2, bchmk_m2);
}

void RunTestSparOpReprMatColCompresserCase2(void) {
  OpRepr a(1), b(2), c(3), d(4);
  SparOpReprMat m1(1, 2), m2(2, 1);
  m1.SetElem(0, 0, a);
  m1.SetElem(0, 1, b);
  m2.SetElem(0, 0, c);
  m2.SetElem(1, 0, d);
  auto bchmk_m1 = m1;
  auto bchmk_m2 = m2;
  SparOpReprMatColCompresser(m1, m2);
  EXPECT_EQ(m1, bchmk_m1);
  EXPECT_EQ(m2, bchmk_m2);
}

void RunTestSparOpReprMatColCompresserCase3(void) {
  OpRepr a(1), b(2);
  SparOpReprMat m1(1, 2), m2(2, 1), bchmk_m1(1, 1), bchmk_m2(1, 1);
  m1.SetElem(0, 0, kIdOpRepr);
  m1.SetElem(0, 1, kIdOpRepr);
  m2.SetElem(0, 0, a);
  m2.SetElem(1, 0, b);
  bchmk_m1.SetElem(0, 0, kIdOpRepr);
  bchmk_m2.SetElem(0, 0, a + b);
  SparOpReprMatColCompresser(m1, m2);
  EXPECT_EQ(m1, bchmk_m1);
  EXPECT_EQ(m2, bchmk_m2);
}

void RunTestSparOpReprMatColCompresserCase4(void) {
  OpRepr a(1), b(2), c(3);
  SparOpReprMat m1(1, 2), m2(2, 1), bchmk_m1(1, 1), bchmk_m2(1, 1);
  m1.SetElem(0, 0, a);
  m1.SetElem(0, 1, a);
  m2.SetElem(0, 0, b);
  m2.SetElem(1, 0, c);
  bchmk_m1.SetElem(0, 0, a);
  bchmk_m2.SetElem(0, 0, b + c);
  SparOpReprMatColCompresser(m1, m2);
  EXPECT_EQ(m1, bchmk_m1);
  EXPECT_EQ(m2, bchmk_m2);
}

void RunTestSparOpReprMatColCompresserCase5(void) {
  OpRepr a(1), b(2), c(3), d(4), e(5);
  SparOpReprMat m1(1, 3), m2(3, 1), bchmk_m1(1, 2), bchmk_m2(2, 1);
  m1.SetElem(0, 0, a);
  m1.SetElem(0, 1, a);
  m1.SetElem(0, 2, b);
  m2.SetElem(0, 0, c);
  m2.SetElem(1, 0, d);
  m2.SetElem(2, 0, e);
  bchmk_m1.SetElem(0, 0, a);
  bchmk_m1.SetElem(0, 1, b);
  bchmk_m2.SetElem(0, 0, c + d);
  bchmk_m2.SetElem(1, 0, e);
  SparOpReprMatColCompresser(m1, m2);
  EXPECT_EQ(m1, bchmk_m1);
  EXPECT_EQ(m2, bchmk_m2);
}

void RunTestSparOpReprMatColCompresserCase6(void) {
  OpRepr a(1), b(2), c(3), d(4), e(5);
  SparOpReprMat m1(1, 3), m2(3, 1), bchmk_m1(1, 2), bchmk_m2(2, 1);
  m1.SetElem(0, 0, a);
  m1.SetElem(0, 1, b);
  m1.SetElem(0, 2, a);
  m2.SetElem(0, 0, c);
  m2.SetElem(1, 0, d);
  m2.SetElem(2, 0, e);
  bchmk_m1.SetElem(0, 0, a);
  bchmk_m1.SetElem(0, 1, b);
  bchmk_m2.SetElem(0, 0, c + e);
  bchmk_m2.SetElem(1, 0, d);
  SparOpReprMatColCompresser(m1, m2);
  EXPECT_EQ(m1, bchmk_m1);
  EXPECT_EQ(m2, bchmk_m2);
}

void RunTestSparOpReprMatColCompresserCase7(void) {
  CNumberLabel alp = 1, bet = 2;
  BasisOpLabel a = 1, b = 2, c = 3, d = 4, e = 5;
  SparOpReprMat m1(2, 3), m2(3, 1), bchmk_m1(2, 2), bchmk_m2(2, 1);
  m1.SetElem(0, 0, OpRepr(alp, a));
  m1.SetElem(0, 1, OpRepr(a));
  m1.SetElem(1, 0, OpRepr(bet, b));
  m1.SetElem(1, 2, OpRepr(b));
  m2.SetElem(0, 0, OpRepr(c));
  m2.SetElem(1, 0, OpRepr(d));
  m2.SetElem(2, 0, OpRepr(e));
  bchmk_m1.SetElem(0, 0, OpRepr(a));
  bchmk_m1.SetElem(1, 1, OpRepr(b));
  bchmk_m2.SetElem(0, 0, OpRepr(d) + OpRepr(alp, c));
  bchmk_m2.SetElem(1, 0, OpRepr(e) + OpRepr(bet, c));
  SparOpReprMatColCompresser(m1, m2);
  EXPECT_EQ(m1, bchmk_m1);
  EXPECT_EQ(m2, bchmk_m2);
}

void RunTestSparOpReprMatColCompresserCase8(void) {
  OpRepr s(1);
  SparOpReprMat m1(2, 6), m2(6, 4), bchmk_m1(2, 4), bchmk_m2(4, 4);
  m1.SetElem(0, 0, kIdOpRepr);
  m1.SetElem(1, 1, kIdOpRepr);
  m1.SetElem(0, 2, s);
  m1.SetElem(1, 3, kIdOpRepr);
  m1.SetElem(0, 4, s);
  m1.SetElem(1, 5, s);
  m2.SetElem(0, 2, s);
  m2.SetElem(1, 0, s);
  m2.SetElem(2, 1, kIdOpRepr);
  m2.SetElem(3, 3, kIdOpRepr);
  m2.SetElem(4, 0, s);
  m2.SetElem(5, 0, kIdOpRepr);
  bchmk_m1.SetElem(0, 0, kIdOpRepr);
  bchmk_m1.SetElem(0, 2, s);
  bchmk_m1.SetElem(1, 1, kIdOpRepr);
  bchmk_m1.SetElem(1, 3, s);
  bchmk_m2.SetElem(0, 2, s);
  bchmk_m2.SetElem(1, 0, s);
  bchmk_m2.SetElem(1, 3, kIdOpRepr);
  bchmk_m2.SetElem(2, 0, s);
  bchmk_m2.SetElem(2, 1, kIdOpRepr);
  bchmk_m2.SetElem(3, 0, kIdOpRepr);
  SparOpReprMatColCompresser(m1, m2);
  EXPECT_EQ(m1, bchmk_m1);
  EXPECT_EQ(m2, bchmk_m2);
}

void RunTestSparOpReprMatColCompresserCase9(void) {
  CNumberLabel j1 = 1, j2 = 2;
  BasisOpLabel s = 1;
  SparOpReprMat m1(1, 3), m2(3, 6), bchmk_m1(1, 2), bchmk_m2(2, 6);
  m1.SetElem(0, 0, kIdOpRepr);
  m1.SetElem(0, 1, OpRepr(j1, s));
  m1.SetElem(0, 2, OpRepr(j2, s));
  m2.SetElem(0, 0, kIdOpRepr);
  m2.SetElem(0, 2, OpRepr(j1, s));
  m2.SetElem(0, 4, OpRepr(j2, s));
  m2.SetElem(1, 1, kIdOpRepr);
  m2.SetElem(1, 5, OpRepr(s));
  m2.SetElem(2, 3, kIdOpRepr);
  bchmk_m1.SetElem(0, 0, kIdOpRepr);
  bchmk_m1.SetElem(0, 1, OpRepr(s));
  bchmk_m2.SetElem(0, 0, kIdOpRepr);
  bchmk_m2.SetElem(0, 2, OpRepr(j1, s));
  bchmk_m2.SetElem(0, 4, OpRepr(j2, s));
  bchmk_m2.SetElem(1, 1, OpRepr(j1, kIdOpLabel));
  bchmk_m2.SetElem(1, 3, OpRepr(j2, kIdOpLabel));
  bchmk_m2.SetElem(1, 5, OpRepr(j1, s));
  SparOpReprMatColCompresser(m1, m2);
  EXPECT_EQ(m1, bchmk_m1);
  EXPECT_EQ(m2, bchmk_m2);
}

void RunTestSparOpReprMatColCompresserCase10(void) {
  CNumberLabel j = 1, k = 2;
  BasisOpLabel sx = 1, sy = 2, sz = 3;
  SparOpReprMat m1(1, 5), m2(5, 6), bchmk_m1(1, 4), bchmk_m2(4, 6);
  m1.SetElem(0, 0, kIdOpRepr);
  m1.SetElem(0, 1, OpRepr(j, sx));
  m1.SetElem(0, 2, OpRepr(j, sy));
  m1.SetElem(0, 3, OpRepr(j, sz));
  m1.SetElem(0, 4, OpRepr(k, sx));
  m2.SetElem(0, 0, OpRepr(j, sx));
  m2.SetElem(0, 1, OpRepr(j, sy));
  m2.SetElem(0, 2, OpRepr(j, sz));
  m2.SetElem(0, 4, OpRepr(k, sz));
  m2.SetElem(1, 5, OpRepr(sx));
  m2.SetElem(2, 5, OpRepr(sy));
  m2.SetElem(3, 5, OpRepr(sz));
  m2.SetElem(4, 3, OpRepr(sx));
  bchmk_m1.SetElem(0, 0, kIdOpRepr);
  bchmk_m1.SetElem(0, 1, OpRepr(sx));
  bchmk_m1.SetElem(0, 2, OpRepr(sy));
  bchmk_m1.SetElem(0, 3, OpRepr(sz));
  bchmk_m2.SetElem(0, 0, OpRepr(j, sx));
  bchmk_m2.SetElem(0, 1, OpRepr(j, sy));
  bchmk_m2.SetElem(0, 2, OpRepr(j, sz));
  bchmk_m2.SetElem(0, 4, OpRepr(k, sz));
  bchmk_m2.SetElem(1, 3, OpRepr(k, sx));
  bchmk_m2.SetElem(1, 5, OpRepr(j, sx));
  bchmk_m2.SetElem(2, 5, OpRepr(j, sy));
  bchmk_m2.SetElem(3, 5, OpRepr(j, sz));
  SparOpReprMatColCompresser(m1, m2);
  EXPECT_EQ(m1, bchmk_m1);
  EXPECT_EQ(m2, bchmk_m2);
}

void RunTestSparOpReprMatColCompresserCase11(void) {
  OpRepr s(1);
  SparOpReprMat m1(2, 2), m2(2, 2);
  m1.SetElem(0, 1, s);
  m1.SetElem(1, 0, kIdOpRepr);
  m1.SetElem(1, 1, kIdOpRepr);
  m2.SetElem(0, 0, kIdOpRepr);
  m2.SetElem(1, 1, s);
  auto bchmk_m1 = m1;
  auto bchmk_m2 = m2;
  SparOpReprMatColCompresser(m1, m2);
  EXPECT_EQ(m1, bchmk_m1);
  EXPECT_EQ(m2, bchmk_m2);
}

TEST(TestSparOpReprMat, TestSparOpReprMatColCompresser) {
  RunTestSparOpReprMatColCompresserCase1();
  RunTestSparOpReprMatColCompresserCase2();
  RunTestSparOpReprMatColCompresserCase3();
  RunTestSparOpReprMatColCompresserCase4();
  RunTestSparOpReprMatColCompresserCase5();
  RunTestSparOpReprMatColCompresserCase6();
  RunTestSparOpReprMatColCompresserCase7();
  RunTestSparOpReprMatColCompresserCase8();
  RunTestSparOpReprMatColCompresserCase9();
  RunTestSparOpReprMatColCompresserCase10();
  RunTestSparOpReprMatColCompresserCase11();
}
