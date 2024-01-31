// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2019-10-30 18:32
* 
* Description: QuantumLiquids/UltraDMRG project. Data structure of a special sparse matrix.
*/
#ifndef QLMPS_ONE_DIM_TN_MPO_MPOGEN_SPARSE_MAT_H
#define QLMPS_ONE_DIM_TN_MPO_MPOGEN_SPARSE_MAT_H

#include <iostream>
#include <vector>
#include "assert.h"

/**
 * Raw-major sparse matrix
 * @tparam ElemType
 */
template<typename ElemType>
class SparMat {
 public:
  SparMat(void) : rows(0), cols(0), data(), indexes() {}

  SparMat(const size_t row_num, const size_t col_num) :
      rows(row_num), cols(col_num),
      data(), indexes(row_num * col_num, -1) {}

  SparMat(const SparMat<ElemType> &spar_mat) :
      rows(spar_mat.rows), cols(spar_mat.cols),
      data(spar_mat.data), indexes(spar_mat.indexes) {}

  SparMat<ElemType> &operator=(const SparMat<ElemType> &spar_mat) {
    rows = spar_mat.rows;
    cols = spar_mat.cols;
    data = spar_mat.data;
    indexes = spar_mat.indexes;
    return *this;
  }

  /**
   * Element getter and setter.
   *
   * @param row   row number
   * @param col   column number
   * @return
   */
  const ElemType &operator()(const size_t row, const size_t col) const {
    auto offset = CalcOffset(row, col);
    if (indexes[offset] == -1) {
      return nullelem;
    } else {
      return data[indexes[offset]];
    }
  }

  void Reserve(const size_t size) {
    data.reserve(size);
  }

  bool IsNull(const size_t x, const size_t y) const {
    auto offset = CalcOffset(x, y);
    if (indexes[offset] == -1) {
      return true;
    } else {
      return false;
    }
  }

  void SetElem(const size_t x, const size_t y, const ElemType &elem) {
    if (elem == nullelem) { return; }
    auto offset = CalcOffset(x, y);
    if (indexes[offset] == -1) {
      data.push_back(elem);
      long idx = data.size() - 1;
      indexes[offset] = idx;
    } else {
      data[indexes[offset]] = elem;
    }
  }

  // Get row and column.
  std::vector<ElemType> GetRow(const size_t row_idx) const {
    assert(row_idx < rows);
    std::vector<ElemType> row;
    row.reserve(cols);
    for (size_t y = 0; y < cols; ++y) {
      row.push_back((*this)(row_idx, y));
    }
    return row;
  }

  std::vector<ElemType> GetCol(const size_t col_idx) const {
    assert(col_idx < cols);
    std::vector<ElemType> col;
    col.reserve(rows);
    for (size_t x = 0; x < rows; ++x) {
      col.push_back((*this)(x, col_idx));
    }
    return col;
  }

  // overload operator==
  bool operator==(const SparMat<ElemType> &rhs) const {
    if (rows != rhs.rows) {
      std::cout << "No same rows!" << std::endl;
      return false;
    }
    if (cols != rhs.cols) {
      std::cout << "No same cols!" << std::endl;
      return false;
    }
    for (size_t x = 0; x < rows; ++x) {
      for (size_t y = 0; y < cols; ++y) {
        if ((*this)(x, y) != rhs(x, y)) {
          std::cout << "No same elem at (" << x << "," << y << ")" << std::endl;
          return false;
        }
      }
    }
    return true;
  }

  // Remove row and column.
  void RemoveRow(const size_t row_idx) {
    assert(row_idx < rows);
    if (rows == 1) {
      *this = SparMat<ElemType>();
      return;
    }
    indexes.erase(indexes.cbegin() + CalcOffset(row_idx, 0),
                  indexes.cbegin() + CalcOffset(row_idx + 1, 0)
    );
    rows = rows - 1;
  }

  void RemoveCol(const size_t col_idx) {
    assert(col_idx < cols);
    if (cols == 1) {
      *this = SparMat<ElemType>();
      return;
    }
    cols = cols - 1;
    const size_t moving_piece_size = cols;
    for (size_t x = 0; x < rows - 1; ++x) {
      size_t delete_elem_number = x + 1;
      size_t fulling_piece_start = CalcOffset(x, col_idx);
      for (size_t i = 0; i < moving_piece_size; i++) {
        indexes[fulling_piece_start + i] = indexes[fulling_piece_start + i + delete_elem_number];
      }
    }
    size_t x = rows - 1;
    size_t delete_elem_number = x + 1;
    size_t fulling_piece_start = CalcOffset(x, col_idx);
    for (size_t i = 0; i < cols * rows - fulling_piece_start; i++) {
      indexes[fulling_piece_start + i] = indexes[fulling_piece_start + i + delete_elem_number];
    }
    indexes.erase(indexes.begin() + cols * rows, indexes.cend());
  }

  // Swap two rows and columns.
  void SwapTwoRows(const size_t row_idx1, const size_t row_idx2) {
    assert(row_idx1 < rows && row_idx2 < rows);
    if (row_idx1 == row_idx2) { return; }
    for (size_t y = 0; y < cols; ++y) {
      auto offset1 = CalcOffset(row_idx1, y);
      auto offset2 = CalcOffset(row_idx2, y);
      std::swap(indexes[offset1], indexes[offset2]);
    }
  }

  void SwapTwoCols(const size_t col_idx1, const size_t col_idx2) {
    assert(col_idx1 < cols && col_idx2 < cols);
    if (col_idx1 == col_idx2) { return; }
    for (size_t x = 0; x < rows; ++x) {
      auto offset1 = CalcOffset(x, col_idx1);
      auto offset2 = CalcOffset(x, col_idx2);
      std::swap(indexes[offset1], indexes[offset2]);
    }
  }

  // Transpose rows and columns.
  void TransposeRows(const std::vector<size_t> &transposed_row_idxs) {
    assert(transposed_row_idxs.size() == rows);
    std::vector<long> new_indexes(indexes.size());
    for (size_t i = 0; i < rows; ++i) {
      auto transposed_row_idx = transposed_row_idxs[i];
      for (size_t y = 0; y < cols; ++y) {
        new_indexes[CalcOffset(i, y)] =
            indexes[CalcOffset(transposed_row_idx, y)];
      }
    }
    indexes = new_indexes;
  }

  void TransposeCols(const std::vector<size_t> &transposed_col_idxs) {
    assert(transposed_col_idxs.size() == cols);
    std::vector<long> new_indexes(indexes.size());
    for (size_t i = 0; i < cols; ++i) {
      auto transposed_col_idx = transposed_col_idxs[i];
      for (size_t x = 0; x < rows; ++x) {
        new_indexes[CalcOffset(x, i)] =
            indexes[CalcOffset(x, transposed_col_idx)];
      }
    }
    indexes = new_indexes;
  }

  size_t CalcOffset(const size_t row, const size_t col) const {
    return row * cols + col;
  }

  void Print() const {
    for (size_t row = 0; row < rows; row++) {
      std::vector<ElemType> row_data = GetRow(row);
      for (size_t col = 0; col < row_data.size(); col++) {
        std::cout << row_data[col] << "\t";
      }
      std::cout << "\n";
    }
    std::cout << std::endl;
  }

  size_t rows;
  size_t cols;
  std::vector<ElemType> data;
  std::vector<long> indexes;

 private:
  size_t CalcOffset_(
      const size_t x, const size_t y, const size_t new_cols) const {
    return x * new_cols + y;
  }

  static ElemType nullelem;
};

template<typename ElemType>
ElemType SparMat<ElemType>::nullelem = ElemType();
#endif /* ifndef QLMPS_ONE_DIM_TN_MPO_MPOGEN_SPARSE_MAT_H */
