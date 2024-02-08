// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Hao-Xin Wang <wanghaoxin1996@gmail.com>
* Creation Date: 2019-10-29 15:35
* 
* Description: QuantumLiquids/UltraDMRG project. Algebra of MPO's coefficient and operator.
*/
#ifndef QLMPS_ONE_DIM_TN_MPO_MPOGEN_COEF_OP_ALG_H
#define QLMPS_ONE_DIM_TN_MPO_MPOGEN_COEF_OP_ALG_H

#include <vector>
#include <algorithm>
#include <iostream>
#include <map>
#include <assert.h>
#include "qlmps/one_dim_tn/mpo/mpogen/symb_alg/sparse_mat.h"

#ifdef Release
#define NDEBUG
#endif

// Forward declarations.
template<typename T>
bool ElemInVec(const T &, const std::vector<T> &, long &pos);

template<typename VecT>
VecT ConcatenateTwoVec(const VecT &, const VecT &);


// Label of c-number.
using CNumberLabel = long;

const CNumberLabel kIdCoefLabel = 0;     // Coefficient label for identity 1.


/**
 * Represent coefficient as a summation of c-numbers which are represented as labels.
 */
class CoefRepr {

 public:
  CoefRepr(void) : coef_label_list_() {}

  CoefRepr(const CNumberLabel &coef_label) :
      coef_label_list_{coef_label} {}

  CoefRepr(const std::vector<CNumberLabel> &coef_label_list) :
      coef_label_list_(coef_label_list) {}

  CoefRepr(const CoefRepr &coef_repr) :
      coef_label_list_(coef_repr.coef_label_list_) {}

  CoefRepr(CoefRepr &&coef_repr) :
      coef_label_list_(std::move(coef_repr.coef_label_list_)) {}

  CoefRepr &operator=(const CoefRepr &rhs) {
    coef_label_list_ = rhs.coef_label_list_;
    return *this;
  }

  CoefRepr &operator=(CoefRepr &&rhs) {
    coef_label_list_ = std::move(rhs.coef_label_list_);
    return *this;
  }

  std::vector<CNumberLabel> GetCoefLabelList(void) const {
    return coef_label_list_;
  }

  bool operator==(const CoefRepr &rhs) const {
    return std::is_permutation(coef_label_list_.begin(), coef_label_list_.end(),
                               rhs.coef_label_list_.begin(), rhs.coef_label_list_.end());
  }

  bool operator!=(const CoefRepr &rhs) const {
    return !(*this == rhs);
  }

  bool operator<(const CoefRepr &rhs) const {
    return coef_label_list_ < rhs.coef_label_list_;
  }

  bool operator<=(const CoefRepr &rhs) const {
    return coef_label_list_ <= rhs.coef_label_list_;
  }

  bool operator>(const CoefRepr &rhs) const {
    return coef_label_list_ > rhs.coef_label_list_;
  }

  bool operator>=(const CoefRepr &rhs) const {
    return coef_label_list_ >= rhs.coef_label_list_;
  }

  CoefRepr &operator+=(const CoefRepr &rhs) {
    coef_label_list_.insert(
        coef_label_list_.end(),
        rhs.coef_label_list_.begin(),
        rhs.coef_label_list_.end()
    );
    return *this;
  }

  CoefRepr operator+(const CoefRepr &rhs) const {
    return CoefRepr(ConcatenateTwoVec(coef_label_list_, rhs.coef_label_list_));
  }

  template<typename CoefT>
  CoefT Realize(const std::vector<CoefT> &label_coef_mapping) const {
    CoefT coef = 0;
    for (auto coef_label : coef_label_list_) {
      coef += label_coef_mapping[coef_label];
    }
    return coef;
  }

  template<typename CoefT>
  std::string ToString(const std::vector<CoefT> &label_coef_mapping) const {
    CoefT coef = Realize(label_coef_mapping);
    return std::to_string(coef);
  }

  std::string ToString() const {
    if (coef_label_list_.size() == 0) {
      return "0";
    } else if (coef_label_list_.size() == 1) {
      CNumberLabel coef_label = coef_label_list_[0];
      if (coef_label == kIdCoefLabel) {
        return "1";
      } else {
        char coef_symbol = 'a' + coef_label - 1;
        return std::string(1, coef_symbol);
      }
    } else {
      std::string symbol_string = "(";
      for (size_t i = 0; i < coef_label_list_.size(); i++) {
        const CNumberLabel coef_label = coef_label_list_[1];
        if (coef_label == kIdCoefLabel) {
          symbol_string += "1";
        } else {
          char coef_symbol = 'a' + coef_label - 1;
          symbol_string.push_back(coef_symbol);
        }
        if (i < coef_label_list_.size() - 1) {
          symbol_string += "+";
        } else {
          symbol_string += ")";
        }
      }
      return symbol_string;
    }
  }

 private:
  std::vector<CNumberLabel> coef_label_list_;
};

const CoefRepr kNullCoefRepr = CoefRepr();            // Coefficient representation for null coefficient.
const CoefRepr kIdCoefRepr = CoefRepr(kIdCoefLabel);  // Coefficient representation for identity coefficient 1.

using CoefReprVec = std::vector<CoefRepr>;


// label of basis operators
using BasisOpLabel = long;

// Representation of operator.
class SparOpReprMat;    // Forward declaration.

class OpRepr {
  friend std::pair<CoefRepr, OpRepr> SeparateCoefAndBase(const OpRepr &);

  friend OpRepr CoefReprOpReprIncompleteMulti(const CoefRepr &, const OpRepr &);

  friend std::vector<OpRepr> CalcSparOpReprMatRowLinCmb(
      const SparOpReprMat &, const CoefReprVec &);

  friend std::vector<OpRepr> CalcSparOpReprMatColLinCmb(
      const SparOpReprMat &, const CoefReprVec &);

 public:
  OpRepr(void) : op_label_coef_repr_map_() {}

  OpRepr(const BasisOpLabel op_label) {
    op_label_coef_repr_map_.insert(std::make_pair(op_label, kIdCoefRepr));
  }

  OpRepr(const CoefRepr &coef_repr, const BasisOpLabel op_label) {
    op_label_coef_repr_map_.insert(std::make_pair(op_label, coef_repr));
  }

  OpRepr(
      const std::vector<CoefRepr> &coef_reprs,
      const std::vector<BasisOpLabel> &op_labels) {
    for (size_t i = 0; i < op_labels.size(); ++i) {
      auto poss_it = op_label_coef_repr_map_.find(op_labels[i]);
      if (poss_it == op_label_coef_repr_map_.end()) {
        op_label_coef_repr_map_.insert(std::make_pair(op_labels[i], coef_reprs[i]));
      } else {
        CoefRepr &that_coef_repr = op_label_coef_repr_map_.at(op_labels[i]);
        that_coef_repr += coef_reprs[i];
      }
    }
  }

  OpRepr(const OpRepr &rhs) {
    op_label_coef_repr_map_ = rhs.op_label_coef_repr_map_;
  }

  OpRepr(const std::vector<BasisOpLabel> &single_op_labels) :
      OpRepr(CoefReprVec(single_op_labels.size(), kIdCoefRepr), single_op_labels) {}

  std::vector<CoefRepr> GetCoefReprList(void) const {
    std::vector<CoefRepr> coef_repr_list;
    coef_repr_list.reserve(op_label_coef_repr_map_.size());
    for (auto iter = op_label_coef_repr_map_.cbegin(); iter != op_label_coef_repr_map_.cend(); iter++) {
      coef_repr_list.push_back(iter->second);
    }
    return coef_repr_list;
  }

  std::vector<BasisOpLabel> GetOpLabelList(void) const {
    std::vector<BasisOpLabel> op_label_list;
    op_label_list.reserve(op_label_coef_repr_map_.size());
    for (auto iter = op_label_coef_repr_map_.cbegin(); iter != op_label_coef_repr_map_.cend(); iter++) {
      op_label_list.push_back(iter->first);
    }
    return op_label_list;
  }

  // Assignment operator
  OpRepr &operator=(const OpRepr &rhs) {
    if (this != &rhs) {
      op_label_coef_repr_map_ = rhs.op_label_coef_repr_map_;
    }
    return *this;
  }

  bool operator<(const OpRepr &rhs) const {
    return op_label_coef_repr_map_ < rhs.op_label_coef_repr_map_;
  }

  bool operator==(const OpRepr &rhs) const {
    return op_label_coef_repr_map_ == rhs.op_label_coef_repr_map_;
  }

  bool operator!=(const OpRepr &rhs) const {
    return !(*this == rhs);
  }

  OpRepr &operator+=(const OpRepr &rhs) {
    const std::map<BasisOpLabel, CoefRepr> &rhs_op_lable_coef_repr_map = rhs.GetOpLabelCoefReprMap_();
    for (auto iter = rhs_op_lable_coef_repr_map.cbegin();
         iter != rhs_op_lable_coef_repr_map.cend();
         iter++) {
      auto poss_it = op_label_coef_repr_map_.find(iter->first);
      if (poss_it == op_label_coef_repr_map_.end()) {
        op_label_coef_repr_map_.insert(std::make_pair(iter->first, iter->second));
      } else {
        CoefRepr &that_coef_repr = op_label_coef_repr_map_.at(iter->first);
        that_coef_repr += iter->second;
      }
    }
    return *this;
  }

  OpRepr operator+(const OpRepr &rhs) const {
    OpRepr res(*this);
    res += rhs;
    return res;
  }

  template<typename CoefT, typename OpT>
  OpT Realize(
      const std::vector<CoefT> &label_coef_mapping,
      const std::vector<OpT> &label_op_mapping) {
    auto base_op_num = op_label_coef_repr_map_.size();
    OpT op;
    if (base_op_num == 0) {
      return OpT();
    } else if (base_op_num == 1) {
      return op_label_coef_repr_map_.begin()->second.Realize(label_coef_mapping) *
          label_op_mapping[op_label_coef_repr_map_.begin()->first];
    } else {
      op = op_label_coef_repr_map_.begin()->second.Realize(label_coef_mapping) *
          label_op_mapping[op_label_coef_repr_map_.begin()->first];
      auto iter = op_label_coef_repr_map_.begin();
      iter++;
      for (; iter != op_label_coef_repr_map_.end(); iter++) {
        op += iter->second.Realize(label_coef_mapping) *
            label_op_mapping[iter->first];
      }
    }
    return op;
  }

  template<typename CoefT>
  std::string ToString(const std::vector<CoefT> &label_coef_mapping
  ) const {
    auto base_op_num = op_label_coef_repr_map_.size();
    if (base_op_num == 0) {
      return "0";
    } else if (base_op_num == 1) {
      std::string op_symbol = op_label_coef_repr_map_.begin()->second.ToString(label_coef_mapping);
      op_symbol += OpLabelToString_(op_label_coef_repr_map_.begin()->first);
      return op_symbol;
    } else {
      std::string op_symbol = op_label_coef_repr_map_.begin()->second.ToString(label_coef_mapping);
      op_symbol += OpLabelToString_(op_label_coef_repr_map_.begin()->first);

      auto iter = op_label_coef_repr_map_.begin();
      iter++;
      for (; iter != op_label_coef_repr_map_.end(); iter++) {
        op_symbol += "+";
        op_symbol += iter->second.ToString(label_coef_mapping);
        op_symbol += OpLabelToString_(iter->first);
      }
      return op_symbol;
    }
  }

  std::string ToString() const {
    auto base_op_num = op_label_coef_repr_map_.size();
    if (base_op_num == 0) {
      return "0";
    } else if (base_op_num == 1) {
      std::string op_symbol = op_label_coef_repr_map_.begin()->second.ToString();
      op_symbol += OpLabelToString_(op_label_coef_repr_map_.begin()->first);
      return op_symbol;
    } else {
      std::string op_symbol = op_label_coef_repr_map_.begin()->second.ToString();
      op_symbol += OpLabelToString_(op_label_coef_repr_map_.begin()->first);

      auto iter = op_label_coef_repr_map_.begin();
      iter++;
      for (; iter != op_label_coef_repr_map_.end(); iter++) {
        op_symbol += "+";
        op_symbol += iter->second.ToString();
        op_symbol += OpLabelToString_(iter->first);
      }
      return op_symbol;
    }
  }

 private:
  const std::map<BasisOpLabel, CoefRepr> &GetOpLabelCoefReprMap_(void) const {
    return op_label_coef_repr_map_;
  }

  std::string OpLabelToString_(BasisOpLabel label) const {
    if (label == kIdCoefLabel) {
      return "Id";
    } else {
      if (label <= 26) {
        return std::string(1, 'A' + label - 1);
      } else {
        return std::string(1, 'A' + (label - 1) % 26) + std::to_string((label - 1) / 26);
      }

    }
  }

  std::map<BasisOpLabel, CoefRepr> op_label_coef_repr_map_;
};

inline std::ostream &operator<<(std::ostream &os, const OpRepr &op_repr) {
  os << op_repr.ToString();
  return os;
}

const OpRepr kNullOpRepr = OpRepr();          // Operator representation for null operator.

using OpReprVec = std::vector<OpRepr>;

inline std::pair<CoefRepr, OpRepr> SeparateCoefAndBase(const OpRepr &op_repr) {
  auto term_num = op_repr.op_label_coef_repr_map_.size();
  if (term_num == 0) {
    return std::make_pair(kNullCoefRepr, kNullOpRepr);
  } else if (term_num == 1) {
    return std::make_pair(op_repr.op_label_coef_repr_map_.begin()->second,
                          OpRepr(op_repr.op_label_coef_repr_map_.begin()->first));
  } else {
    auto iter = op_repr.op_label_coef_repr_map_.begin();
    auto coef = iter->second;
    iter++;
    for (; iter != op_repr.op_label_coef_repr_map_.end(); iter++) {
      auto coef1 = iter->second;
      if (coef1 != coef) {
        return std::make_pair(kIdCoefRepr, OpRepr(op_repr));
      }
    }
    auto res_op_repr = op_repr;
    for (std::map<BasisOpLabel, CoefRepr>::iterator iter = res_op_repr.op_label_coef_repr_map_.begin();
         iter != res_op_repr.op_label_coef_repr_map_.end();
         iter++) {
      iter->second = kIdCoefRepr;
    }
    return std::make_pair(coef, res_op_repr);
  }
}

inline CoefRepr GetOpReprCoef(const OpRepr &op_repr) {
  return SeparateCoefAndBase(op_repr).first;
}


// Sparse coefficient representation matrix.
using SparCoefReprMat = SparMat<CoefRepr>;


// Sparse operator representation matrix.
using SparOpReprMatBase = SparMat<OpRepr>;

class SparOpReprMat : public SparOpReprMatBase {
 public:
  SparOpReprMat(void) : SparOpReprMatBase() {}

  SparOpReprMat(const size_t row_num, const size_t col_num) :
      SparOpReprMatBase(row_num, col_num) {}

  SparOpReprMat(const SparOpReprMat &spar_mat) :
      SparOpReprMatBase(spar_mat) {}

  SparOpReprMat &operator=(const SparOpReprMat &spar_mat) {
    rows = spar_mat.rows;
    cols = spar_mat.cols;
    data = spar_mat.data;
    indexes = spar_mat.indexes;
    return *this;
  }

  ///< sort row according # of non null element, return the order of sorting
  std::vector<size_t> SortRows(void) {
    auto mapping = GenSortRowsMapping_();
    std::sort(mapping.begin(), mapping.end());
    std::vector<size_t> sorted_row_idxs(rows);
    for (size_t i = 0; i < rows; ++i) {
      sorted_row_idxs[i] = mapping[i].second;
    }
    TransposeRows(sorted_row_idxs);
    return sorted_row_idxs;
  }

  std::vector<size_t> SortCols(void) {
    auto mapping = GenSortColsMapping_();
    std::sort(mapping.begin(), mapping.end());
    std::vector<size_t> sorted_col_idxs(cols);
    for (size_t i = 0; i < cols; ++i) {
      sorted_col_idxs[i] = mapping[i].second;
    }
    TransposeCols(sorted_col_idxs);
    return sorted_col_idxs;
  }

  /**
   * A rough implementation. Only work when the coefs are all the same
   * @param row_idx
   * @return
   */
  CoefRepr TryCatchRowCommonDivisorCoef(const size_t row_idx) {
    size_t nonull_op_repr_coefs_num = 0;
    CoefRepr res_coef, res_coef_last;
    std::vector<CoefRepr> nonull_op_repr_coefs;
    for (size_t y = 0; y < cols; ++y) {
      if (indexes[CalcOffset(row_idx, y)] != -1) {
        nonull_op_repr_coefs_num++;
        if (nonull_op_repr_coefs_num == 1) {
          res_coef = GetOpReprCoef((*this)(row_idx, y));
        } else {
          res_coef_last = res_coef;
          res_coef = GetOpReprCoef((*this)(row_idx, y));
          if (res_coef_last != res_coef) {
            return kIdCoefRepr;
          }
        }
      }
    }
    return res_coef;
  }

  /**
   * A rough implementation. Only work when the coefs are all the same
   *
   * @param col_idx
   * @return
   */
  CoefRepr TryCatchColCommonDivisorCoef(const size_t col_idx) {
    size_t nonull_op_repr_coefs_num = 0;
    CoefRepr res_coef, res_coef_last;
    std::vector<CoefRepr> nonull_op_repr_coefs;
    for (size_t x = 0; x < rows; ++x) {
      if (indexes[CalcOffset(x, col_idx)] != -1) {
        nonull_op_repr_coefs_num++;
        if (nonull_op_repr_coefs_num == 1) {
          res_coef = GetOpReprCoef((*this)(x, col_idx));
        } else {
          res_coef_last = res_coef;
          res_coef = GetOpReprCoef((*this)(x, col_idx));
          if (res_coef_last != res_coef) {
            return kIdCoefRepr;
          }
        }
      }
    }
    return res_coef;
  }

  void RemoveRowCoef(const size_t row_idx) {
    for (size_t y = 0; y < cols; ++y) {
      if (indexes[CalcOffset(row_idx, y)] != -1) {
        auto elem = (*this)(row_idx, y);
        this->SetElem(row_idx, y, SeparateCoefAndBase(elem).second);
      }
    }
  }

  /**
   * Remove all the coefficients in one columns
   * @param col_idx
   */
  void RemoveColCoef(const size_t col_idx) {
    for (size_t x = 0; x < rows; ++x) {
      if (indexes[CalcOffset(x, col_idx)] != -1) {
        auto elem = (*this)(x, col_idx);
        this->SetElem(x, col_idx, SeparateCoefAndBase(elem).second);
      }
    }
  }

  ///< linear combination coefficients for expanding row_idx-th row as summation of previous rows.
  CoefReprVec CalcRowLinCmb(const size_t row_idx) const {
    auto row = GetRow(row_idx);
    CoefReprVec cmb_coefs;
    for (size_t x = 0; x < row_idx; ++x) {
      if (row == GetRow(x)) {
        cmb_coefs = CoefReprVec(row_idx, kNullCoefRepr);
        cmb_coefs[x] = kIdCoefRepr;
        return cmb_coefs;
      }
    }
    cmb_coefs.reserve(row_idx);
    for (size_t x = 0; x < row_idx; ++x) {
      cmb_coefs.emplace_back(CalcRowOverlap_(row, x));
    }
    return cmb_coefs;
  }

  CoefReprVec CalcColLinCmb(const size_t col_idx) const {
    auto col = GetCol(col_idx);
    CoefReprVec cmb_coefs;
    cmb_coefs.reserve(col_idx);
    for (size_t y = 0; y < col_idx; ++y) {
      cmb_coefs.emplace_back(CalcColOverlap_(col, y));
    }
    return cmb_coefs;
  }

 private:
  using SortMapping = std::vector<std::pair<size_t, size_t>>;   // # of no null : row_idx

  SortMapping GenSortRowsMapping_(void) const {
    SortMapping mapping;
    mapping.reserve(rows);
    for (size_t x = 0; x < rows; ++x) {
      size_t nonull_elem_num = 0;
      for (size_t y = 0; y < cols; ++y) {
        if (indexes[CalcOffset(x, y)] != -1) { nonull_elem_num++; }
      }
      mapping.push_back(std::make_pair(nonull_elem_num, x));
    }
    return mapping;
  }

  SortMapping GenSortColsMapping_(void) const {
    SortMapping mapping;
    mapping.reserve(cols);
    for (size_t y = 0; y < cols; ++y) {
      size_t nonull_elem_num = 0;
      for (size_t x = 0; x < rows; ++x) {
        if (indexes[CalcOffset(x, y)] != -1) { nonull_elem_num++; }
      }
      mapping.push_back(std::make_pair(nonull_elem_num, y));
    }
    return mapping;
  }

/**
 * @todo optimize
 */
  CoefRepr CalcRowOverlap_(
      const std::vector<OpRepr> &row, const size_t tgt_row_idx) const {
    CoefReprVec poss_overlaps;
    poss_overlaps.reserve(cols); //local variable so we can reserve more.
    for (size_t y = 0; y < cols; ++y) {
      if (indexes[CalcOffset(tgt_row_idx, y)] != -1) {
        auto tgt_op = row[y];
        auto base_op = (*this)(tgt_row_idx, y);
        if (tgt_op == base_op) {
          poss_overlaps.push_back(kIdCoefRepr);
        } else {
          std::pair<CoefRepr, OpRepr> tgt_coef_and_base_op = SeparateCoefAndBase(tgt_op);
          if (tgt_coef_and_base_op.second == base_op) {
            poss_overlaps.emplace_back(tgt_coef_and_base_op.first);
          } else {
            return kNullCoefRepr;
          }
        }
        if (poss_overlaps.size() > 1 && poss_overlaps.back() != poss_overlaps[0]) {
          return kNullCoefRepr;
        }
      }
    }
    if (poss_overlaps.empty()) { return kNullCoefRepr; }
    return poss_overlaps[0];
  }

  CoefRepr CalcColOverlap_(
      const std::vector<OpRepr> &col, const size_t tgt_col_idx) const {
    CoefReprVec poss_overlaps;
    poss_overlaps.reserve(rows);
    for (size_t x = 0; x < rows; ++x) {
      if (indexes[CalcOffset(x, tgt_col_idx)] != -1) {
        auto tgt_op = col[x];
        auto base_op = (*this)(x, tgt_col_idx);
        if (tgt_op == base_op) {
          poss_overlaps.push_back(kIdCoefRepr);
        } else {
          auto tgt_coef_and_base_op = SeparateCoefAndBase(tgt_op);
          if (tgt_coef_and_base_op.second == base_op) {
            poss_overlaps.emplace_back(tgt_coef_and_base_op.first);
          } else {
            return kNullCoefRepr;
          }
        }
        if (poss_overlaps.size() > 1) {
          if (poss_overlaps.back() != poss_overlaps[0]) {
            return kNullCoefRepr;
          }
        }
      }
    }
    if (poss_overlaps.empty()) { return kNullCoefRepr; }
    return poss_overlaps[0];
  }
};

using SparOpReprMatVec = std::vector<SparOpReprMat>;

/** Incomplete multiplication for SparMat.
 * `op` must be `kNullOpRepr` or its coefficients must all be identity, or `coef` is identity.
 */
inline OpRepr CoefReprOpReprIncompleteMulti(const CoefRepr &coef, const OpRepr &op) {
  if (op == kNullOpRepr) { return kNullOpRepr; }
  if (coef == kIdCoefRepr) { return op; }
#ifndef NDEBUG
  for (auto &[sub_op, c] : op.op_label_coef_repr_map_) {
    if (c != kIdCoefRepr) {
      std::cout << "CoefReprOpReprIncompleteMulti fail!" << std::endl;
      exit(1);
    }
  }
#endif
  OpRepr res_op_repr(op);
  for (auto &[sub_op, c] : res_op_repr.op_label_coef_repr_map_) {
    c = coef;
  }
  return res_op_repr;
}

inline void SparCoefReprMatSparOpReprMatIncompleteMultiKernel(
    const SparCoefReprMat &coef_mat, const SparOpReprMat &op_mat,
    const size_t coef_mat_row_idx, const size_t op_mat_col_idx,
    SparOpReprMat &res) {
  OpRepr res_elem;
  size_t coef_elem_offset = coef_mat_row_idx * coef_mat.cols;
  size_t op_elem_offset = op_mat_col_idx;
  for (size_t i = 0; i < coef_mat.cols; ++i) {
    int coef_mat_index = coef_mat.indexes[coef_elem_offset];
    int op_mat_idx = op_mat.indexes[op_elem_offset];
    if (coef_mat_index != -1 &&
        op_mat_idx != -1) {
      res_elem += CoefReprOpReprIncompleteMulti(
          coef_mat.data[coef_mat_index],
          op_mat.data[op_mat_idx]);
    }
    coef_elem_offset++;
    op_elem_offset += op_mat.cols;
  }
  if (res_elem != kNullOpRepr) {
    res.SetElem(coef_mat_row_idx, op_mat_col_idx, res_elem);
  }
}

inline void SparOpReprMatSparCoefReprMatIncompleteMultiKernel(
    const SparOpReprMat &op_mat, const SparCoefReprMat &coef_mat,
    const size_t op_mat_row_idx, const size_t coef_mat_col_idx,
    SparOpReprMat &res) {
  OpRepr res_elem;
  for (size_t i = 0; i < op_mat.cols; ++i) {
    if (op_mat.indexes[op_mat.CalcOffset(op_mat_row_idx, i)] != -1 &&
        coef_mat.indexes[coef_mat.CalcOffset(i, coef_mat_col_idx)] != -1) {
      res_elem += CoefReprOpReprIncompleteMulti(
          coef_mat(i, coef_mat_col_idx),
          op_mat(op_mat_row_idx, i));
    }
  }
  if (res_elem != kNullOpRepr) {
    res.SetElem(op_mat_row_idx, coef_mat_col_idx, res_elem);
  }
}

///< performance hot pots
inline SparOpReprMat SparCoefReprMatSparOpReprMatIncompleteMulti(
    const SparCoefReprMat &coef_mat, const SparOpReprMat &op_mat) {
  assert(coef_mat.cols == op_mat.rows);
  SparOpReprMat res(coef_mat.rows, op_mat.cols);
  // future:
  // #pragma omp parallel for default(none) \
  //                          shared(coef_mat, op_mat, res) \
  //                          schedule(dynamic) \
  //                          num_threads(4)
  // for(size_t i = 0; i < coef_mat.rows * op_mat.cols; i++){
  //   const size_t x = i / op_mat.cols;
  //   const size_t y = i % op_mat.cols;
  //   SparCoefReprMatSparOpReprMatIncompleteMultiKernel(
  //         coef_mat, op_mat, x, y, res);
  // }

  for (size_t x = 0; x < coef_mat.rows; ++x) {
    for (size_t y = 0; y < op_mat.cols; ++y) {
      SparCoefReprMatSparOpReprMatIncompleteMultiKernel(
          coef_mat, op_mat, x, y, res);
    }
  }
  return res;
}

// TODO: important optimize point (more than 80% time sometime)
inline SparOpReprMat SparOpReprMatSparCoefReprMatIncompleteMulti(
    const SparOpReprMat &op_mat, const SparCoefReprMat &coef_mat) {
  assert(op_mat.cols == coef_mat.rows);
  SparOpReprMat res(op_mat.rows, coef_mat.cols);
  for (size_t x = 0; x < op_mat.rows; ++x) {
    for (size_t y = 0; y < coef_mat.cols; ++y) {
      SparOpReprMatSparCoefReprMatIncompleteMultiKernel(
          op_mat, coef_mat, x, y, res);
    }
  }
  return res;
}


// Row and column delinearization.
/* TODO: So bad implementation, need refactor. */
/**
 * only support special case
 */
inline OpReprVec CalcSparOpReprMatRowLinCmb(
    const SparOpReprMat &m, const CoefReprVec &cmb) {
  auto work_row_num = cmb.size();
  assert(work_row_num > 0);
  auto res = OpReprVec(m.cols, kNullOpRepr);
  for (size_t i = 0; i < work_row_num; ++i) {
    auto cmb_coef = cmb[i];
    auto row = m.GetRow(i);
    if (cmb_coef == kIdCoefRepr) {
      for (size_t j = 0; j < m.cols; ++j) {
        res[j] += row[j];
      }
    } else if (cmb_coef == kNullCoefRepr) {
      // Do nothing.
    } else {
      for (size_t j = 0; j < m.cols; ++j) {
        auto elem = row[j];
        for (auto &[sub_op, coef_repr] : elem.op_label_coef_repr_map_) {
          assert(coef_repr == kIdCoefRepr); //To Do other case
          coef_repr = cmb_coef;
        }
        res[j] += elem;
      }
    }
  }
  return res;
}

// only support special case
inline OpReprVec CalcSparOpReprMatColLinCmb(
    const SparOpReprMat &m, const CoefReprVec &cmb) {
  auto work_col_num = cmb.size();
  assert(work_col_num > 0);
  auto res = OpReprVec(m.rows, kNullOpRepr);
  for (size_t i = 0; i < work_col_num; ++i) {
    auto cmb_coef = cmb[i];
    if (cmb_coef == kIdCoefRepr) {
      for (size_t j = 0; j < m.rows; ++j) {
        res[j] += m(j, i);
      }
    } else if (cmb_coef == kNullCoefRepr) {
      // Do nothing.
    } else {
      for (size_t j = 0; j < m.rows; ++j) {
        auto elem = m(j, i);
        for (auto &[sub_op, coef_repr] : elem.op_label_coef_repr_map_) {
          assert(coef_repr == kIdCoefRepr); //todo : support other case
          coef_repr = cmb_coef;
        }
        res[j] += elem;
      }
    }
  }
  return res;
}

inline void SparOpReprMatRowDelinearize(
    SparOpReprMat &target, SparOpReprMat &follower) {
  auto row_num = target.rows;
  SparCoefReprMat trans_mat(row_num, row_num);
  trans_mat.Reserve(row_num * row_num);
  size_t deleted_row_size = 0;
  trans_mat.SetElem(0, 0, kIdCoefRepr);

  //Find linearlization rows,  remove the rows, and construct transition matrix
  for (size_t i = 1; i < row_num; ++i) {
    auto cmb = target.CalcRowLinCmb(i);
    if (CalcSparOpReprMatRowLinCmb(target, cmb) == target.GetRow(i)) {
      target.RemoveRow(i);
      row_num = row_num - 1;
      trans_mat.RemoveCol(trans_mat.cols - 1);//remove the last cols
      for (size_t j = 0; j < i; ++j) {
        trans_mat.SetElem(i + deleted_row_size, j, cmb[j]);
      }
      i--;
      deleted_row_size++;
    } else {
      trans_mat.SetElem(i + deleted_row_size, i, kIdCoefRepr);
    }
  }
  // Calculate new follower.
  if (trans_mat.cols < trans_mat.rows) {
    follower = SparOpReprMatSparCoefReprMatIncompleteMulti(
        follower, trans_mat);
  }
}

/**
 * @todo optimize
 */
inline void SparOpReprMatColDelinearize(
    SparOpReprMat &target, SparOpReprMat &follower) {
  auto col_num = target.cols;
  SparCoefReprMat trans_mat(col_num, col_num);
  trans_mat.Reserve(col_num * col_num);
  size_t deleted_col_size = 0;
  trans_mat.SetElem(0, 0, kIdCoefRepr);
  //Find linearlization cols,  remove the cols, and construct transition matrix
  for (size_t i = 1; i < col_num; ++i) {
    auto cmb = target.CalcColLinCmb(i);
    if (CalcSparOpReprMatColLinCmb(target, cmb) == target.GetCol(i)) {
      // Remove the col.
      target.RemoveCol(i);
      col_num = col_num - 1;
      trans_mat.RemoveRow(trans_mat.rows - 1);//remove the last row
      for (size_t j = 0; j < i; ++j) {
        trans_mat.SetElem(j, i + deleted_col_size, cmb[j]);
      }
      i--;
      deleted_col_size++;
    } else {
      trans_mat.SetElem(i, i + deleted_col_size, kIdCoefRepr);
    }
  }
  // Calculate new follower.
  if (trans_mat.rows < trans_mat.cols) {
    follower = SparCoefReprMatSparOpReprMatIncompleteMulti(
        trans_mat, follower);
  }
}

// Row and column compresser.
inline void SparOpReprMatRowCompresser(
    SparOpReprMat &target, SparOpReprMat &follower) {
  assert(target.rows == follower.cols);
  auto row_num = target.rows;
  if (row_num == 1) { return; }
  // Sort rows of target and transpose cols of follower.
  auto sorted_row_idxs = target.SortRows();
  follower.TransposeCols(sorted_row_idxs);
  // Separate row coefficients of target.
  bool need_separate_row_coef = false;
  SparCoefReprMat row_coef_trans_mat(row_num, row_num);
  for (size_t row_idx = 0; row_idx < row_num; ++row_idx) {
    auto row_coef = target.TryCatchRowCommonDivisorCoef(row_idx);
    if (row_coef != kNullCoefRepr) {
      row_coef_trans_mat.SetElem(row_idx, row_idx, row_coef);
    } else {
      row_coef_trans_mat.SetElem(row_idx, row_idx, kIdCoefRepr);
    }
    if ((row_coef != kNullCoefRepr) && (row_coef != kIdCoefRepr)) {
      need_separate_row_coef = true;
      target.RemoveRowCoef(row_idx);
    }
  }
  if (need_separate_row_coef) {
    follower = SparOpReprMatSparCoefReprMatIncompleteMulti(
        follower, row_coef_trans_mat);
  }
  // Delinearize rows of target.
  SparOpReprMatRowDelinearize(target, follower);
}

/**
 * Compress Sparse Operator Representation Matrix by Column Delinearlization
 * This function is the performance hot spot of MPO generation.
 *
 * @param target    The target mpo's matrix representation
 * @param follower  The next site mpo's matrix representation
 *
 */
inline void SparOpReprMatColCompresser(
    SparOpReprMat &target, SparOpReprMat &follower) {
  assert(target.cols == follower.rows);
  auto col_num = target.cols;
  if (col_num == 1) { return; }
  // Sort cols of target and transpose rows of follower.
  auto sorted_col_idxs = target.SortCols();
  follower.TransposeRows(sorted_col_idxs);
  // Separate col coefficients of target.
  bool need_separate_col_coef = false;
  SparCoefReprMat col_coef_trans_mat(col_num, col_num);
  for (size_t col_idx = 0; col_idx < col_num; ++col_idx) {
    auto col_coef = target.TryCatchColCommonDivisorCoef(col_idx);
    if (col_coef != kNullCoefRepr) {
      col_coef_trans_mat.SetElem(col_idx, col_idx, col_coef);
    } else {
      col_coef_trans_mat.SetElem(col_idx, col_idx, kIdCoefRepr);
    }
    if ((col_coef != kNullCoefRepr) && (col_coef != kIdCoefRepr)) {
      need_separate_col_coef = true;
      target.RemoveColCoef(col_idx);  //should be division.
      // Here removing coefficients is work because the realization of TryCatchColCommonDivisorCoef is very crude.
    }
  }
  if (need_separate_col_coef) {
    follower = SparCoefReprMatSparOpReprMatIncompleteMulti(
        col_coef_trans_mat, follower);
  }
  // Delinearize cols of target.
  SparOpReprMatColDelinearize(target, follower);
}

// Helpers.
template<typename T>
bool ElemInVec(const T &e, const std::vector<T> &v, long &pos) {
  for (size_t i = 0; i < v.size(); ++i) {
    if (e == v[i]) {
      pos = i;
      return true;
    }
  }
  pos = -1;
  return false;
}

template<typename VecT>
VecT ConcatenateTwoVec(const VecT &va, const VecT &vb) {
  VecT res;
  res.reserve(va.size() + vb.size());
  res.insert(res.end(), va.begin(), va.end());
  res.insert(res.end(), vb.begin(), vb.end());
  return res;
}

#endif /* ifndef QLMPS_ONE_DIM_TN_MPO_MPOGEN_COEF_OP_ALG_H */
