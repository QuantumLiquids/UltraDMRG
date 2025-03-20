// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2019-09-27 17:38
*
* Description: QuantumLiquids/UltraDMRG project. Implantation details for MPO generator.
*/
#include "qlmps/consts.h"     // kNullUintVec, kNullUintVecVec
#include "qlmps/one_dim_tn/mpo/mpogen/mpogen.h"
#include "qlmps/one_dim_tn/mpo/mpogen/symb_alg/coef_op_alg.h"
#include "qlten/qlten.h"
#include "qlmps/utilities.h"           // Real

#include <iostream>
#include <iomanip>
#include <algorithm>    // is_sorted
#include <map>

#include <assert.h>     // assert

#ifdef Release
#define NDEBUG
#endif

namespace qlmps {
using namespace qlten;

// Forward declarations.
template<typename TenT>
void AddOpToHeadMpoTen(TenT *, const TenT &, const size_t);

template<typename TenT>
void AddOpToTailMpoTen(TenT *, const TenT &, const size_t);

template<typename TenT>
void AddOpToCentMpoTen(TenT *, const TenT &, const size_t, const size_t);

template<typename TenElemT, typename QNT>
std::vector<OpRepr> GenIdOpReprs(const SiteVec<TenElemT, QNT> &site_vec,
                                 LabelConvertor<QLTensor<TenElemT, QNT>> &op_label_convertor) {
  std::vector<OpRepr> id_op_reprs;
  id_op_reprs.reserve(site_vec.size);
  for (const auto &id_op : site_vec.id_ops) {
    id_op_reprs.push_back(OpRepr(op_label_convertor.Convert(id_op)));
  }
  return id_op_reprs;
}

template<typename TenElemT, typename QNT>
MPOGenerator<TenElemT, QNT>::MPOGenerator(const SiteVec<TenElemT, QNT> &site_vec)
    :MPOGenerator(site_vec, site_vec.sites[0].GetQNSct(0).GetQn() - site_vec.sites[0].GetQNSct(0).GetQn()) {}
/**
Create a MPO generator. Create a MPO generator using the sites of the system
which is described by a SiteVec.

@param site_vec The local Hilbert spaces of each sites of the system.
@param zero_div The zero value of the given quantum number type which is used
       to set the divergence of the MPO.

@since version 0.2.0
*/
template<typename TenElemT, typename QNT>
MPOGenerator<TenElemT, QNT>::MPOGenerator(
    const SiteVec<TenElemT, QNT> &site_vec,
    const QNT &zero_div
) : N_(site_vec.size),
    site_vec_(site_vec),
    zero_div_(zero_div),
    id_op_vector_(site_vec.id_ops),
    coef_label_convertor_(TenElemT(1)),
    op_label_convertor_(),  //note here the construction order is quite tricky
    fsm_(site_vec.size, GenIdOpReprs(site_vec, op_label_convertor_)) {
  pb_out_vector_.reserve(N_);
  pb_in_vector_.reserve(N_);
  for (size_t i = 0; i < N_; ++i) {
    pb_out_vector_.emplace_back(site_vec.sites[i]);
    pb_in_vector_.emplace_back(InverseIndex(site_vec.sites[i]));
  }
}

/**
The most generic API for adding a many-body term to the MPO generator. Notice
that the indexes of the operators have to be ascending sorted.

@param coef The coefficient of the term.
@param local_ops All the local (on-site) operators in the term.
@param local_ops_idxs The site indexes of these local operators.

@since version 0.2.0
*/
template<typename TenElemT, typename QNT>
void MPOGenerator<TenElemT, QNT>::AddTerm(
    TenElemT coef,
    QLTensorVec local_ops,
    std::vector<size_t> local_ops_idxs
) {
  assert(local_ops.size() == local_ops_idxs.size());
  if (coef == TenElemT(0)) { return; }
  if (qlten::abs(coef) < kDoubleEpsilon) {
    std::cout << "warning: too small hamiltonian coefficient. Neglect the term." << coef << std::endl;
    return;
  }   // If coef is zero, do nothing.
#ifndef NDEBUG
  for (size_t i = 0; i < local_ops.size(); i++) {
    const auto &op = local_ops[i];
    const size_t site = local_ops_idxs[i];
    assert(site_vec_.sites[site] == op.GetIndex(1));
  }
#endif
  // sort the site idxs
  if (!std::is_sorted(local_ops_idxs.cbegin(), local_ops_idxs.cend())) {
    std::cout << "sort operators and sites according the site order. " << std::endl;
    std::vector<size_t> order_indices(local_ops.size());
    std::iota(order_indices.begin(), order_indices.end(), 0);
    std::sort(order_indices.begin(), order_indices.end(),
              [&](size_t i, size_t j) -> bool {
                return local_ops_idxs[i] < local_ops_idxs[j];
              });
    QLTensorVec sorted_local_ops(local_ops.size());
    std::transform(order_indices.begin(), order_indices.end(), sorted_local_ops.begin(),
                   [&](std::size_t i) { return local_ops[i]; });
    local_ops = sorted_local_ops;
    std::sort(local_ops_idxs.begin(), local_ops_idxs.end());
  }
  assert(local_ops_idxs.back() < N_);

  //< Input Operator should remove some linear relationship. Here we remove the sign.
  for (size_t i = 0; i < local_ops.size(); i++) {
    auto &op = local_ops[i];
    const TenElemT first_data = op.GetFirstNonZeroElement();
    if (Real(first_data) < 0.0) {
      coef *= (-1);
      op *= (-1);
    }
  }

  auto coef_label = coef_label_convertor_.Convert(coef);
  auto ntrvl_ops_idxs_head = local_ops_idxs.front();
  auto ntrvl_ops_idxs_tail = local_ops_idxs.back();
  OpReprVec ntrvl_ops_reprs;
  for (size_t i = ntrvl_ops_idxs_head; i <= ntrvl_ops_idxs_tail; ++i) {
    auto poss_it = std::find(local_ops_idxs.cbegin(), local_ops_idxs.cend(), i);
    if (poss_it != local_ops_idxs.cend()) {     // Nontrivial operator
      auto local_op_loc =
          poss_it - local_ops_idxs.cbegin();    // Location of the local operator in the local operators list.
      auto op_label = op_label_convertor_.Convert(local_ops[local_op_loc]);
      if (local_op_loc == 0) {
        ntrvl_ops_reprs.push_back(OpRepr(coef_label, op_label));
      } else {
        ntrvl_ops_reprs.push_back(OpRepr(op_label));
      }
    } else {
      auto op_label = op_label_convertor_.Convert(id_op_vector_[i]);
      ntrvl_ops_reprs.push_back(OpRepr(op_label));
    }
  }
  assert(
      ntrvl_ops_reprs.size() == (ntrvl_ops_idxs_tail - ntrvl_ops_idxs_head + 1)
  );

  fsm_.AddPath(ntrvl_ops_idxs_head, ntrvl_ops_idxs_tail, ntrvl_ops_reprs);
}

/**
Add a many-body term defined by physical operators and insertion operators to
the MPO generator. The indexes of the operators have to be ascending sorted.

@param coef The coefficient of the term.
@param phys_ops Operators with physical meaning in this term. Like
       \f$c^{\dagger}\f$ operator in the \f$-t c^{\dagger}_{i} c_{j}\f$
       hopping term. Its size must be larger than 1.
@param phys_ops_idxs The corresponding site indexes of the physical operators.
@param inst_ops Operators which will be inserted between physical operators and
       also behind the last physical operator as a tail string. For example the
       Jordan-Wigner string operator.
@param inst_ops_idxs_set Each element defines the explicit site indexes of the
       corresponding inserting operator. If it is set to empty (default value),
       every site between the corresponding physical operators will be inserted
       a same insertion operator.

@since version 0.2.0
*/
template<typename TenElemT, typename QNT>
void MPOGenerator<TenElemT, QNT>::AddTerm(
    const TenElemT coef,
    const QLTensorVec &phys_ops,
    const std::vector<size_t> &phys_ops_idxs,
    const QLTensorVec &inst_ops,
    const std::vector<std::vector<size_t>> &inst_ops_idxs_set
) {
  assert(phys_ops.size() >= 2);
  assert(phys_ops.size() == phys_ops_idxs.size());
  assert(
      (inst_ops.size() == phys_ops.size() - 1) ||
          (inst_ops.size() == phys_ops.size())
  );
  if (inst_ops_idxs_set != kNullUintVecVec) {
    assert(inst_ops_idxs_set.size() == inst_ops.size());
  }

  QLTensorVec local_ops;
  std::vector<size_t> local_ops_idxs;
  for (size_t i = 0; i < phys_ops.size() - 1; ++i) {
    local_ops.push_back(phys_ops[i]);
    local_ops_idxs.push_back(phys_ops_idxs[i]);
    if (inst_ops_idxs_set == kNullUintVecVec) {
      for (size_t j = phys_ops_idxs[i] + 1; j < phys_ops_idxs[i + 1]; ++j) {
        local_ops.push_back(inst_ops[i]);
        local_ops_idxs.push_back(j);
      }
    } else {
      for (auto inst_op_idx : inst_ops_idxs_set[i]) {
        local_ops.push_back(inst_ops[i]);
        local_ops_idxs.push_back(inst_op_idx);
      }
    }
  }
  // Deal with the last physical operator and possible insertion operator tail
  // string.
  local_ops.push_back(phys_ops.back());
  local_ops_idxs.push_back(phys_ops_idxs.back());
  if (inst_ops.size() == phys_ops.size()) {
    if (inst_ops_idxs_set == kNullUintVecVec) {
      for (size_t j = phys_ops_idxs.back(); j < N_; ++j) {
        local_ops.push_back(inst_ops.back());
        local_ops_idxs.push_back(j);
      }
    } else {
      for (auto inst_op_idx : inst_ops_idxs_set.back()) {
        local_ops.push_back(inst_ops.back());
        local_ops_idxs.push_back(inst_op_idx);
      }
    }
  }

  AddTerm(coef, local_ops, local_ops_idxs);
}

/**
Add one-body or two-body interaction term.

@param coef The coefficient of the term.
@param op1 The first physical operator for the term.
@param op1_idx The site index of the first physical operator.
@param op2 The second physical operator for the term.
@param op2_idx The site index of the second physical operator.
@param inst_op The insertion operator for the two-body interaction term.
@param inst_op_idxs The explicit site indexes of the insertion operator.

@since version 0.2.0

@note for the insertion included case (fermion), the op1_idx and op2_idx should be ordered.
*/
template<typename TenElemT, typename QNT>
void MPOGenerator<TenElemT, QNT>::AddTerm(
    const TenElemT coef,
    const QLTensorT &op1,
    const size_t op1_idx,
    const QLTensorT &op2,
    const size_t op2_idx,
    const QLTensorT &inst_op,
    const std::vector<size_t> &inst_op_idxs
) {
  if (op2 == QLTensorT()) {     // One-body interaction term
    QLTensorVec local_ops = {op1};
    std::vector<size_t> local_ops_idxs = {op1_idx};
    AddTerm(coef, local_ops, local_ops_idxs);     // Use the most generic API
  } else {                      // Two-body interaction term
    assert(op2_idx != 0);
    if (inst_op == QLTensorT()) {     // Trivial insertion operator
      AddTerm(coef, {op1, op2}, {op1_idx, op2_idx});
    } else {                          // Non-trivial insertion operator
      if (inst_op_idxs == kNullUintVec) {    // Uniform insertion
        assert(op1_idx < op2_idx);
        AddTerm(coef, {op1, op2}, {op1_idx, op2_idx}, {inst_op});
      } else {                              // Non-uniform insertion
        AddTerm(
            coef,
            {op1, op2}, {op1_idx, op2_idx},
            {inst_op}, {inst_op_idxs}
        );
      }
    }
  }
}

template<typename TenElemT, typename QNT>
MPO<typename MPOGenerator<TenElemT, QNT>::QLTensorT>
MPOGenerator<TenElemT, QNT>::Gen(const bool compress, const bool show_matrix) {
  SparOpReprMatVec fsm_mat_repr;
  if (compress) { fsm_mat_repr = fsm_.GenCompressedMatRepr(show_matrix); }
  else { fsm_mat_repr = fsm_.GenMatRepr(); }

  auto label_coef_mapping = coef_label_convertor_.GetLabelObjMapping();
  auto label_op_mapping = op_label_convertor_.GetLabelObjMapping();

  // Print MPO tensors virtual bond dimension.
  std::cout << "[";
  for (auto &mpo_ten_repr : fsm_mat_repr) {
    std::cout << "\t" << mpo_ten_repr.cols;
  }
  std::cout << "]" << std::endl;

  MPO<QLTensorT> mpo(N_);
  IndexT trans_vb({QNSctT(zero_div_, 1)}, OUT);
  std::vector<size_t> transposed_idxs;
  for (size_t i = 0; i < N_; ++i) {
    if (i == 0) {
      transposed_idxs = SortSparOpReprMatColsByQN_(
          fsm_mat_repr[i], trans_vb, label_op_mapping);
      mpo[i] = HeadMpoTenRepr2MpoTen_(
          fsm_mat_repr[i], trans_vb,
          label_coef_mapping, label_op_mapping);
    } else if (i == N_ - 1) {
      fsm_mat_repr[i].TransposeRows(transposed_idxs);
      auto lvb = InverseIndex(trans_vb);
      mpo[i] = TailMpoTenRepr2MpoTen_(
          fsm_mat_repr[i], lvb,
          label_coef_mapping, label_op_mapping);
    } else {
      fsm_mat_repr[i].TransposeRows(transposed_idxs);
      auto lvb = InverseIndex(trans_vb);
      transposed_idxs = SortSparOpReprMatColsByQN_(
          fsm_mat_repr[i], trans_vb, label_op_mapping);
      mpo[i] = CentMpoTenRepr2MpoTen_(
          fsm_mat_repr[i], lvb, trans_vb,
          label_coef_mapping, label_op_mapping, i);
    }
  }
  return mpo;
}

template<typename TenElemT, typename QNT>
MatReprMPO<typename MPOGenerator<TenElemT, QNT>::QLTensorT>
MPOGenerator<TenElemT, QNT>::GenMatReprMPO(const bool show_matrix) {
  using QLTensorT = typename MPOGenerator<TenElemT, QNT>::QLTensorT;
  auto fsm_comp_mat_repr = fsm_.GenCompressedMatRepr(show_matrix);
  auto label_coef_mapping = coef_label_convertor_.GetLabelObjMapping();
  auto label_op_mapping = op_label_convertor_.GetLabelObjMapping();
  // Print MPO tensors virtual bond dimension.
  std::cout << "[";
  for (auto &mpo_ten_repr : fsm_comp_mat_repr) {
    std::cout << "\t" << mpo_ten_repr.cols;
  }
  std::cout << "]" << std::endl;

  MatReprMPO<QLTensorT> mat_repr_mpo(N_);
  for (size_t i = 0; i < N_; ++i) {
    mat_repr_mpo[i] = SparMat<QLTensorT>(fsm_comp_mat_repr[i].rows, fsm_comp_mat_repr[i].cols);
    for (size_t x = 0; x < fsm_comp_mat_repr[i].rows; ++x) {
      for (size_t y = 0; y < fsm_comp_mat_repr[i].cols; ++y) {
        auto elem = fsm_comp_mat_repr[i](x, y);
        if (elem != kNullOpRepr) {
          mat_repr_mpo[i].SetElem(x, y, elem.Realize(label_coef_mapping, label_op_mapping));
        }
      }
    }
  }
  return mat_repr_mpo;
}

/**
 * Calculate target right virtual bond quantum number
 *
 * @param x         coordinate of row of the operator in the sparse matrix representation of operators
 * @param y         useless in this function
 * @param op_repr   operator representation
 * @param label_op_mapping  to get the operator in QLTensor form from operator representation
 * @param trans_vb  the right virtual bond of last mpo tensor
 * @return          the quantum number on the right virtual bond to embedding the operator
 */
template<typename TenElemT, typename QNT>
QNT MPOGenerator<TenElemT, QNT>::CalcTgtRvbQN_(
    const size_t x, const size_t y, const OpRepr &op_repr,
    const QLTensorVec &label_op_mapping, const IndexT &trans_vb
) {
  auto lvb = InverseIndex(trans_vb);
  auto lvb_qn = lvb.GetQNSctFromActualCoor(x).GetQn();
  auto op0_in_op_repr = label_op_mapping[op_repr.GetOpLabelList()[0]];
  return lvb_qn - Div(op0_in_op_repr);
}

/**
 * Sort Sparse Operator Representation Matrix Columns by Quantum Number
 *
 * @param op_repr_mat
 * @param trans_vb  input: left virtual bond of the mpo tensor; output: right virtual bond of the mpo tensor.
 * @param label_op_mapping
 * @return  How the sparse matrix of operator should be transpose, and the matrix has been transposed in the subroutine.
 */
template<typename TenElemT, typename QNT>
std::vector<size_t> MPOGenerator<TenElemT, QNT>::SortSparOpReprMatColsByQN_(
    SparOpReprMat &op_repr_mat, IndexT &trans_vb,
    const QLTensorVec &label_op_mapping) {
  std::vector<std::pair<QNT, size_t>> rvb_qn_dim_pairs;
  std::vector<size_t> transposed_idxs;
  for (size_t col = 0; col < op_repr_mat.cols; ++col) {
    bool has_ntrvl_op = false;
    QNT col_rvb_qn(zero_div_);
    for (size_t row = 0; row < op_repr_mat.rows; ++row) {
      auto elem = op_repr_mat(row, col);
      if (elem != kNullOpRepr) {
        auto rvb_qn = CalcTgtRvbQN_(
            row, col, elem, label_op_mapping, trans_vb
        );
        if (!has_ntrvl_op) {
          col_rvb_qn = rvb_qn;
          has_ntrvl_op = true;
          bool has_qn = false;
          size_t offset = 0;
          for (auto &qn_dim : rvb_qn_dim_pairs) {
            if (qn_dim.first == rvb_qn) {
              qn_dim.second += 1;
              auto beg_it = transposed_idxs.begin();
              transposed_idxs.insert(beg_it + offset, col);
              has_qn = true;
              break;
            } else {
              offset += qn_dim.second;
            }
          }
          if (!has_qn) {
            rvb_qn_dim_pairs.push_back(std::make_pair(rvb_qn, 1));
            auto beg_it = transposed_idxs.begin();
            transposed_idxs.insert(beg_it + offset, col);
          }
        } else {
          assert(rvb_qn == col_rvb_qn);
        }
      }
    }
  }
  op_repr_mat.TransposeCols(transposed_idxs);
  std::vector<QNSctT> rvb_qnscts;
  rvb_qnscts.reserve(rvb_qn_dim_pairs.size());
  for (auto &qn_dim : rvb_qn_dim_pairs) {
    rvb_qnscts.push_back(QNSctT(qn_dim.first, qn_dim.second));
  }
  trans_vb = IndexT(rvb_qnscts, OUT);
  return transposed_idxs;
}

template<typename TenElemT, typename QNT>
typename MPOGenerator<TenElemT, QNT>::QLTensorT
MPOGenerator<TenElemT, QNT>::HeadMpoTenRepr2MpoTen_(
    const SparOpReprMat &op_repr_mat,
    const IndexT &rvb,
    const TenElemVec &label_coef_mapping, const QLTensorVec &label_op_mapping
) {
  QNT qn_eg = rvb.GetQNSct(0).GetQn();
  QNT qn0 = qn_eg - qn_eg;
  IndexT lvb = IndexT({QNSctT(qn0, 1)}, TenIndexDirType::IN);
  auto mpo_ten = QLTensorT({lvb, pb_in_vector_.front(), pb_out_vector_.front(), rvb});
  for (size_t y = 0; y < op_repr_mat.cols; ++y) {
    auto elem = op_repr_mat(0, y);
    if (elem != kNullOpRepr) {
      auto op = elem.Realize(label_coef_mapping, label_op_mapping);
      AddOpToHeadMpoTen(&mpo_ten, op, y);
    }
  }
  return mpo_ten;
}

template<typename TenElemT, typename QNT>
typename MPOGenerator<TenElemT, QNT>::QLTensorT
MPOGenerator<TenElemT, QNT>::TailMpoTenRepr2MpoTen_(
    const SparOpReprMat &op_repr_mat,
    const IndexT &lvb,
    const TenElemVec &label_coef_mapping, const QLTensorVec &label_op_mapping) {
  QNT qn_eg = lvb.GetQNSct(0).GetQn();
  QNT qn0 = qn_eg - qn_eg;
  IndexT rvb = IndexT({QNSctT(qn0, 1)}, TenIndexDirType::OUT);
  auto mpo_ten = QLTensorT({lvb, pb_in_vector_.back(), pb_out_vector_.back(), rvb});
  for (size_t x = 0; x < op_repr_mat.rows; ++x) {
    auto elem = op_repr_mat(x, 0);
    if (elem != kNullOpRepr) {
      auto op = elem.Realize(label_coef_mapping, label_op_mapping);
      AddOpToTailMpoTen(&mpo_ten, op, x);
    }
  }
  return mpo_ten;
}

template<typename TenElemT, typename QNT>
typename MPOGenerator<TenElemT, QNT>::QLTensorT
MPOGenerator<TenElemT, QNT>::CentMpoTenRepr2MpoTen_(
    const SparOpReprMat &op_repr_mat,
    const IndexT &lvb,
    const IndexT &rvb,
    const TenElemVec &label_coef_mapping, const QLTensorVec &label_op_mapping,
    const size_t site
) {
  auto mpo_ten = QLTensorT({lvb, pb_in_vector_[site], pb_out_vector_[site], rvb});
  for (size_t x = 0; x < op_repr_mat.rows; ++x) {
    for (size_t y = 0; y < op_repr_mat.cols; ++y) {
      auto elem = op_repr_mat(x, y);
      if (elem != kNullOpRepr) {
        auto op = elem.Realize(label_coef_mapping, label_op_mapping);
        AddOpToCentMpoTen(&mpo_ten, op, x, y);
      }
    }
  }
  return mpo_ten;
}

template<typename TenT>
void AddOpToHeadMpoTen(TenT *pmpo_ten, const TenT &rop, const size_t rvb_coor) {
  for (size_t bpb_coor = 0; bpb_coor < rop.GetIndexes()[0].dim(); ++bpb_coor) {
    for (size_t tpb_coor = 0; tpb_coor < rop.GetIndexes()[1].dim(); ++tpb_coor) {
      auto elem = rop.GetElem({bpb_coor, tpb_coor});
      if (elem != 0.0) {
        (*pmpo_ten)(0, bpb_coor, tpb_coor, rvb_coor) = elem;
      }
    }
  }
}

template<typename TenT>
void AddOpToTailMpoTen(TenT *pmpo_ten, const TenT &rop, const size_t lvb_coor) {
  for (size_t bpb_coor = 0; bpb_coor < rop.GetIndexes()[0].dim(); ++bpb_coor) {
    for (size_t tpb_coor = 0; tpb_coor < rop.GetIndexes()[1].dim(); ++tpb_coor) {
      auto elem = rop.GetElem({bpb_coor, tpb_coor});
      if (elem != 0.0) {
        (*pmpo_ten)(lvb_coor, bpb_coor, tpb_coor, 0) = elem;
      }
    }
  }
}

template<typename TenT>
void AddOpToCentMpoTen(
    TenT *pmpo_ten, const TenT &rop,
    const size_t lvb_coor, const size_t rvb_coor
) {
  for (size_t bpb_coor = 0; bpb_coor < rop.GetIndexes()[0].dim(); ++bpb_coor) {
    for (size_t tpb_coor = 0; tpb_coor < rop.GetIndexes()[1].dim(); ++tpb_coor) {
      auto elem = rop.GetElem({bpb_coor, tpb_coor});
      if (elem != 0.0) {
        (*pmpo_ten)(lvb_coor, bpb_coor, tpb_coor, rvb_coor) = elem;
      }
    }
  }
}
} /* qlmps */
