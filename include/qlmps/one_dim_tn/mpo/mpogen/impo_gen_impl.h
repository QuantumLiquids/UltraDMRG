// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Hao-Xin Wang <wanghaoxin1996@gmail.com>
* Creation Date: 2024-02-05
*
* Description: QuantumLiquids/UltraDMRG project. infinite MPO Generator, implementation.
*/

#ifndef QLMPS_ONE_DIM_TN_MPO_MPOGEN_IMPOGEN_IMPL_H
#define QLMPS_ONE_DIM_TN_MPO_MPOGEN_IMPOGEN_IMPL_H

namespace qlmps {
using namespace qlten;

/// Constructor
template<typename TenElemT, typename QNT>
inline iMPOGenerator<TenElemT, QNT>::iMPOGenerator(
    const SiteVec<TenElemT, QNT> &site_vec) :
    unit_cell_size_(site_vec.size()),
    site_vec_(site_vec),
    id_op_vector_(site_vec.id_ops),
    coef_label_convertor_(TenElemT(1)),
    op_label_convertor_(),
    winding_num_max_(0),
    ifsm_(site_vec.size(), GenIdOpReprs(site_vec, op_label_convertor_)) {}

template<typename TenElemT, typename QNT>
inline void iMPOGenerator<TenElemT, QNT>::AddTerm(const TenElemT coef,
                                           const QLTensorVec &phys_ops,
                                           const std::vector<size_t> &phys_ops_sites,
                                           const QLTensorVec &inst_ops,
                                           const std::vector<std::vector<size_t>> &inst_ops_sites_set) {
  assert(phys_ops.size() >= 2);
  assert(phys_ops.size() == phys_ops_sites.size());
  assert(
      (inst_ops.size() == phys_ops.size() - 1) ||
          (inst_ops.size() == phys_ops.size())
  );
  assert(inst_ops_sites_set.size() == inst_ops.size());

  QLTensorVec local_ops;
  std::vector<size_t> local_ops_sites;
  for (size_t i = 0; i < phys_ops.size() - 1; ++i) {
    local_ops.push_back(phys_ops[i]);
    local_ops_sites.push_back(phys_ops_sites[i]);
    if (inst_ops_sites_set == kNullUintVecVec) { // what's this?
      for (size_t j = phys_ops_sites[i] + 1; j < phys_ops_sites[i + 1]; ++j) {
        local_ops.push_back(inst_ops[i]);
        local_ops_sites.push_back(j);
      }
    } else {
      for (auto inst_op_idx : inst_ops_sites_set[i]) {
        local_ops.push_back(inst_ops[i]);
        local_ops_sites.push_back(inst_op_idx);
      }
    }
  }

  // Deal with the last physical operator and possible insertion operator tail
  // string.
  local_ops.push_back(phys_ops.back());
  local_ops_sites.push_back(phys_ops_sites.back());
  if (inst_ops.size() == phys_ops.size()) {
    if (inst_ops_sites_set == kNullUintVecVec) {
      for (size_t j = phys_ops_sites.back(); j < unit_cell_size_; ++j) {
        local_ops.push_back(inst_ops.back());
        local_ops_sites.push_back(j);
      }
    } else {
      for (auto inst_op_idx : inst_ops_sites_set.back()) {
        local_ops.push_back(inst_ops.back());
        local_ops_sites.push_back(inst_op_idx);
      }
    }
  }

  AddTerm(coef, local_ops, local_ops_sites);
}

template<typename TenElemT, typename QNT>
inline MPO<QLTensor<TenElemT, QNT>> iMPOGenerator<TenElemT, QNT>::Gen() {

}

template<typename TenElemT, typename QNT>
inline void iMPOGenerator<TenElemT, QNT>::EvaluateMatReprQN_(const SparOpReprMatVec &mat_repr_sym_mpo) {
  QNT some_qn = site_vec_.sites[0].GetQNSct(0).GetQn();
  QNT zero_qn = some_qn + (-some_qn);
  auto label_op_mapping = op_label_convertor_.GetLabelObjMapping();
  mat_rep_qns_.reserve(unit_cell_size_);
  mat_rep_qns_[0].first[0] = zero_qn;
  std::vector<size_t> valid_rows({0});
  for (size_t pesudo_site = 0; pesudo_site < (winding_num_max_ + 1) * unit_cell_size_; pesudo_site++) {
    const size_t site = pesudo_site % unit_cell_size_;
    std::vector<size_t> valid_cols;
    const SparOpReprMat &sym_op_mat = mat_repr_sym_mpo[site];
    for (size_t row : valid_rows) {
      auto lvb_qn = mat_rep_qns_[site].first[row];
      for (size_t col = 0; col < sym_op_mat.cols; col++) {
        if (sym_op_mat.IsNull(row, col)) {
          continue;
        }
        const OpRepr &op = sym_op_mat(row, col);
        Tensor op0 = label_op_mapping[op.GetOpLabelList()[0]];
        auto rvb_qn = lvb_qn - Div(op0);
        if (mat_rep_qns_[site].second[col].has_value()) {
          assert(mat_rep_qns_[site].second[col] == rvb_qn);
        } else {
          mat_rep_qns_[site].second[col] = rvb_qn;
        }
        valid_cols.push_back(col);
      }
    }

    //prepare for next site
    valid_rows = valid_cols;
    mat_rep_qns_[(site + 1) % unit_cell_size_].first = mat_rep_qns_[site].second;
  }
}

// Helper
template<typename T>
inline std::vector<size_t> ReorderVecToSameElemNearBy(std::vector<T> &data) {
  std::unordered_map<T, std::vector<size_t>> element_indices;

  // Collect indices for each unique element
  for (size_t i = 0; i < data.size(); ++i)
    element_indices[data[i]].push_back(i);

  std::vector<T> sorted_data;
  std::vector<size_t> indices;

  // Reorder the data and collect the indices
  for (const auto &pair : element_indices) {
    sorted_data.insert(sorted_data.end(), pair.second.size(), pair.first);
    indices.insert(indices.end(), pair.second.begin(), pair.second.end());
  }

  data = sorted_data;  // Assign the sorted data back to the original vector

  return indices;
}

template<typename TenElemT, typename QNT>
inline std::vector<size_t> iMPOGenerator<TenElemT, QNT>::SortSymOpMatColsByQN_(SparOpReprMat &sym_op_mat,
                                                                        size_t site) {
  MatrixColQNT &col_qns = mat_rep_qns_[site].second;
  for (auto &qn : col_qns) {
    assert(qn.has_value());
  }
  std::vector<size_t> transposed_idxs = ReorderVecToSameElemNearBy(col_qns);
  sym_op_mat.TransposeCols(transposed_idxs);
  return transposed_idxs;
}

template<typename TenElemT, typename QNT>
inline OpReprVec iMPOGenerator<TenElemT, QNT>::TransferHamiltonianTermToConsecutiveOpRepr_(
    TenElemT coef,
    QLTensorVec local_ops,
    std::vector<size_t> local_ops_sites
) {
  assert(local_ops_sites.size() == local_ops.size());
  assert(std::is_sorted(local_ops_sites.cbegin(), local_ops_sites.cend()));
  auto coef_label = coef_label_convertor_.Convert(coef);
  auto ntrvl_ops_idxs_head = local_ops_sites.front();
  auto ntrvl_ops_idxs_tail = local_ops_sites.back();
  OpReprVec ntrvl_ops_reprs;
  for (size_t i = ntrvl_ops_idxs_head; i <= ntrvl_ops_idxs_tail; ++i) {
    auto poss_it = std::find(local_ops_sites.cbegin(), local_ops_sites.cend(), i);
    if (poss_it != local_ops_sites.cend()) {     // Nontrivial operator
      auto local_op_loc =
          poss_it - local_ops_sites.cbegin();    // Location of the local operator in the local operators list.
      auto op_label = op_label_convertor_.Convert(local_ops[local_op_loc]);
      if (i == ntrvl_ops_idxs_tail) { //coefficient add to last site
        ntrvl_ops_reprs.push_back(OpRepr(coef_label, op_label));
      } else {
        ntrvl_ops_reprs.push_back(OpRepr(op_label));
      }
    } else {
      auto id_op_label = op_label_convertor_.Convert(id_op_vector_[i]);
      ntrvl_ops_reprs.push_back(OpRepr(id_op_label));
    }
  }
  return ntrvl_ops_reprs;
}

}/* qlmps */

#endif //QLMPS_ONE_DIM_TN_MPO_MPOGEN_IMPOGEN_IMPL_H
