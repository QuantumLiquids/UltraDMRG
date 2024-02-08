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

template<typename TenElemT, typename QNT>
iMPOGenerator<TenElemT, QNT>::iMPOGenerator(
    const SiteVec<TenElemT, QNT> &site_vec) :
    unit_cell_size_(site_vec.size()),
    site_vec_(site_vec),
    id_op_vector_(site_vec.id_ops),
    coef_label_convertor_(TenElemT(1)),
    op_label_convertor_(),
    winding_num_max_(0),
    ifsm_(site_vec.size(), GenIdOpReprs(site_vec, op_label_convertor_)) {}

template<typename TenElemT, typename QNT>
void iMPOGenerator<TenElemT, QNT>::EvaluateMatReprQN_(const SparOpReprMatVec &mat_repr_sym_mpo) {
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

template<typename T>
std::vector<size_t> ReorderVecToSameElemNearBy(std::vector<T> &data) {
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
std::vector<size_t> iMPOGenerator<TenElemT, QNT>::SortSymOpMatColsByQN_(SparOpReprMat &sym_op_mat,
                                                                        size_t site) {
  MatrixColQNT &col_qns = mat_rep_qns_[site].second;
  for (auto &qn : col_qns) {
    assert(qn.has_value());
  }
  std::vector<size_t> transposed_idxs = ReorderVecToSameElemNearBy(col_qns);
  sym_op_mat.TransposeCols(transposed_idxs);
  return transposed_idxs;
}

}//qlmps

#endif //QLMPS_ONE_DIM_TN_MPO_MPOGEN_IMPOGEN_IMPL_H
