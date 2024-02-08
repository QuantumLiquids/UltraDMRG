// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Hao-Xin Wang <wanghaoxin1996@gmail.com>
* Creation Date: 2024-02-05
*
* Description: QuantumLiquids/UltraDMRG project. infinite MPO Generator.
*/

#ifndef QLMPS_ONE_DIM_TN_MPO_MPOGEN_IMPOGEN_H
#define QLMPS_ONE_DIM_TN_MPO_MPOGEN_IMPOGEN_H

#include <optional>
#include "qlten/qlten.h"
#include "qlmps/one_dim_tn/mpo/mpogen/ifsm.h"     //iFSM
#include "qlmps/site_vec.h"                       // SiteVec

namespace qlmps {
using namespace qlten;
template<typename TenElemT, typename QNT>
class iMPOGenerator {
 public:
  using TenElemVec = std::vector<TenElemT>;
  using IndexT = Index<QNT>;
  using QNSctT = QNSector<QNT>;
  using Tensor = QLTensor<TenElemT, QNT>;
  using QLTensorVec = std::vector<Tensor>;

  using MatrixRowQNT = std::vector<std::optional<QNT>>;
  using MatrixColQNT = std::vector<std::optional<QNT>>;
  using MatrixQNT = std::pair<MatrixRowQNT, MatrixColQNT>;

  iMPOGenerator(const SiteVec<TenElemT, QNT> &);
  void AddTerm(
      TenElemT coef,
      QLTensorVec local_ops,
      std::vector<size_t> local_ops_sites
  ) {
    AddConsecutiveOps_(local_ops_sites.front(),
                       TransferHamiltonianTermToConsecutiveOpRepr_(
                           coef, local_ops, local_ops_sites
                       ));
  }

  void AddTerm(
      const TenElemT coef,
      const QLTensorVec &phys_ops,
      const std::vector<size_t> &phys_ops_sites,
      const QLTensorVec &inst_ops,
      const std::vector<std::vector<size_t>> &inst_ops_sites_set
  ) {
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

  // Single site term
  void AddTerm(
      const TenElemT coef, const size_t site, const Tensor &op
  ) {
    AddTerm(coef, QLTensorVec{op}, std::vector<size_t>{site});
  }

  // Two-site term
  void AddTerm(
      const TenElemT coef,
      const size_t site1, Tensor &op1,
      const size_t site2, Tensor &op2
  ) {
    AddTerm(coef, QLTensorVec{op1, op2}, std::vector<size_t>{site1, site2});
  }

  void AddTerm(
      const TenElemT coef,
      const size_t site1, Tensor &op1,
      const size_t site2, Tensor &op2,
      const std::vector<size_t> &inst_op_sites,
      const Tensor &inst_op
  ) {
    AddTerm(
        coef,
        {op1, op2}, {site1, site2},
        {inst_op}, {inst_op_sites}
    );
  }

 private:
  void EvaluateMatReprQN_(const SparOpReprMatVec &mat_repr_sym_mpo);
  std::vector<size_t> SortSymOpMatColsByQN_(SparOpReprMat &sym_op_mat,
                                            size_t site);
  void AddConsecutiveOps_(
      const size_t head_site,
      const OpReprVec &ops_reprs
  ) {
    ifsm_.AddPath(head_site, ops_reprs);
    size_t winding_num = ops_reprs.size() / unit_cell_size_;
    winding_num_max_ = std::max(winding_num, winding_num_max_);
  }
  /**
   * The first element of local_ops_site should be in [0, unit_cell_length),
   * And ascent. If winding, just increase it by mutiplication
   * of unite_cell_length
   * @param coef
   * @param local_ops
   * @param local_ops_site
   */
  OpReprVec TransferHamiltonianTermToConsecutiveOpRepr_(
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
  size_t unit_cell_size_;
  SiteVec<TenElemT, QNT> site_vec_;
  std::vector<Tensor> id_op_vector_;
  LabelConvertor<TenElemT> coef_label_convertor_;
  LabelConvertor<Tensor> op_label_convertor_;
  size_t winding_num_max_;
  iFSM ifsm_;
  std::vector<MatrixQNT> mat_rep_qns_;
};

} //qlmps

#include "qlmps/one_dim_tn/mpo/mpogen/impo_gen_impl.h"
#endif //QLMPS_ONE_DIM_TN_MPO_MPOGEN_IMPOGEN_H
