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

  /// Constructor
  iMPOGenerator(const SiteVec<TenElemT, QNT> &);

  /// General add hamiltonian terms
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
  /// General add hamiltonian terms which have insertion operators
  void AddTerm(
      const TenElemT coef,
      const QLTensorVec &phys_ops,
      const std::vector<size_t> &phys_ops_sites,
      const QLTensorVec &inst_ops,
      const std::vector<std::vector<size_t>> &inst_ops_sites_set);

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

  // Two-site term with insertion operators, like fermionic models.
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

  MPO<Tensor> Gen();

 private:
  void EvaluateMatReprQN_(const SparOpReprMatVec &mat_repr_sym_mpo);
  std::vector<size_t> SortSymOpMatColsByQN_(SparOpReprMat &sym_op_mat,
                                            size_t site);
  //Base function used in AddTerm
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
   * And ascent. If winding, just increase it by multiplication
   * of unite_cell_length
   * @param coef
   * @param local_ops
   * @param local_ops_site
   */
  OpReprVec TransferHamiltonianTermToConsecutiveOpRepr_(
      TenElemT coef,
  QLTensorVec local_ops,
      std::vector<size_t> local_ops_sites
  );

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
