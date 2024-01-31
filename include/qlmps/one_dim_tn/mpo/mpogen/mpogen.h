// SPDX-License-Identifier: LGPL-3.0-only

/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2020-07-29 20:43
* 
* Description: QuantumLiquids/UltraDMRG project. MPO generator.
*/

/**
@file mpogen.h
@brief MPO generator for generic quantum many-body systems.
*/
#ifndef QLMPS_ONE_DIM_TN_MPO_MPOGEN_MPOGEN_H
#define QLMPS_ONE_DIM_TN_MPO_MPOGEN_MPOGEN_H

#include "qlmps/consts.h"       // kNullUintVec, kNullUintVecVec
#include "qlmps/site_vec.h"     // SiteVec
#include "qlmps/one_dim_tn/mpo/mpo.h"    // MPO
#include "qlmps/one_dim_tn/mat_repr_mpo.h" //MatReprMPO
#include "qlmps/one_dim_tn/mpo/mpogen/fsm.h"
#include "qlmps/one_dim_tn/mpo/mpogen/symb_alg/coef_op_alg.h"
#include "qlten/qlten.h"

namespace qlmps {
using namespace qlten;

/**
A generic MPO generator. A matrix-product operator (MPO) generator which can
generate an efficient MPO for a quantum many-body system with any type of n-body
interaction term.

@tparam TenElemType Element type of the MPO tensors, can be QLTEN_Double or
        QLTEN_Complex.

@since version 0.0.0
*/
template<typename TenElemT, typename QNT>
class MPOGenerator {
 public:
  using TenElemVec = std::vector<TenElemT>;
  using IndexT = Index<QNT>;
  using QNSctT = QNSector<QNT>;
  using QLTensorT = QLTensor<TenElemT, QNT>;
  using QLTensorVec = std::vector<QLTensorT>;
  using PQLTensorVec = std::vector<QLTensorT *>;

  MPOGenerator(const SiteVec<TenElemT, QNT> &);

  MPOGenerator(const SiteVec<TenElemT, QNT> &, const QNT &);

  void AddTerm(
      TenElemT,
      QLTensorVec,
      std::vector<size_t>
  );

  void AddTerm(
      const TenElemT,
      const QLTensorVec &,
      const std::vector<size_t> &,
      const QLTensorVec &,
      const std::vector<std::vector<size_t>> &inst_ops_idxs_set = kNullUintVecVec
  );

  void AddTerm(
      const TenElemT,
      const QLTensorT &,
      const size_t,
      const QLTensorT &op2 = QLTensorT(),
      const size_t op2_idx = 0,
      const QLTensorT &inst_op = QLTensorT(),
      const std::vector<size_t> &inst_op_idxs = kNullUintVec
  );

  FSM GetFSM(void) { return fsm_; }

  MPO<QLTensorT> Gen(const bool compress = true, const bool show_matrix = false);

  MatReprMPO<QLTensorT> GenMatReprMPO(const bool show_matrix = false);

 private:
  size_t N_;
  SiteVec<TenElemT, QNT> site_vec_;
  std::vector<IndexT> pb_in_vector_;
  std::vector<IndexT> pb_out_vector_;
  QNT zero_div_;
  std::vector<QLTensorT> id_op_vector_;
  FSM fsm_;
  LabelConvertor<TenElemT> coef_label_convertor_;
  LabelConvertor<QLTensorT> op_label_convertor_;

  std::vector<size_t> SortSparOpReprMatColsByQN_(
      SparOpReprMat &, IndexT &, const QLTensorVec &
  );

  QNT CalcTgtRvbQN_(
      const size_t, const size_t, const OpRepr &,
      const QLTensorVec &, const IndexT &
  );

  QLTensorT HeadMpoTenRepr2MpoTen_(
      const SparOpReprMat &,
      const IndexT &,
      const TenElemVec &, const QLTensorVec &
  );

  QLTensorT TailMpoTenRepr2MpoTen_(
      const SparOpReprMat &,
      const IndexT &,
      const TenElemVec &, const QLTensorVec &
  );

  QLTensorT CentMpoTenRepr2MpoTen_(
      const SparOpReprMat &,
      const IndexT &,
      const IndexT &,
      const TenElemVec &,
      const QLTensorVec &, const size_t
  );
};
} /* qlmps */


// Implementation details
#include "qlmps/one_dim_tn/mpo/mpogen/mpogen_impl.h"

#endif /* ifndef QLMPS_ONE_DIM_TN_MPO_MPOGEN_MPOGEN_H */
