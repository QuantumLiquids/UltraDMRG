// SPDX-License-Identifier: LGPL-3.0-only

/*
* Author: Hao-Xin Wang <wanghaoxin1996@gmail.com>
* Creation Date: 2023-04-22
*
* Description: QuantumLiquids/UltraDMRG project. DMRG.
*/

#ifndef QLMPS_ALGORITHM_DMRG_DMRG_H
#define QLMPS_ALGORITHM_DMRG_DMRG_H

#include "qlmps/one_dim_tn/mpo/mat_repr_mpo.h"

namespace qlmps {
template<typename TenT>
using RightBlockOperatorGroup = std::vector<TenT>; // right block == environment

template<typename TenT>
using LeftBlockOperatorGroup = std::vector<TenT>; //left block == system

template<typename TenT>
using BlockOperatorGroup = std::vector<TenT>;

template<typename TenT>
using BlockSiteHamiltonianTerm = std::array<TenT *, 2>;

template<typename TenT>
using BlockSiteHamiltonianTermGroup = std::vector<BlockSiteHamiltonianTerm<TenT>>;

template<typename TenT>
using SiteBlockHamiltonianTerm = std::array<TenT *, 2>;

template<typename TenT>
using SiteBlockHamiltonianTermGroup = std::vector<SiteBlockHamiltonianTerm<TenT>>;

template<typename TenT>
using SuperBlockHamiltonianTerms = std::vector<std::pair<BlockSiteHamiltonianTermGroup<TenT>,
                                                         SiteBlockHamiltonianTermGroup<TenT>>>;

template<typename TenT>
using EffectiveHamiltonianTerm = std::array<TenT *, 4>;

template<typename TenT>
using EffectiveHamiltonianTermGroup = std::vector<EffectiveHamiltonianTerm<TenT>>;

template<typename TenT>
class EffectiveHamiltonian {
 public:
  EffectiveHamiltonianTermGroup<TenT> GetEffectiveHamiltonianTermGroup() {

  }

  RightBlockOperatorGroup<TenT> right_op_gp;
  LeftBlockOperatorGroup<TenT> left_op_gp;
  MatReprMPO<TenT> mat_repr_mpo_a; //left site
  MatReprMPO<TenT> mat_repr_mpo_b; //right site
};

}

#include "qlmps/algorithm/dmrg/dmrg_impl.h"
#include "qlmps/algorithm/dmrg/dmrg_init.h"

#endif //QLMPS_ALGORITHM_DMRG_DMRG_H
