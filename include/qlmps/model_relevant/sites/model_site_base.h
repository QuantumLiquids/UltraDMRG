// SPDX-License-Identifier: LGPL-3.0-only
/*
 * Author: Hao-Xin Wang <wanghaoxin1996@gmail.com>
 * Creation Date: 4/4/2025
 *
 * Description: QuantumLiquids/UltraDMRG project.
 */

#ifndef QLMPS_MODEL_RELEVANT_SITES_MODEL_SITE_BASE_H
#define QLMPS_MODEL_RELEVANT_SITES_MODEL_SITE_BASE_H

#include "qlten/qlten.h"
#include "qlmps/site_vec.h"

namespace qlmps {
namespace sites {

template<typename QNT>
struct ModelSiteBase {
  using QNSctT = qlten::QNSector<QNT>;
  using IndexT = qlten::Index<QNT>;
 public:
  template<typename TenElemT>
  SiteVec<TenElemT, QNT> GenUniformSites(const size_t N) {
    return SiteVec<TenElemT, QNT>(N, phys_bond_out);
  }
  IndexT phys_bond_out;
  IndexT phys_bond_in;
};

}; //sites
}; // qlmps

#include "qlmps/model_relevant/sites/spin/spin_one_half_sites.h"

#include "qlmps/model_relevant/sites/fermion/hubbard_sites.h"
#include "qlmps/model_relevant/sites/fermion/tJ_sites.h"

#endif //QLMPS_MODEL_RELEVANT_SITES_MODEL_SITE_BASE_H
