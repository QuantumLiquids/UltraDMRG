// SPDX-License-Identifier: LGPL-3.0-only
/*
 * Author: Hao-Xin Wang <wanghaoxin1996@gmail.com>
 * Creation Date: 4/4/2025
 *
 * Description: QuantumLiquids/UltraDMRG project.
 * Define the sites/hilbert space for spin-1/2 models.
 */

#ifndef QLMPS_MODEL_RELEVANT_SITES_SPIN_SPIN_ONE_HALF_SITES_H
#define QLMPS_MODEL_RELEVANT_SITES_SPIN_SPIN_ONE_HALF_SITES_H

#include "qlten/qlten.h"
#include "qlmps/model_relevant/sites/model_site_base.h"

namespace qlmps {
using namespace qlten;
namespace sites {

/**
 * @tparam QNT : can be 1. qlten::special_qn::U1QN (conserve spin Sz)
 *                      2. qlten::special_qn::TrivialRepQN (no symmetry)
 */
template<typename QNT>
struct SpinOneHalfSite : public ModelSiteBase<QNT> {
  using QNSctT = qlten::QNSector<QNT>;
  using IndexT = qlten::Index<QNT>;

  SpinOneHalfSite() {
    if constexpr (std::is_same_v<QNT, qlten::special_qn::U1QN>) {
      // Spin Sz conservation
      this->phys_bond_out = IndexT({QNSctT(QNT({QNCard("Sz", U1QNVal(1))}), 1),
                                    QNSctT(QNT({QNCard("Sz", U1QNVal(-1))}), 1)}, TenIndexDirType::OUT);
    } else if constexpr (std::is_same_v<QNT, qlten::special_qn::TrivialRepQN>) {
      // No symmetry
      this->phys_bond_out = IndexT({QNSctT(QNT(), 2)}, TenIndexDirType::OUT);
    } else {
      // Handle unsupported QNT types with a compile-time error
      static_assert(false, "Unsupported QNT type for SpinOneHalfSite constructor");
    }
    this->phys_bond_in = InverseIndex(this->phys_bond_out);
  }

  // Define the order of the basis, map to the numbers 0 and 1
  const size_t spin_up = 0;
  const size_t spin_down = 1;

}; // SpinOneHalfSite

} // namespace sites
} // namespace qlmps

#endif // QLMPS_MODEL_RELEVANT_SITES_SPIN_SPIN_ONE_HALF_SITES_H