// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author:  Hao-Xin Wang <wanghaoxin1996@gmail.com>
* Creation Date: 2025-04-04
*
* Description: QuantumLiquids/UltraDMRG project.
 * Define the sites/hilbert space on the t-J like models.
*/

#ifndef QLMPS_MODEL_RELEVANT_SITES_FERMION_TJ_SITES_H
#define QLMPS_MODEL_RELEVANT_SITES_FERMION_TJ_SITES_H

#include "qlten/qlten.h"
#include "qlmps/model_relevant/sites/model_site_base.h"

namespace qlmps {
using namespace qlten;
namespace sites {
/**
 * @tparam QNT : can be 1. qlten::special_qn::U1U1QN
 *                      2. qlten::special_qn::TrivialRepQN
 *                      3. qlten::special_qn::U1QN  (conserve spin Sz rather particle number)
 */
template<typename QNT>
struct tJSite : public ModelSiteBase<QNT> {
  using QNSctT = qlten::QNSector<QNT>;
  using IndexT = qlten::Index<QNT>;

  tJSite() {
    if constexpr (std::is_same_v<QNT, qlten::special_qn::U1U1QN>) {
      // U(1) x U(1) symmetry
      this->phys_bond_out = IndexT({QNSctT(QNT({QNCard("N", U1QNVal(1)), QNCard("Sz", U1QNVal(1))}), 1),
                                    QNSctT(QNT({QNCard("N", U1QNVal(1)), QNCard("Sz", U1QNVal(-1))}), 1),
                                    QNSctT(QNT({QNCard("N", U1QNVal(0)), QNCard("Sz", U1QNVal(0))}), 1)},
                                   TenIndexDirType::OUT
      );
    } else if constexpr (std::is_same_v<QNT, qlten::special_qn::TrivialRepQN>) {
      // No any symmetry
      this->phys_bond_out = IndexT({QNSctT(QNT(), 3)}, TenIndexDirType::OUT);
    } else if constexpr (std::is_same_v<QNT, qlten::special_qn::U1QN>) {
      // Spin Sz conservation
      this->phys_bond_out = IndexT({QNSctT(QNT({QNCard("Sz", U1QNVal(1))}), 1),
                                    QNSctT(QNT({QNCard("Sz", U1QNVal(-1))}), 1),
                                    QNSctT(QNT({QNCard("Sz", U1QNVal(0))}), 1)}, TenIndexDirType::OUT);
    } else {
      // Handle unsupported QNT types with a compile-time error
      static_assert(false, "Unsupported QNT type for tJSite constructor");
    }
    spin_up = 0;
    spin_down = 1;
    empty = 2;
    this->phys_bond_in = InverseIndex(this->phys_bond_out);
  }

  // define the order of the basis, map to the numbers 0,1, and 2
  size_t spin_up;
  size_t spin_down;
  size_t empty;

}; // tJSite

}//sites
}//qlmps

#endif //QLMPS_MODEL_RELEVANT_SITES_FERMION_TJ_SITES_H