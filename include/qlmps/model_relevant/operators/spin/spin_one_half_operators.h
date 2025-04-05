// SPDX-License-Identifier: LGPL-3.0-only
/*
 * Author: Hao-Xin Wang <wanghaoxin1996@gmail.com>
 * Creation Date: 4/4/2025
 *
 * Description: QuantumLiquids/UltraDMRG project.
 * Define the operators for spin-1/2 models.
 */

#ifndef QLMPS_MODEL_RELEVANT_OPERATORS_SPIN_SPIN_ONE_HALF_OPERATORS_H
#define QLMPS_MODEL_RELEVANT_OPERATORS_SPIN_SPIN_ONE_HALF_OPERATORS_H

#include "qlmps/model_relevant/sites/spin/spin_one_half_sites.h"

namespace qlmps {

/**
 * Tensor({col, row})  =  <row | \hat O | col >
 */
template<typename TenElemT, typename QNT>
struct SpinOneHalfOperators {
  using QNSctT = qlten::QNSector<QNT>;
  using IndexT = qlten::Index<QNT>;
  using Tensor = qlten::QLTensor<TenElemT, QNT>;

  SpinOneHalfOperators() : SpinOneHalfOperators(sites::SpinOneHalfSite<QNT>()) {};

  SpinOneHalfOperators(const sites::SpinOneHalfSite<QNT> &site) : sz({site.phys_bond_in, site.phys_bond_out}),
                                                                  sp({site.phys_bond_in, site.phys_bond_out}),
                                                                  sm({site.phys_bond_in, site.phys_bond_out}),
                                                                  id({site.phys_bond_in, site.phys_bond_out}),
                                                                  sx({site.phys_bond_in, site.phys_bond_out}),
                                                                  sy({site.phys_bond_in, site.phys_bond_out}) {
    const size_t spin_up = site.spin_up;
    const size_t spin_down = site.spin_down;

    // Spin operators
    sz({spin_up, spin_up}) = 0.5;         // ⟨↑| S_z |↑⟩ = 0.5
    sz({spin_down, spin_down}) = -0.5;    // ⟨↓| S_z |↓⟩ = -0.5
    sp({spin_down, spin_up}) = 1.0;       // ⟨↑| S^+ |↓⟩ = 1.0
    sm({spin_up, spin_down}) = 1.0;       // ⟨↓| S^- |↑⟩ = 1.0
    id({spin_up, spin_up}) = 1.0;         // ⟨↑| I |↑⟩ = 1
    id({spin_down, spin_down}) = 1.0;     // ⟨↓| I |↓⟩ = 1

    if constexpr (std::is_same_v<QNT, qlten::special_qn::TrivialRepQN>) {
      sx({spin_down, spin_up}) = 0.5;     // ⟨↑| S_x |↓⟩ = 0.5
      sx({spin_up, spin_down}) = 0.5;     // ⟨↓| S_x |↑⟩ = 0.5
      if constexpr (std::is_same_v<TenElemT, QLTEN_Complex>) {
        sy({spin_down, spin_up}) = std::complex<double>(0, -0.5);  // ⟨↑| S_y |↓⟩ = -0.5i
        sy({spin_up, spin_down}) = std::complex<double>(0, 0.5);   // ⟨↓| S_y |↑⟩ = 0.5i
      }
    }
  }

  // Spin operators
  Tensor sz;
  Tensor sp;
  Tensor sm;
  Tensor id;

  // Access to sx is restricted to TrivialRepQN
  template<typename Q = QNT>
  typename std::enable_if<std::is_same<Q, qlten::special_qn::TrivialRepQN>::value, Tensor>::type GetSx() const {
    return sx;
  }

  // Access to sy is restricted to TrivialRepQN and QLTEN_Complex
  template<typename Q = QNT, typename T = TenElemT>
  typename std::enable_if<std::is_same<Q, qlten::special_qn::TrivialRepQN>::value &&
      std::is_same<T, QLTEN_Complex>::value, Tensor>::type GetSy() const {
    return sy;
  }

 private:
  Tensor sx; // Only defined for TrivialRepQN
  Tensor sy; // Only defined for TrivialRepQN and QLTEN_Complex
};

/**
 * Add Heisenberg coupling to the MPO generator.
 * The coupling is defined as:
 *   H = J * (S_i^z S_j^z + 1/2 (S_i^+ S_j^- + S_i^- S_j^+))
 * where S_i^z, S_i^+, and S_i^- are the spin operators at site i.
 *
 * OperatorT shoule contain sz, sp, sm operators. 
 * For example, SpinOneHalfOperators<TenElemT, QNT>, tJOperators<TenElemT, QNT>, HubbardOperators<TenElemT, QNT> etc.
 */
template<typename TenElemT, typename QNT, typename OperatorT>
void AddHeisenbergCoupling(MPOGenerator<TenElemT, QNT> &mpo_gen,
                           const double J,
                           const size_t i, const size_t j,
                           const OperatorT &ops = OperatorT()) {
  assert(i != j);
  const size_t site1 = std::min(i, j), site2 = std::max(i, j);
  mpo_gen.AddTerm(J, {ops.sz, ops.sz}, {site1, site2});
  mpo_gen.AddTerm(J / 2, {ops.sp, ops.sm}, {site1, site2});
  mpo_gen.AddTerm(J / 2, {ops.sm, ops.sp}, {site1, site2});
}

} // namespace qlmps

#endif // QLMPS_MODEL_RELEVANT_OPERATORS_SPIN_SPIN_ONE_HALF_OPERATORS_H