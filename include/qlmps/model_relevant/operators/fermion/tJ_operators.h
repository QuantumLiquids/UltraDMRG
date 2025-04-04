// SPDX-License-Identifier: LGPL-3.0-only
/*
 * Author: Hao-Xin Wang <wanghaoxin1996@gmail.com>
 * Creation Date: 4/4/2025
 *
 * Description: QuantumLiquids/UltraDMRG project.
 */

#ifndef QLMPS_MODEL_RELEVANT_OPERATORS_FERMION_TJ_OPERATORS_H
#define QLMPS_MODEL_RELEVANT_OPERATORS_FERMION_TJ_OPERATORS_H

#include "qlmps/model_relevant/sites/fermion/tJ_sites.h"

namespace qlmps {

/**
 * Tensor({col, row})  =  <row | \hat O | col >
 */
template<typename TenElemT, typename QNT>
struct tJOperators {
  using QNSctT = qlten::QNSector<QNT>;
  using IndexT = qlten::Index<QNT>;
  using Tensor = qlten::QLTensor<TenElemT, QNT>;

  tJOperators() : tJOperators(sites::tJSite<QNT>()) {};

  tJOperators(const sites::tJSite<QNT> &site) : sz({site.phys_bond_in, site.phys_bond_out}),
                                                sp({site.phys_bond_in, site.phys_bond_out}),
                                                sm({site.phys_bond_in, site.phys_bond_out}),
                                                id({site.phys_bond_in, site.phys_bond_out}),
                                                f({site.phys_bond_in, site.phys_bond_out}),
                                                bupc({site.phys_bond_in, site.phys_bond_out}),
                                                bupa({site.phys_bond_in, site.phys_bond_out}),
                                                bdnc({site.phys_bond_in, site.phys_bond_out}),
                                                bdna({site.phys_bond_in, site.phys_bond_out}),
                                                nf({site.phys_bond_in, site.phys_bond_out}),
                                                nup({site.phys_bond_in, site.phys_bond_out}),
                                                ndn({site.phys_bond_in, site.phys_bond_out}),
                                                bdnc_multi_bupa({site.phys_bond_in, site.phys_bond_out}),
                                                bupc_multi_bdna({site.phys_bond_in, site.phys_bond_out}) {
    const size_t empty = site.empty;
    const size_t spin_up = site.spin_up;
    const size_t spin_down = site.spin_down;

    // Spin operators
    sz({spin_up, spin_up}) = 0.5;         // ⟨↑| S_z |↑⟩ = 0.5
    sz({spin_down, spin_down}) = -0.5;    // ⟨↓| S_z |↓⟩ = -0.5
    sp({spin_down, spin_up}) = 1.0;       // ⟨↑| S^+ |↓⟩ = 1.0
    sm({spin_up, spin_down}) = 1.0;       // ⟨↓| S^- |↑⟩ = 1.0
    id({spin_up, spin_up}) = 1.0;         // ⟨↑| I |↑⟩ = 1
    id({spin_down, spin_down}) = 1.0;     // ⟨↓| I |↓⟩ = 1
    id({empty, empty}) = 1.0;             // ⟨0| I |0⟩ = 1

    // Fermion parity operator
    f({spin_up, spin_up}) = -1;           // ⟨↑| (-1)^N |↑⟩ = -1 (N=1)
    f({spin_down, spin_down}) = -1;       // ⟨↓| (-1)^N |↓⟩ = -1 (N=1)
    f({empty, empty}) = 1;                // ⟨0| (-1)^N |0⟩ = 1 (N=0)

    // Boson operators
    bupc({empty, spin_up}) = 1;           // ⟨↑| c^†_↑ |0⟩ = 1
    bdnc({empty, spin_down}) = 1;         // ⟨↓| c^†_↓ |0⟩ = 1
    bupa({spin_up, empty}) = 1;           // ⟨0| c_↑ |↑⟩ = 1
    bdna({spin_down, empty}) = 1;         // ⟨0| c_↓ |↓⟩ = 1

    // Number operators
    nf({spin_up, spin_up}) = 1;           // ⟨↑| n |↑⟩ = 1
    nf({spin_down, spin_down}) = 1;       // ⟨↓| n |↓⟩ = 1
    nup({spin_up, spin_up}) = 1;          // ⟨↑| n_↑ |↑⟩ = 1
    ndn({spin_down, spin_down}) = 1;      // ⟨↓| n_↓ |↓⟩ = 1

    // Composite operators
    bdnc_multi_bupa({spin_down, spin_up}) = 1;  // ⟨↑| c^†_↓ c_↑ |↓⟩ = 1
    bupc_multi_bdna({spin_up, spin_down}) = 1;  // ⟨↓| c^†_↑ c_↓ |↑⟩ = 1
  }

  // Spin operators
  Tensor sz;
  Tensor sp;
  Tensor sm;
  Tensor id;

  // Fermion and boson operators
  Tensor f;
  Tensor bupc;  // b_up^creation
  Tensor bupa;  // b_up^annihilation
  Tensor bdnc;  // b_down^creation
  Tensor bdna;  // b_down^annihilation

  // Number operators
  Tensor nf;      // fermion number
  Tensor nup;     // spin-up number operator
  Tensor ndn;     // spin-down number operator

  // Composite operators
  Tensor bdnc_multi_bupa;  // c^†_↓ c_↑
  Tensor bupc_multi_bdna;  // c^†_↑ c_↓
};

/**
 * Add hopping terms of t-J model with specified site indices (i, j) into the MPO generator.
 *  The hopping term is given by
 *  -t \sum_{\sigma} c_{i,\sigma}^dag c_{j,\sigma} + (-t)^* \sum_{\sigma} c_{j,\sigma}^dag c_{i,\sigma}
 * 
 *  i can be less or larger than j.
 *  i cannot equal j.
 *
 *  hopping amplitude t can be a complex number but no spin dependence.
 */
template<typename TenElemT, typename QNT>
void AddTJHoppingTerms(MPOGenerator<TenElemT, QNT> &mpo_gen,
                       const TenElemT &t,
                       const size_t i, const size_t j,
                       const tJOperators<TenElemT, QNT> &ops = tJOperators<TenElemT, QNT>()) {
  const size_t site1 = std::min(i, j), site2 = std::max(i, j);
  TenElemT t0;
  if (i < j) {
    t0 = t;
  } else if (i > j) {
    t0 = CalcConj(t);
  } else {
    throw std::runtime_error("i cannot equal j");
  }
  mpo_gen.AddTerm(-t0, {ops.bupc, ops.bupa}, {site1, site2}, {ops.f});
  mpo_gen.AddTerm(-t0, {ops.bdnc, ops.bdna}, {site1, site2}, {ops.f});

  mpo_gen.AddTerm(-CalcConj(t0), {ops.bupa, ops.bupc}, {site1, site2}, {ops.f});
  mpo_gen.AddTerm(-CalcConj(t0), {ops.bdna, ops.bdnc}, {site1, site2}, {ops.f});
}

}//qlmps

#endif //QLMPS_MODEL_RELEVANT_OPERATORS_FERMION_TJ_OPERATORS_H