// SPDX-License-Identifier: LGPL-3.0-only
/*
 * Author: Hao-Xin Wang <wanghaoxin1996@gmail.com>
 * Creation Date: 4/4/2025
 *
 * Description: QuantumLiquids/UltraDMRG project.
 */

#ifndef QLMPS_MODEL_RELEVANT_OPERATORS_FERMION_HUBBARD_OPERATORS_H
#define QLMPS_MODEL_RELEVANT_OPERATORS_FERMION_HUBBARD_OPERATORS_H

#include "qlmps/model_relevant/sites/fermion/hubbard_sites.h"
#include "qlmps/one_dim_tn/mpo/mpogen/mpogen.h"

namespace qlmps {

/**
 * Tensor({col, row})  =  <row | \hat O | col >
 */
template<typename TenElemT, typename QNT>
struct HubbardOperators {
  using QNSctT = qlten::QNSector<QNT>;
  using IndexT = qlten::Index<QNT>;
  using Tensor = qlten::QLTensor<TenElemT, QNT>;

  HubbardOperators() : HubbardOperators(sites::HubbardSite<QNT>()) {};

  HubbardOperators(const sites::HubbardSite<QNT> &site) : sz({site.phys_bond_in, site.phys_bond_out}),
                                                          sp({site.phys_bond_in, site.phys_bond_out}),
                                                          sm({site.phys_bond_in, site.phys_bond_out}),
                                                          id({site.phys_bond_in, site.phys_bond_out}),
                                                          sx({site.phys_bond_in, site.phys_bond_out}),
                                                          sy({site.phys_bond_in, site.phys_bond_out}),
                                                          f({site.phys_bond_in, site.phys_bond_out}),
                                                          bupc({site.phys_bond_in, site.phys_bond_out}),
                                                          bupa({site.phys_bond_in, site.phys_bond_out}),
                                                          bdnc({site.phys_bond_in, site.phys_bond_out}),
                                                          bdna({site.phys_bond_in, site.phys_bond_out}),
                                                          bupcF({site.phys_bond_in, site.phys_bond_out}),
                                                          bupaF({site.phys_bond_in, site.phys_bond_out}),
                                                          Fbdnc({site.phys_bond_in, site.phys_bond_out}),
                                                          Fbdna({site.phys_bond_in, site.phys_bond_out}),
                                                          cupccdnc({site.phys_bond_in, site.phys_bond_out}),
                                                          cdnacupa({site.phys_bond_in, site.phys_bond_out}),
                                                          nupndn({site.phys_bond_in, site.phys_bond_out}),
                                                          nf({site.phys_bond_in, site.phys_bond_out}),
                                                          nfsquare({site.phys_bond_in, site.phys_bond_out}),
                                                          nup({site.phys_bond_in, site.phys_bond_out}),
                                                          ndn({site.phys_bond_in, site.phys_bond_out}),
                                                          cupccdna(sp),      // Initialize reference
                                                           cdnccupa(sm),      // Initialize reference
                                                           Uterm(nupndn)      // Initialize reference
  {
    const size_t empty = site.empty;
    const size_t double_occupy = site.double_occupancy;
    const size_t spin_up = site.spin_up;
    const size_t spin_down = site.spin_down;

    // Spin operators
    sz({spin_up, spin_up}) = 0.5;         // ⟨↑| S_z |↑⟩ = 0.5
    sz({spin_down, spin_down}) = -0.5;    // ⟨↓| S_z |↓⟩ = -0.5
    sp({spin_down, spin_up}) = 1.0;       // ⟨↑| S^+ |↓⟩ = 1.0
    sm({spin_up, spin_down}) = 1.0;       // ⟨↓| S^- |↑⟩ = 1.0
    id({double_occupy, double_occupy}) = 1.0;  // ⟨↑↓| I |↑↓⟩ = 1
    id({spin_up, spin_up}) = 1.0;         // ⟨↑| I |↑⟩ = 1
    id({spin_down, spin_down}) = 1.0;     // ⟨↓| I |↓⟩ = 1
    id({empty, empty}) = 1.0;             // ⟨0| I |0⟩ = 1
    if constexpr (std::is_same_v<QNT, qlten::special_qn::TrivialRepQN>) {
      sx({spin_down, spin_up}) = 0.5;     // ⟨↑| S_x |↓⟩ = 0.5
      sx({spin_up, spin_down}) = 0.5;     // ⟨↓| S_x |↑⟩ = 0.5
      if constexpr (std::is_same_v<TenElemT, QLTEN_Complex>) {
        sy({spin_down, spin_up}) = std::complex<double>(0, -0.5);  // ⟨↑| S_y |↓⟩ = -0.5i
        sy({spin_up, spin_down}) = std::complex<double>(0, 0.5);   // ⟨↓| S_y |↑⟩ = 0.5i
      }
    }

    // Fermion parity operator
    f({double_occupy, double_occupy}) = 1;  // ⟨↑↓| (-1)^N |↑↓⟩ = 1 (N=2)
    f({spin_up, spin_up}) = -1;             // ⟨↑| (-1)^N |↑⟩ = -1 (N=1)
    f({spin_down, spin_down}) = -1;         // ⟨↓| (-1)^N |↓⟩ = -1 (N=1)
    f({empty, empty}) = 1;                  // ⟨0| (-1)^N |0⟩ = 1 (N=0)

    // Hardcore boson operators (fermionic creation/annihilation)
    bupc({spin_down, double_occupy}) = 1;  // ⟨↑↓| c^†_↑ |↓⟩ = 1
    bupc({empty, spin_up}) = 1;            // ⟨↑| c^†_↑ |0⟩ = 1
    bdnc({spin_up, double_occupy}) = 1;    // ⟨↑↓| c^†_↓ |↑⟩ = 1
    bdnc({empty, spin_down}) = 1;          // ⟨↓| c^†_↓ |0⟩ = 1
    bupa({double_occupy, spin_down}) = 1;  // ⟨↓| c_↑ |↑↓⟩ = 1
    bupa({spin_up, empty}) = 1;            // ⟨0| c_↑ |↑⟩ = 1
    bdna({double_occupy, spin_up}) = 1;    // ⟨↑| c_↓ |↑↓⟩ = 1
    bdna({spin_down, empty}) = 1;          // ⟨0| c_↓ |↓⟩ = 1

    // Composite operators
    bupcF({spin_down, double_occupy}) = -1;  // ⟨↑↓| c^†_↑ f |↓⟩ = -1
    bupcF({empty, spin_up}) = 1;             // ⟨↑| c^†_↑ f |0⟩ = 1
    Fbdnc({spin_up, double_occupy}) = 1;     // ⟨↑↓| f c^†_↓ |↑⟩ = 1
    Fbdnc({empty, spin_down}) = -1;          // ⟨↓| f c^†_↓ |0⟩ = -1
    bupaF({double_occupy, spin_down}) = 1;   // ⟨↓| c_↑ f |↑↓⟩ = 1
    bupaF({spin_up, empty}) = -1;            // ⟨0| c_↑ f |↑⟩ = -1
    Fbdna({double_occupy, spin_up}) = -1;    // ⟨↑| f c_↓ |↑↓⟩ = -1
    Fbdna({spin_down, empty}) = 1;           // ⟨0| f c_↓ |↓⟩ = 1

    // Pairing operators
    cupccdnc({empty, double_occupy}) = 1;    // ⟨↑↓| c^†_↑ c^†_↓ |0⟩ = 1
    cdnacupa({double_occupy, empty}) = 1;    // ⟨0| c_↓ c_↑ |↑↓⟩ = 1

    // Hubbard U term and fermion number operators
    nupndn({double_occupy, double_occupy}) = 1;  // ⟨↑↓| n_↑ n_↓ |↑↓⟩ = 1
    nf({double_occupy, double_occupy}) = 2;      // ⟨↑↓| (n_↑ + n_↓) |↑↓⟩ = 2
    nf({spin_up, spin_up}) = 1;                  // ⟨↑| (n_↑ + n_↓) |↑⟩ = 1
    nf({spin_down, spin_down}) = 1;              // ⟨↓| (n_↑ + n_↓) |↓⟩ = 1
    nfsquare({double_occupy, double_occupy}) = 4;// ⟨↑↓| (n_↑ + n_↓)^2 |↑↓⟩ = 4
    nfsquare({spin_up, spin_up}) = 1;            // ⟨↑| (n_↑ + n_↓)^2 |↑⟩ = 1
    nfsquare({spin_down, spin_down}) = 1;        // ⟨↓| (n_↑ + n_↓)^2 |↓⟩ = 1
    nup({double_occupy, double_occupy}) = 1;     // ⟨↑↓| n_↑ |↑↓⟩ = 1
    nup({spin_up, spin_up}) = 1;                 // ⟨↑| n_↑ |↑⟩ = 1
    ndn({double_occupy, double_occupy}) = 1;     // ⟨↑↓| n_↓ |↑↓⟩ = 1
    ndn({spin_down, spin_down}) = 1;             // ⟨↓| n_↓ |↓⟩ = 1
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

// Fermion parity operator
  Tensor f;

// Hardcore boson operators
  Tensor bupc;  // b_up^creation
  Tensor bupa;  // b_up^annihilation
  Tensor bdnc;  // b_down^creation
  Tensor bdna;  // b_down^annihilation

// Composite operators
  Tensor bupcF; // bupc * f
  Tensor bupaF; // bupa * f
  Tensor Fbdnc; // f * bdnc
  Tensor Fbdna; // f * bdna

// Pairing operators
  Tensor cupccdnc;  // c_up^creation * c_down^creation
  Tensor cdnacupa;  // onsite pair operator (c_down*c_up)
  const Tensor &cupccdna; // exciton pair
  const Tensor &cdnccupa;

// density relevant operators
  Tensor nupndn;  // nup * ndown
  const Tensor &Uterm;   // Hubbard U term
  Tensor nf;      // fermion number (nup + ndn)
  Tensor nfsquare;// square of fermion number
  Tensor nup;     // spin-up number operator
  Tensor ndn;     // spin-down number operator

 private:
  Tensor sx;
  Tensor sy;
};

/**
 * Add hopping terms of Hubbard model with specified site indices (i, j) into the MPO generator.
 *  The hopping term is given by
 *  -t \sum_{\sigma} \left(c_{i,\sigma}^dag c_{j,\sigma} + c_{j,\sigma}^dag c_{i,\sigma}\right)
 *
 *  i can be less or larger than j.
 *  i cannot equal j.
 *
 *  hopping amplitude t can only be a real number.
 */
template<typename TenElemT, typename QNT>
void AddHubbardHoppingTerms(MPOGenerator<TenElemT, QNT> &mpo_gen,
                            const double t,
                            const size_t i, const size_t j,
                            const HubbardOperators<TenElemT, QNT> &ops = HubbardOperators<TenElemT, QNT>()) {
  const size_t site1 = std::min(i, j), site2 = std::max(i, j);
  mpo_gen.AddTerm(-t, ops.bupcF, site1, ops.bupa, site2, ops.f);
  mpo_gen.AddTerm(t, ops.bupaF, site1, ops.bupc, site2, ops.f);
  mpo_gen.AddTerm(-t, ops.bdnc, site1, ops.Fbdna, site2, ops.f);
  mpo_gen.AddTerm(t, ops.bdna, site1, ops.Fbdnc, site2, ops.f);
}

}//qlmps
#endif //QLMPS_MODEL_RELEVANT_OPERATORS_FERMION_HUBBARD_OPERATORS_H
