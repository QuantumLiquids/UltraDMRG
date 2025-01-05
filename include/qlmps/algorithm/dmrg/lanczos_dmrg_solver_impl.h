// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Hao-Xin Wang <wanghaoxin1996@gmail.com>
* Creation Date: 2023-04-23
*
* Description: QuantumLiquids/UltraDMRG project. Implementation details for Lanczos solver in DMRG.
*/

/**
@file lanczos_dmrg_solver_impl.h
@brief Implementation details for Lanczos solver in DMRG.
*/

#ifndef QLMPS_ALGORITHM_LANCZOS_DMRG_SOLVER_IMPL_H
#define QLMPS_ALGORITHM_LANCZOS_DMRG_SOLVER_IMPL_H

#include "qlmps/algorithm/lanczos_params.h"    // LanczosParams
#include "qlten/qlten.h"
#include "qlten/utility/timer.h"                // Timer
#include "qlmps/algorithm/dmrg/dmrg.h"         // EffectiveHamiltonianTerm

#include <iostream>
#include <vector>     // vector
#include <cstring>

namespace qlmps {

using namespace qlten;

// Forward declarations.
template<typename TenElemT, typename QNT>
void combine_operators_in_super_blk_hamiltonian(
    SuperBlockHamiltonianTerms<QLTensor<TenElemT, QNT>> &eff_ham,
    std::vector<QLTensor<TenElemT, QNT>> &block_site_ops,  //output
    std::vector<QLTensor<TenElemT, QNT>> &site_block_ops   //output
);

template<typename TenElemT, typename QNT>
QLTensor<TenElemT, QNT> *super_block_hamiltonian_mul_two_site_state(
    const std::vector<QLTensor<TenElemT, QNT>> &block_site_ops,
    const std::vector<QLTensor<TenElemT, QNT>> &site_block_ops,
    QLTensor<TenElemT, QNT> *state
);

/**
Obtain the lowest energy eigenvalue and corresponding eigenstate from the effective
Hamiltonian and a initial state using Lanczos algorithm.

eff_ham will be erased after calling Lanczos

@param peff_ham Effective Hamiltonian as a vector of pointer-to-tensors.
@param pinit_state Pointer to initial state for Lanczos iteration.
@param params Parameters for Lanczos solver.
*/
template<typename TenT>
LanczosRes<TenT> LanczosSolver(
    SuperBlockHamiltonianTerms<TenT> &eff_ham,
    TenT *pinit_state,
    const LanczosParams &params,
    std::vector<TenT> &block_site_ops,  //output
    std::vector<TenT> &site_block_ops   //output
) {
  // Take care that init_state will be destroyed after call the solver
  size_t eff_ham_eff_dim = pinit_state->size();

  LanczosRes<TenT> lancz_res;

  std::vector<std::vector<size_t>> energy_measu_ctrct_axes;
  if (pinit_state->Rank() == 3) {            // For single site update algorithm
    energy_measu_ctrct_axes = {{0, 1, 2}, {0, 1, 2}};
  } else if (pinit_state->Rank() == 4) {    // For two site update algorithm
    energy_measu_ctrct_axes = {{0, 1, 2, 3}, {0, 1, 2, 3}};
  }

  std::vector<TenT *> bases(params.max_iterations, nullptr);
  std::vector<QLTEN_Double> a(params.max_iterations, 0.0);
  std::vector<QLTEN_Double> b(params.max_iterations, 0.0);
  std::vector<QLTEN_Double> N(params.max_iterations, 0.0);

  // Initialize Lanczos iteration.
  pinit_state->Normalize();
  bases[0] = pinit_state;

#ifdef QLMPS_TIMING_MODE
  Timer mat_vec_timer("lancz_mat_vec");
#endif

  combine_operators_in_super_blk_hamiltonian(eff_ham, block_site_ops, site_block_ops);
  auto last_mat_mul_vec_res = super_block_hamiltonian_mul_two_site_state(block_site_ops, site_block_ops, bases[0]);

#ifdef QLMPS_TIMING_MODE
  mat_vec_timer.PrintElapsed();
#endif

  TenT temp_scalar_ten;
  auto base_dag = Dag(*bases[0]);
  Contract(
      last_mat_mul_vec_res, &base_dag,
      energy_measu_ctrct_axes,
      &temp_scalar_ten
  );
  a[0] = Real(temp_scalar_ten());;
  N[0] = 0.0;
  size_t m = 0;
  QLTEN_Double energy0;
  energy0 = a[0];
  // Lanczos iterations.
  while (true) {
    m += 1;
    auto gamma = last_mat_mul_vec_res;
    if (m == 1) {
      LinearCombine({-a[m - 1]}, {bases[m - 1]}, 1.0, gamma);
    } else {
      LinearCombine(
          {-a[m - 1], -std::sqrt(N[m - 1])},
          {bases[m - 1], bases[m - 2]},
          1.0,
          gamma
      );
    }
    auto norm_gamma = gamma->Normalize();
    QLTEN_Double eigval;
    QLTEN_Double *eigvec = nullptr;
    if (norm_gamma == 0.0) {
      if (m == 1) {
        lancz_res.iters = m;
        lancz_res.gs_eng = energy0;
        lancz_res.gs_vec = new TenT(*bases[0]);
        LanczosFree(eigvec, bases, last_mat_mul_vec_res);
        return lancz_res;
      } else {
        TridiagGsSolver(a, b, m, eigval, eigvec, 'V');
        auto gs_vec = new TenT(bases[0]->GetIndexes());
        LinearCombine(m, eigvec, bases, 0.0, gs_vec);
        lancz_res.iters = m;
        lancz_res.gs_eng = energy0;
        lancz_res.gs_vec = gs_vec;
        LanczosFree(eigvec, bases, m, last_mat_mul_vec_res);
        return lancz_res;
      }
    }

    N[m] = norm_gamma * norm_gamma;
    b[m - 1] = norm_gamma;
    bases[m] = gamma;

#ifdef QLMPS_TIMING_MODE
    mat_vec_timer.ClearAndRestart();
#endif

    last_mat_mul_vec_res = super_block_hamiltonian_mul_two_site_state(block_site_ops, site_block_ops, bases[m]);

#ifdef QLMPS_TIMING_MODE
    mat_vec_timer.PrintElapsed();
#endif

    TenT temp_scalar_ten;
    auto base_dag = Dag(*bases[m]);
    Contract(
        last_mat_mul_vec_res,
        &base_dag,
        energy_measu_ctrct_axes,
        &temp_scalar_ten
    );
    a[m] = Real(temp_scalar_ten());
    TridiagGsSolver(a, b, m + 1, eigval, eigvec, 'N');
    auto energy0_new = eigval;
    if (
        ((energy0 - energy0_new) < params.error) ||
            (m == eff_ham_eff_dim) ||
            (m == params.max_iterations - 1)
        ) {
      TridiagGsSolver(a, b, m + 1, eigval, eigvec, 'V');
      energy0 = energy0_new;
      auto gs_vec = new TenT(bases[0]->GetIndexes());
      LinearCombine(m + 1, eigvec, bases, 0.0, gs_vec);
      lancz_res.iters = m + 1;
      lancz_res.gs_eng = energy0;
      lancz_res.gs_vec = gs_vec;
      LanczosFree(eigvec, bases, m + 1, last_mat_mul_vec_res);
      return lancz_res;
    } else {
      energy0 = energy0_new;
    }
  }
}

/*
 * |----1                       1-----
 * |          1        1             |
 * |          |        |             |
 * |          |        |             |
 * |          0        0             |
 * |          1        2             |
 * |          |        |             |
 * |----0 0-------------------3 0----|
 */
template<typename TenElemT, typename QNT>
void combine_operators_in_super_blk_hamiltonian( //first time do this
    SuperBlockHamiltonianTerms<QLTensor<TenElemT, QNT>> &eff_ham,
    std::vector<QLTensor<TenElemT, QNT>> &block_site_ops,  //output
    std::vector<QLTensor<TenElemT, QNT>> &site_block_ops   //output
) {
  using TenT = QLTensor<TenElemT, QNT>;
  size_t num_terms = eff_ham.size();
  block_site_ops.resize(num_terms);
  site_block_ops.resize(num_terms);
  for (size_t i = 0; i < num_terms; i++) {
    // for block-site
    auto &block_site_terms = eff_ham[i].first;
    auto pblock_site_ops_res_s = std::vector<TenT *>(block_site_terms.size());
    for (size_t j = 0; j < block_site_terms.size(); j++) {
      pblock_site_ops_res_s[j] = new TenT();
      Contract(block_site_terms[j][0], block_site_terms[j][1], {{}, {}}, pblock_site_ops_res_s[j]);
    }
    std::vector<TenElemT> coefs = std::vector<TenElemT>(block_site_terms.size(), TenElemT(1.0));
    LinearCombine(coefs, pblock_site_ops_res_s, TenElemT(0.0), &block_site_ops[i]);
    for (size_t j = 0; j < block_site_terms.size(); j++) {
      delete pblock_site_ops_res_s[j];
    }
    block_site_ops[i].Transpose({1, 3, 0, 2});
    // for site-block
    auto &site_block_terms = eff_ham[i].second;
    auto psite_block_ops_res_s = std::vector<TenT *>(site_block_terms.size());
    for (size_t j = 0; j < site_block_terms.size(); j++) {
      psite_block_ops_res_s[j] = new TenT();
      Contract(site_block_terms[j][0], site_block_terms[j][1], {{}, {}}, psite_block_ops_res_s[j]);
    }
    coefs = std::vector<TenElemT>(site_block_terms.size(), TenElemT(1.0));
    LinearCombine(coefs, psite_block_ops_res_s, TenElemT(0.0), &site_block_ops[i]);
    for (size_t j = 0; j < site_block_terms.size(); j++) {
      delete psite_block_ops_res_s[j];
    }
    site_block_ops[i].Transpose({0, 2, 1, 3});
  }
  eff_ham.clear();
}

template<typename TenElemT, typename QNT>
QLTensor<TenElemT, QNT> *super_block_hamiltonian_mul_two_site_state(
    const std::vector<QLTensor<TenElemT, QNT>> &block_site_ops,
    const std::vector<QLTensor<TenElemT, QNT>> &site_block_ops,
    QLTensor<TenElemT, QNT> *state
) {
  using TenT = QLTensor<TenElemT, QNT>;
  size_t num_terms = block_site_ops.size();
  auto multiplication_res = std::vector<TenT>(num_terms);
  auto pmultiplication_res = std::vector<TenT *>(num_terms);
  const std::vector<TenElemT> &coefs = std::vector<TenElemT>(num_terms, TenElemT(1.0));
  for (size_t i = 0; i < num_terms; i++) {
    TenT temp1;
    Contract(&block_site_ops[i], state, {{2, 3}, {0, 1}}, &temp1);
    Contract(&temp1, &site_block_ops[i], {{2, 3}, {0, 1}}, &multiplication_res[i]);
    pmultiplication_res[i] = &multiplication_res[i];
  }
  auto res = new TenT;
  //TODO: optimize the summation
  LinearCombine(coefs, pmultiplication_res, TenElemT(0.0), res);
  return res;
}

} /* qlmps */

#endif//QLMPS_ALGORITHM_LANCZOS_DMRG_SOLVER_IMPL_H
