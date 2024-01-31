// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Hao-Xin Wang <wanghaoxin1996@gmail.com>
* Creation Date: 2023-05-12
*
* Description: QuantumLiquids/UltraDMRG project. Implementation details for Lanczos solver in DMRG.
*/


#ifndef QLMPS_ALGO_MPI_LANCZOS_SOLVER_MPI_MASTER_H
#define QLMPS_ALGO_MPI_LANCZOS_SOLVER_MPI_MASTER_H


/**
@file lanczos_dmrg_solver_mpi_master.h
@brief Implementation details for Lanczos solver in DMRG.
*/
#include "qlmps/algorithm/lanczos_params.h"    // LanczosParams
#include "qlten/qlten.h"
#include "qlten/utility/timer.h"                // Timer
#include "qlmps/algorithm/dmrg/dmrg.h"         // EffectiveHamiltonianTerm
#include "qlmps/algo_mpi/mps_algo_order.h"

#include <iostream>
#include <vector>     // vector
#include <cstring>

#include "qlmps/algo_mpi/dmrg/dmrg_mpi_impl_master.h"

namespace qlmps {

using namespace qlten;

/**
Obtain the lowest energy eigenvalue and corresponding eigenstate from the effective
Hamiltonian and a initial state using Lanczos algorithm.

@param rpeff_ham Effective Hamiltonian as a vector of pointer-to-tensors.
@param pinit_state Pointer to initial state for Lanczos iteration.
@param eff_ham_mul_state Function pointer to effective Hamiltonian multiply to state.
@param params Parameters for Lanczos solver.
*/
template<typename TenElemT, typename QNT>
LanczosRes<QLTensor<TenElemT, QNT>> DMRGMPIMasterExecutor<TenElemT,
                                                          QNT>::LanczosSolver_(DMRGMPIMasterExecutor::Tensor *pinit_state) {
  const LanczosParams &params = sweep_params.lancz_params;
  // Take care that init_state will be destroyed after call the solver
  size_t eff_ham_eff_dim = pinit_state->size();

  LanczosRes<Tensor> lancz_res;

  std::vector<std::vector<size_t>> energy_measu_ctrct_axes;
  if (pinit_state->Rank() == 3) {            // For single site update algorithm
    energy_measu_ctrct_axes = {{0, 1, 2}, {0, 1, 2}};
  } else if (pinit_state->Rank() == 4) {    // For two site update algorithm
    energy_measu_ctrct_axes = {{0, 1, 2, 3}, {0, 1, 2, 3}};
  }

  std::vector<Tensor *> bases(params.max_iterations, nullptr);
  std::vector<QLTEN_Double> a(params.max_iterations, 0.0);
  std::vector<QLTEN_Double> b(params.max_iterations, 0.0);
  std::vector<QLTEN_Double> N(params.max_iterations, 0.0);

  // Initialize Lanczos iteration.
  pinit_state->Normalize();
  bases[0] = pinit_state;

#ifdef QLMPS_TIMING_MODE
  Timer mat_vec_timer("lancz_mat_vec");
#endif
  MasterBroadcastOrder(lanczos_mat_vec_dynamic, world_);

  auto last_mat_mul_vec_res =
      DynamicHamiltonianMultiplyState_(*bases[0]);

#ifdef QLMPS_TIMING_MODE
  mat_vec_timer.PrintElapsed();
#endif

  Tensor temp_scalar_ten;
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
        lancz_res.gs_vec = new Tensor(*bases[0]);
        LanczosFree(eigvec, bases, last_mat_mul_vec_res);
        MasterBroadcastOrder(lanczos_finish, world_);
        return lancz_res;
      } else {
        TridiagGsSolver(a, b, m, eigval, eigvec, 'V');
        auto gs_vec = new Tensor(bases[0]->GetIndexes());
        LinearCombine(m, eigvec, bases, 0.0, gs_vec);
        lancz_res.iters = m;
        lancz_res.gs_eng = energy0;
        lancz_res.gs_vec = gs_vec;
        LanczosFree(eigvec, bases, m, last_mat_mul_vec_res);
        MasterBroadcastOrder(lanczos_finish, world_);
        return lancz_res;
      }
    }

    N[m] = norm_gamma * norm_gamma;
    b[m - 1] = norm_gamma;
    bases[m] = gamma;

#ifdef QLMPS_TIMING_MODE
    mat_vec_timer.ClearAndRestart();
#endif
    MasterBroadcastOrder(lanczos_mat_vec_static, world_);
    last_mat_mul_vec_res = StaticHamiltonianMultiplyState_(*bases[m], a[m]);

#ifdef QLMPS_TIMING_MODE
    mat_vec_timer.PrintElapsed();
#endif

    TridiagGsSolver(a, b, m + 1, eigval, eigvec, 'N');
    auto energy0_new = eigval;

    if (
        ((energy0 - energy0_new) < params.error) ||
            (m == eff_ham_eff_dim) ||
            (m == params.max_iterations - 1)
        ) {
      TridiagGsSolver(a, b, m + 1, eigval, eigvec, 'V');
      energy0 = energy0_new;
      auto gs_vec = new Tensor(bases[0]->GetIndexes());
      LinearCombine(m + 1, eigvec, bases, 0.0, gs_vec);
      lancz_res.iters = m + 1;
      lancz_res.gs_eng = energy0;
      lancz_res.gs_vec = gs_vec;
      LanczosFree(eigvec, bases, m + 1, last_mat_mul_vec_res);
      MasterBroadcastOrder(lanczos_finish, world_);
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
QLTensor<TenElemT, QNT> *DMRGMPIMasterExecutor<TenElemT, QNT>::DynamicHamiltonianMultiplyState_(
    DMRGMPIMasterExecutor::Tensor &state) {
  size_t num_terms = hamiltonian_terms_.size();
#ifdef QLMPS_MPI_TIMING_MODE
  Timer broadcast_state_timer("broadcast_state_send");
#endif
  mpi::broadcast(world_, num_terms, kMasterRank);
  SendBroadCastQLTensor(world_, state, kMasterRank);
#ifdef QLMPS_MPI_TIMING_MODE
  broadcast_state_timer.PrintElapsed();
#endif

  const size_t task_num = num_terms;

  auto multiplication_res = std::vector<Tensor>(num_terms);
  auto pmultiplication_res = std::vector<Tensor *>(num_terms);
  const std::vector<TenElemT> &coefs = std::vector<TenElemT>(num_terms, TenElemT(1.0));

  for (size_t i = 0; i < num_terms; i++) {
    pmultiplication_res[i] = &multiplication_res[i];
  }
#ifdef QLMPS_TIMING_MODE
  Timer dispatch_ham_timer("first_round_dispatch_ham_send");
  dispatch_ham_timer.Suspend();
#endif

  for (size_t task_id = 0; task_id < task_num; task_id++) {
    Tensor recv_res;
    auto recv_status = recv_qlten(world_, mpi::any_source, mpi::any_tag, recv_res);
    if (recv_status.tag() < task_num) {
      multiplication_res[recv_status.tag()] = std::move(recv_res);
    }
    size_t slave_id = recv_status.source();
    world_.send(slave_id, slave_id, task_id);
    auto &block_site_terms = hamiltonian_terms_[task_id].first;
    auto &site_block_terms = hamiltonian_terms_[task_id].second;
#ifdef QLMPS_TIMING_MODE
    dispatch_ham_timer.Restart();
#endif
    SendBlockSiteHamiltonianTermGroup_(block_site_terms, slave_id);
    SendSiteBlockHamiltonianTermGroup_(site_block_terms, slave_id);
#ifdef QLMPS_TIMING_MODE
    dispatch_ham_timer.Suspend();
#endif
  }
#ifdef QLMPS_TIMING_MODE
  dispatch_ham_timer.PrintElapsed();
#endif

  const size_t termination_signal = task_num + 10086;//10086 is chosen to make a mock.
  for (size_t i = 1; i <= slave_num_; i++) {
    Tensor recv_res;
    auto recv_status = recv_qlten(world_, mpi::any_source, mpi::any_tag, recv_res);
    if (recv_status.tag() < task_num) {
      multiplication_res[recv_status.tag()] = std::move(recv_res);
    }
    size_t slave_id = recv_status.source();
    world_.send(slave_id, slave_id, termination_signal);
  }
#ifdef QLMPS_MPI_TIMING_MODE
  Timer sum_state_timer("parallel_summation_reduce");
#endif
  auto res = new Tensor();
  //TODO: optimize the summation
  LinearCombine(coefs, pmultiplication_res, TenElemT(0.0), res);
#ifdef QLMPS_MPI_TIMING_MODE
  sum_state_timer.PrintElapsed();
#endif
  return res;
}

template<typename TenElemT, typename QNT>
QLTensor<TenElemT, QNT> *DMRGMPIMasterExecutor<TenElemT, QNT>::StaticHamiltonianMultiplyState_(
    DMRGMPIMasterExecutor::Tensor &state,
    QLTEN_Double &overlap) {
  using Tensor = QLTensor<TenElemT, QNT>;
#ifdef QLMPS_MPI_TIMING_MODE
  Timer broadcast_state_timer("broadcast_state_send");
#endif
  SendBroadCastQLTensor(world_, state, kMasterRank);
#ifdef QLMPS_MPI_TIMING_MODE
  broadcast_state_timer.PrintElapsed();
#endif
  auto multiplication_res = std::vector<Tensor>(slave_num_, Tensor(state.GetIndexes()));
  auto pmultiplication_res = std::vector<Tensor *>(slave_num_);
  const std::vector<TenElemT> &coefs = std::vector<TenElemT>(slave_num_, TenElemT(1.0));
  for (size_t i = 0; i < slave_num_; i++) {
    pmultiplication_res[i] = &multiplication_res[i];
  }
  for (size_t i = 0; i < slave_num_; i++) {
    auto& bsdt = multiplication_res[i].GetBlkSparDataTen();
    bsdt.MPIRecv(world_, mpi::any_source, 10085);
  }
#ifdef QLMPS_MPI_TIMING_MODE
  Timer sum_state_timer("parallel_summation_reduce");
#endif
  auto res = new Tensor();
  LinearCombine(coefs, pmultiplication_res, TenElemT(0.0), res);
#ifdef QLMPS_MPI_TIMING_MODE
  sum_state_timer.PrintElapsed();
#endif
  for (size_t i = 1; i <= slave_num_; i++) {
    QLTEN_Double sub_overlap;
    world_.recv(mpi::any_source, 10087, sub_overlap);
    overlap += sub_overlap;
  }
  return res;
}

} /* qlmps */

#endif