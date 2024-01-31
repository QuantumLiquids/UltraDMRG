// SPDX-License-Identifier: LGPL-3.0-only

/*
* Author: Hao-Xin Wang <wanghaoxin1996@gmail.com>
* Creation Date: 2021-08-06 
*
* Description: QuantumLiquids/UltraDMRG project. Lanczos solver based on distributed memory parallel, master side.
*/

/**
@file lanczos_solver_mpi.h
@brief Lanczos solver based on distributed memory parallel.
*/

/**
 * For the note of mpi, we use //${number} to mark synchronous things 
 */

#ifndef QLMPS_ALGO_MPI_LANCZOS_SOLVER_MPI_H
#define QLMPS_ALGO_MPI_LANCZOS_SOLVER_MPI_H

#include <stdlib.h>                             // size_t
#include "qlmps/algorithm/lanczos_params.h"    // Lanczos Params
#include "boost/mpi.hpp"                        // mpi::communicator
#include "qlmps/algo_mpi/mps_algo_order.h"          // order
#include "qlten/qlten.h"                        // QLTensor

namespace qlmps {
using namespace qlten;
namespace mpi = boost::mpi;

//Forward deceleration
template<typename ElemT, typename QNT>
QLTEN_Double master_two_site_eff_ham_mul_state(
    const std::vector<QLTensor<ElemT, QNT> *> &,
    QLTensor<ElemT, QNT> *,
    QLTensor<ElemT, QNT> &,
    mpi::communicator
);

/**
Obtain the lowest energy eigenvalue and corresponding eigenstate from the effective
Hamiltonian and a initial state using Lanczos algorithm, with the help of distributed paralization.

 @param rpeff_ham Effective Hamiltonian as a vector of pointer-to-tensors.
 @param pinit_state Pointer to initial state for Lanczos iteration.
 @param params Parameters for Lanczos solver.
 @note only support two site update now.
*/
template<typename TenT>
LanczosRes<TenT> MasterLanczosSolver(
    const std::vector<TenT *> &rpeff_ham,
    TenT *pinit_state,
    const LanczosParams &params,
    mpi::communicator &world
) {
  // Take care that init_state will be destroyed after call the solver
  size_t eff_ham_eff_dim = pinit_state->size();

  const size_t eff_ham_size = pinit_state->Rank();//4
#ifdef QLMPS_TIMING_MODE
  Timer broadcast_eff_ham_timer("broadcast_eff_ham_send");
#endif
  SendBroadCastQLTensor(world, (*rpeff_ham[0]), kMasterRank);
  SendBroadCastQLTensor(world, (*rpeff_ham[eff_ham_size-1]), kMasterRank);

#ifdef QLMPS_TIMING_MODE
  broadcast_eff_ham_timer.PrintElapsed();
  Timer lancozs_after_send_ham("lancz_total_time_after_send_ham");
#endif

  LanczosRes<TenT> lancz_res;

  std::vector<TenT *> bases(params.max_iterations);
  std::vector<QLTEN_Double> a(params.max_iterations, 0.0);
  std::vector<QLTEN_Double> b(params.max_iterations, 0.0);

  // Initialize Lanczos iteration.
#ifdef QLMPS_TIMING_MODE
  Timer normalize_timer("lancz_normlize");
#endif
  pinit_state->Normalize();
  bases[0] = pinit_state;

#ifdef QLMPS_TIMING_MODE
  normalize_timer.PrintElapsed();
  Timer mat_vec_timer("lancz_mat_vec_and_overlap");
#endif
  //first time matrix multiply state will always be done, so here don't need to send order
  TenT *last_mat_mul_vec_res = new TenT();
  a[0] = master_two_site_eff_ham_mul_state(
      rpeff_ham,
      bases[0],
      *last_mat_mul_vec_res,
      world
  );

#ifdef QLMPS_TIMING_MODE
  mat_vec_timer.PrintElapsed();
#endif
  size_t m = 0;
  QLTEN_Double energy0;
  energy0 = a[0];
  // Lanczos iterations.
  while (true) {
    m += 1;
#ifdef QLMPS_TIMING_MODE
    Timer linear_combine_timer("lancz_linear_combine");
#endif
    auto gamma = last_mat_mul_vec_res;
    if (m == 1) {
      LinearCombine({-a[m - 1]}, {bases[m - 1]}, 1.0, gamma);
    } else {
      LinearCombine(
          {-a[m - 1], -b[m - 2]},
          {bases[m - 1], bases[m - 2]},
          1.0,
          gamma
      );
    }
#ifdef QLMPS_TIMING_MODE
    linear_combine_timer.PrintElapsed();
    normalize_timer.ClearAndRestart();
#endif
    auto norm_gamma = gamma->Normalize();
#ifdef QLMPS_TIMING_MODE
    normalize_timer.PrintElapsed();
    Timer trigssolver("lancz_triadiag_gs_solver_total");
    trigssolver.Suspend();
#endif
    QLTEN_Double eigval;
    QLTEN_Double *eigvec = nullptr;
    if (norm_gamma == 0.0) {
      if (m == 1) {
#ifdef QLMPS_TIMING_MODE
        Timer lancozs_post_precessing("lancz_post_processing");
#endif
        lancz_res.iters = m;
        lancz_res.gs_eng = energy0;
        lancz_res.gs_vec = new TenT(*bases[0]);
        LanczosFree(eigvec, bases, last_mat_mul_vec_res);
        MasterBroadcastOrder(lanczos_finish, world);
#ifdef QLMPS_TIMING_MODE
        lancozs_post_precessing.PrintElapsed();
        lancozs_after_send_ham.PrintElapsed();
#endif
        return lancz_res;
      } else {
#ifdef QLMPS_TIMING_MODE
        trigssolver.Restart();
#endif
        TridiagGsSolver(a, b, m, eigval, eigvec, 'V');
#ifdef QLMPS_TIMING_MODE
        trigssolver.PrintElapsed();
        Timer final_linear_combine_timer("lancz_finial_linear_combine");
#endif
        auto gs_vec = new TenT(bases[0]->GetIndexes());
        LinearCombine(m, eigvec, bases, 0.0, gs_vec);
#ifdef QLMPS_TIMING_MODE
        final_linear_combine_timer.PrintElapsed();
#endif
#ifdef QLMPS_TIMING_MODE
        Timer lancozs_post_precessing("lancz_post_processing");
#endif
        lancz_res.iters = m;
        lancz_res.gs_eng = energy0;
        lancz_res.gs_vec = gs_vec;
        LanczosFree(eigvec, bases, last_mat_mul_vec_res);
        MasterBroadcastOrder(lanczos_finish, world);
#ifdef QLMPS_TIMING_MODE
        lancozs_post_precessing.PrintElapsed();
        lancozs_after_send_ham.PrintElapsed();
#endif
        return lancz_res;
      }
    }

    b[m - 1] = norm_gamma;
    bases[m] = gamma;

#ifdef QLMPS_TIMING_MODE
    mat_vec_timer.ClearAndRestart();
#endif
    MasterBroadcastOrder(lanczos_mat_vec_dynamic, world);
    last_mat_mul_vec_res = new TenT();
    a[m] = master_two_site_eff_ham_mul_state(
        rpeff_ham,
        bases[m],
        *last_mat_mul_vec_res,
        world
    );
#ifdef QLMPS_TIMING_MODE
    mat_vec_timer.PrintElapsed();
    trigssolver.Restart();
#endif

    TridiagGsSolver(a, b, m + 1, eigval, eigvec, 'N');
#ifdef QLMPS_TIMING_MODE
    trigssolver.Suspend();
#endif
    auto energy0_new = eigval;
    if (
        ((energy0 - energy0_new) < params.error) ||
            (m == eff_ham_eff_dim) ||
            (m == params.max_iterations - 1)
        ) {
#ifdef QLMPS_TIMING_MODE
      trigssolver.Restart();
#endif
      TridiagGsSolver(a, b, m + 1, eigval, eigvec, 'V');
#ifdef QLMPS_TIMING_MODE
      trigssolver.PrintElapsed();
      Timer final_linear_combine_timer("lancz_finial_linear_combine");
#endif
      energy0 = energy0_new;
      auto gs_vec = new TenT(bases[0]->GetIndexes());
      LinearCombine(m + 1, eigvec, bases, 0.0, gs_vec);
#ifdef QLMPS_TIMING_MODE
      final_linear_combine_timer.PrintElapsed();
#endif
#ifdef QLMPS_TIMING_MODE
      Timer lancozs_post_precessing("lancz_post_processing");
#endif
      lancz_res.iters = m;
      lancz_res.gs_eng = energy0;
      lancz_res.gs_vec = gs_vec;
      LanczosFree(eigvec, bases, last_mat_mul_vec_res);
      MasterBroadcastOrder(lanczos_finish, world);
#ifdef QLMPS_TIMING_MODE
      lancozs_post_precessing.PrintElapsed();
      lancozs_after_send_ham.PrintElapsed();
#endif
      return lancz_res;
    } else {
      energy0 = energy0_new;
    }
  }
}

/**
 * two site effective hamiltonian multiplying on state, the dispatch works of master.
 * and calculation the overlap by the way
 * 
 * Once the order lanczos_mat_vec_dynamic has broadcast to Slave, master will prepare to arrage the tasks.
 * tasks from 0 to slave_num-1 will automatically done by salves at first,
 * other tasks will be sorted from difficulty to easy, to get a more balancing assignment.
 * 
 * @param eff_ham   effective hamiltonian
 * @param state     wave function
 * @param res       the result of effective hamiltonian multiple state,
 *                  as return, need to be a default tensor
 * @param world     boost::mpi::communicator
 * 
 * @return overlap of <res|(*state)>
 */
template<typename ElemT, typename QNT>
QLTEN_Double master_two_site_eff_ham_mul_state(
    const std::vector<QLTensor<ElemT, QNT> *> &eff_ham,
    QLTensor<ElemT, QNT> *state,
    QLTensor<ElemT, QNT> &res,
    mpi::communicator world
) {
  assert(res.IsDefault());
  using TenT = QLTensor<ElemT, QNT>;
#ifdef QLMPS_MPI_TIMING_MODE
  Timer broadcast_state_timer(" broadcast_state_send");
#endif
  SendBroadCastQLTensor(world, *state, kMasterRank);
#ifdef QLMPS_MPI_TIMING_MODE
  broadcast_state_timer.PrintElapsed();
#endif
  //prepare
  const size_t split_idx = 0;
  const Index<QNT> &splited_index = eff_ham[0]->GetIndexes()[split_idx];
  const size_t task_size = splited_index.GetQNSctNum();//total task number
  const QNSectorVec<QNT> &split_qnscts = splited_index.GetQNScts();
  std::vector<TenT> res_list;
  res_list.reserve(task_size);
  const size_t slave_size = world.size() - 1; //total number of slaves
  TenT res_shell = TenT(state->GetIndexes());
  std::vector<size_t> arraging_tasks(task_size);
  std::vector<size_t> task_difficuty(task_size);
  std::iota(arraging_tasks.begin(), arraging_tasks.end(), slave_size);
  for (size_t i = 0; i < task_size; i++) {
    task_difficuty[i] = split_qnscts[i].GetDegeneracy();
  }
  if (task_size > 2 * slave_size) {
    std::partial_sort(arraging_tasks.begin(),
                      arraging_tasks.begin() + slave_size,
                      arraging_tasks.end() - slave_size,
                      [&task_difficuty](size_t task1, size_t task2) {
                        return task_difficuty[task1] > task_difficuty[task2];
                      });
    //TODO: add support for multithread, note mpi::environment env( mt::multiple ) outside
    //      Also note omp order, maybe need non-block communication
    for (size_t i = 0; i < task_size; i++) {
      if (i == slave_size) { //when arraged the most large slave_size jobs, sort the other work
        std::sort(arraging_tasks.begin() + slave_size,
                  arraging_tasks.end() - slave_size,
                  [&task_difficuty](size_t task1, size_t task2) {
                    return task_difficuty[task1] > task_difficuty[task2];
                  });
        for (size_t j = i; j < task_size; j++) {
          res_list.push_back(res_shell);
        }
      } else {
        res_list.push_back(res_shell);
      }
      auto &bsdt = res_list[i].GetBlkSparDataTen();
      mpi::status recv_status = bsdt.MPIRecv(world, mpi::any_source, mpi::any_tag);
      int slave_identifier = recv_status.source();
      world.send(slave_identifier, 2 * slave_identifier, arraging_tasks[i]);
    }
  } else if (task_size > slave_size) {
    std::sort(arraging_tasks.begin(),
              arraging_tasks.end() - slave_size,
              [&task_difficuty](size_t task1, size_t task2) {
                return task_difficuty[task1] > task_difficuty[task2];
              });
    for (size_t i = 0; i < task_size; i++) {
      res_list.push_back(res_shell);
      auto &bsdt = res_list[i].GetBlkSparDataTen();
      mpi::status recv_status = bsdt.MPIRecv(world, mpi::any_source, mpi::any_tag);
      int slave_identifier = recv_status.source();
      world.send(slave_identifier, 2 * slave_identifier, arraging_tasks[i]);
    }
  } else {
    for (size_t i = 0; i < task_size; i++) {
      res_list.push_back(res_shell);
      auto &bsdt = res_list[i].GetBlkSparDataTen();
      mpi::status recv_status = bsdt.MPIRecv(world, mpi::any_source, mpi::any_tag);
      int slave_identifier = recv_status.source();
      world.send(slave_identifier, 2 * slave_identifier, arraging_tasks[i]);
    }
  }

  QLTEN_Double overlap(0.0), overlap_sub;
  for (size_t i = 0; i < std::min(task_size, slave_size); i++) {
    int slave_identifier = i + 1;
    int all_final_signal = -1;
    world.send(slave_identifier, 3 * slave_identifier + task_size, all_final_signal);
    world.recv(slave_identifier, 4 * slave_identifier + task_size, overlap_sub);
    overlap += overlap_sub;
  }

#ifdef QLMPS_MPI_TIMING_MODE
  Timer sum_state_timer(" parallel_summation_reduce");
#endif
  CollectiveLinearCombine(res_list, res);
#ifdef QLMPS_MPI_TIMING_MODE
  sum_state_timer.PrintElapsed();
#endif
  return overlap;
}

}//qlmps
#endif