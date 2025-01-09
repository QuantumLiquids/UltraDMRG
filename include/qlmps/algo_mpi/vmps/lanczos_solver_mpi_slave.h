/// SPDX-License-Identifier: LGPL-3.0-only

/*
* Author: Hao-Xin Wang <wanghaoxin1996@gmail.com>
* Creation Date: 2023-04-1
*
* Description: QuantumLiquids/UltraDMRG project. Lanczos solver based on distributed memory parallel, slave side.
*/


#ifndef QLMPS_ALGO_MPI_LANCZOS_SOLVER_MPI_SLAVE_H
#define QLMPS_ALGO_MPI_LANCZOS_SOLVER_MPI_SLAVE_H

#include <cstdlib>                             // size_t
#include "qlmps/algorithm/lanczos_params.h"    // Lanczos Params
#include "qlmps/algo_mpi/mps_algo_order.h"     // order
#include "qlten/qlten.h"                       // QLTensor

namespace qlmps {
using namespace qlten;

template<typename ElemT, typename QNT>
void slave_two_site_eff_ham_mul_state(
    const std::vector<QLTensor<ElemT, QNT> *> &,
    const MPI_Comm &
);

/**
 * SlaveLanczosSolver
 * Receive the effective Hamiltonian, and do the multiplications until master terminates.
 *
 * @note effective Hamiltonian tensors are not deleted after calling this function.
 * @note deceleration the typename TenT when call this function
*/
template<typename TenT>
void SlaveLanczosSolver(
    std::vector<TenT *> &rpeff_ham,
    const MPI_Comm &comm
) {
// Receive Hamiltonian
#ifdef GQMPS_MPI_TIMING_MODE
  Timer broadcast_eff_ham_timer("broadcast_eff_ham_recv");
#endif
  MPI_Bcast(*rpeff_ham[0], kMPIMasterRank, comm);
  MPI_Bcast(*rpeff_ham[two_site_eff_ham_size - 1], kMPIMasterRank, comm);
#ifdef GQMPS_MPI_TIMING_MODE
  broadcast_eff_ham_timer.PrintElapsed();
#endif

  MPS_AlGO_ORDER order = lanczos_mat_vec_dynamic;
  while (order == lanczos_mat_vec_dynamic) {
    slave_two_site_eff_ham_mul_state(rpeff_ham, comm);
    order = SlaveGetBroadcastOrder(kMPIMasterRank, comm);
  }
  assert(order == lanczos_finish);

  return;
}

/**
 * two site effective hamiltonian multiplying on state,
 * split index contract tasks worked on slave. The works are controlled by master.
 *
 * every slave should have a copy of all of the eff_ham before call this function.
 * slave do not need prepare the state.
 *
 * @param eff_ham   effective hamiltonian
 * @param state     wave function
 *
 * @return the result of effective hamiltonian multiple
 */
template<typename ElemT, typename QNT>
void slave_two_site_eff_ham_mul_state(
    const std::vector<QLTensor<ElemT, QNT> *> &eff_ham,
    const MPI_Comm &comm
) {
  int rank;
  MPI_Comm_rank(comm, &rank);
  const size_t worker_rank = rank;
  using TenT = QLTensor<ElemT, QNT>;
  TenT *state = new TenT();
  TenT temp_scalar_ten;
#ifdef QLMPS_MPI_TIMING_MODE
  Timer broadcast_state_timer(" broadcast_state_recv");
#endif
  MPI_Bcast(*state, kMPIMasterRank, comm);
#ifdef QLMPS_MPI_TIMING_MODE
  broadcast_state_timer.PrintElapsed();
#endif
  auto base_dag = Dag(*state);
  // Timer slave_prepare_timer(" slave "+ std::to_string(rank) +"'s prepare");
  const size_t split_idx = 0;
  const Index<QNT> &splited_index = eff_ham[0]->GetIndexes()[split_idx];
  const size_t task_num = splited_index.GetQNSctNum();
  //slave also need to know the total task number used to judge if finish this works
  size_t task_count = 0;
  if (worker_rank > task_num) {
    //no task, happy~
#ifdef QLMPS_MPI_TIMING_MODE
    std::cout << "Slave has done task_count = " << task_count << std::endl;
#endif
    delete state;
    return;
  }
  // slave_prepare_timer.PrintElapsed();
  //$1
#ifdef QLMPS_MPI_TIMING_MODE
  Timer salve_communication_timer(" slave "+std::to_string(worker_rank) +"'s communication");
  salve_communication_timer.Suspend();
  Timer slave_work_timer(" slave "+ std::to_string(worker_rank) +"'s work");
#endif
  //first task
  size_t task = worker_rank - 1;
  TenT eff_ham0_times_state;
  TenT temp1, res;
  //First contract
  TensorContraction1SectorExecutor<ElemT, QNT> ctrct_executor(
      eff_ham[0],
      split_idx,
      task,
      state,
      {{2}, {0}},
      &eff_ham0_times_state
  );

  ctrct_executor.Execute();

  Contract<ElemT, QNT, true, true>(eff_ham0_times_state, *eff_ham[1], 1, 0, 2, res);
  eff_ham0_times_state.GetBlkSparDataTen().Clear();// save for memory
  Contract<ElemT, QNT, true, true>(res, *eff_ham[2], 4, 0, 2, temp1);
  Contract<ElemT, QNT, true, false>(temp1, *eff_ham[3], 4, 1, 2, res);
  temp1.GetBlkSparDataTen().Clear();

  Contract(
      &res, &base_dag,
      {{0, 1, 2, 3}, {0, 1, 2, 3}},
      &temp_scalar_ten
  );
  QLTEN_Double overlap = Real(temp_scalar_ten());
  //$2
  // send_qlten(rank, kMPIMasterRank, task, res);//tag = task
  auto &bsdt = res.GetBlkSparDataTen();
  // std::cout << "task " << task << " finished, sending " << std::endl;
  task_count++;
#ifdef QLMPS_MPI_TIMING_MODE
  salve_communication_timer.Restart();
#endif
  bsdt.MPI_Send(comm, kMPIMasterRank, task);//tag = task
  hp_numeric::MPI_Recv(task, kMPIMasterRank, 2 * worker_rank, comm);
#ifdef QLMPS_MPI_TIMING_MODE
  salve_communication_timer.Suspend();
#endif
  while (task < task_num) {
    TenT temp1, res, temp_scalar_ten;
    ctrct_executor.SetSelectedQNSect(task);
    ctrct_executor.Execute();

    Contract<ElemT, QNT, true, true>(eff_ham0_times_state, *eff_ham[1], 1, 0, 2, res);
    eff_ham0_times_state.GetBlkSparDataTen().Clear();// save for memory
    Contract<ElemT, QNT, true, true>(res, *eff_ham[2], 4, 0, 2, temp1);
    Contract<ElemT, QNT, true, false>(temp1, *eff_ham[3], 4, 1, 2, res);
    temp1.GetBlkSparDataTen().Clear();

    Contract(
        &res, &base_dag,
        {{0, 1, 2, 3}, {0, 1, 2, 3}},
        &temp_scalar_ten
    );
    overlap += Real(temp_scalar_ten());

    auto &bsdt = res.GetBlkSparDataTen();
    task_count++;
#ifdef QLMPS_MPI_TIMING_MODE
    salve_communication_timer.Restart();
#endif
    bsdt.MPI_Send(comm, kMPIMasterRank, task);//tag = task
    hp_numeric::MPI_Recv(task, kMPIMasterRank, 2 * worker_rank, comm);
#ifdef QLMPS_MPI_TIMING_MODE
    salve_communication_timer.Suspend();
#endif
  }
#ifdef QLMPS_MPI_TIMING_MODE
  slave_work_timer.PrintElapsed();
  salve_communication_timer.PrintElapsed();
  std::cout << "Slave " << worker_rank<< " has done task_count = " << task_count << std::endl;
#endif
  delete state;
  size_t all_final_signal;
  hp_numeric::MPI_Recv(all_final_signal, kMPIMasterRank, 3 * worker_rank + task_num, comm);
  assert(all_final_signal == FinalSignal(task_num));
  HANDLE_MPI_ERROR(::MPI_Send(&overlap, 1, MPI_DOUBLE, kMPIMasterRank, 4 * worker_rank + task_num, comm));
}

}//qlmps


#endif //QLMPS_ALGO_MPI_LANCZOS_SOLVER_MPI_SLAVE_H
