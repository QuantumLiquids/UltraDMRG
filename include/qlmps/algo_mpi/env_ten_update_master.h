// SPDX-License-Identifier: LGPL-3.0-only

/*
* Author: Hao-Xin Wang <wanghaoxin1996@gmail.com>
* Creation Date: 2023-04-10
*
* Description: QuantumLiquids/UltraDMRG project. Environment tensors update function, master side.
*/

/**
 @file env_tensor_update_master.h
 @brief Environment tensors update function, master side.
*/


#ifndef QLMPS_ALGO_MPI_ENV_TENSOR_UPDATE_MASTER_H
#define QLMPS_ALGO_MPI_ENV_TENSOR_UPDATE_MASTER_H

#include "qlten/qlten.h"
#include "mps_algo_order.h"

namespace qlmps {
using namespace qlten;

//helper
inline int TaskTag(int rank) {
  return 2 * rank;
}

/** Growing left environment tensors, master function
 *
 * @param mpo   the one mpo tensor been using in this step of the update
 * @param mps   the one mps tensor been using in this step of the update
 * @return the new left environment tensor
 *
 * @note Note the returned tensor are newed in memory and need to be deleted outside the function
 * @note lenv has been sent to the workers
 */
template<typename TenElemT, typename QNT>
inline QLTensor<TenElemT, QNT> *MasterGrowLeftEnvironment(
    const QLTensor<TenElemT, QNT> &mpo,
    QLTensor<TenElemT, QNT> &mps,
    const MPI_Comm &comm
) {
  using TenT = QLTensor<TenElemT, QNT>;
  int rank, mpi_size;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &mpi_size);
#ifdef QLMPS_MPI_TIMING_MODE
  Timer broadcast_mps_timer("grow_env_broadcast_mps_send");
#endif
  mps.MPI_Bcast(rank, comm);
#ifdef QLMPS_MPI_TIMING_MODE
  broadcast_mps_timer.PrintElapsed();
#endif
  const size_t split_idx = 2; //index of mps tensor
  const Index<QNT> &splited_index = mps.GetIndexes()[split_idx];
  const int task_size = splited_index.GetQNSctNum();
  const QNSectorVec<QNT> &split_qnscts = splited_index.GetQNScts();
  std::vector<TenT> res_list;
  res_list.reserve(task_size);
  const int worker_size = mpi_size - 1;
  IndexVec<QNT> res_indexes(3);
  res_indexes[2] = splited_index;
  res_indexes[1] = mpo.GetIndexes()[3];
  res_indexes[0] = InverseIndex(splited_index);
  TenT res_shell = TenT(res_indexes);
  for (size_t j = 0; j < task_size; j++) {
    res_list.push_back(res_shell);
  }
  if (worker_size < task_size) {
    std::vector<size_t> task_difficuty(task_size);
    for (size_t i = 0; i < task_size; i++) {
      task_difficuty[i] = split_qnscts[i].GetDegeneracy();
    }
    std::vector<size_t> arraging_tasks(task_size - worker_size);
    std::iota(arraging_tasks.begin(), arraging_tasks.end(), worker_size);
#ifdef QLMPS_MPI_TIMING_MODE
    Timer sort_timer("grow_env_master_sort_task");
#endif
    std::sort(arraging_tasks.begin(),
              arraging_tasks.end(),
              [&task_difficuty](size_t task1, size_t task2) {
                return task_difficuty[task1] > task_difficuty[task2];
              });
#ifdef QLMPS_MPI_TIMING_MODE
    sort_timer.PrintElapsed();
#endif
    for (size_t i = 0; i < task_size - worker_size; i++) {
      auto &bsdt = res_list[i].GetBlkSparDataTen();
      MPI_Status status = bsdt.MPI_Recv(comm, MPI_ANY_SOURCE, MPI_ANY_TAG);
      hp_numeric::MPI_Send(arraging_tasks[i], status.MPI_SOURCE, TaskTag(status.MPI_SOURCE), comm);
    }

  }
  for (size_t i = std::max(0, task_size - worker_size); i < task_size; i++) {
    auto &bsdt = res_list[i].GetBlkSparDataTen();
    MPI_Status status = bsdt.MPI_Recv(comm, MPI_ANY_SOURCE, MPI_ANY_TAG);
    hp_numeric::MPI_Send(2 * task_size, status.MPI_SOURCE, TaskTag(status.MPI_SOURCE), comm);
  }
#ifdef QLMPS_MPI_TIMING_MODE
  Timer sum_state_timer(" parallel_summation_reduce");
#endif
  TenT *res = new TenT();
  CollectiveLinearCombine(res_list, *res);
#ifdef QLMPS_MPI_TIMING_MODE
  sum_state_timer.PrintElapsed();
#endif
  return res;
}

/**
 * dual function of MasterGrowLeftEnvironment
 */
template<typename TenElemT, typename QNT>
inline QLTensor<TenElemT, QNT> *MasterGrowRightEnvironment(
    const QLTensor<TenElemT, QNT> &mpo,
    QLTensor<TenElemT, QNT> &mps,
    const MPI_Comm &comm
) {
  int rank, mpi_size;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &mpi_size);
  using TenT = QLTensor<TenElemT, QNT>;
#ifdef QLMPS_MPI_TIMING_MODE
  Timer broadcast_mps_timer("grow_env_broadcast_mps_send");
#endif
  mps.MPI_Bcast(rank, comm);
#ifdef QLMPS_MPI_TIMING_MODE
  broadcast_mps_timer.PrintElapsed();
#endif
  const size_t split_idx = 0; //index of mps tensor
  const Index<QNT> &splited_index = mps.GetIndexes()[split_idx];
  const int task_size = splited_index.GetQNSctNum();
  const QNSectorVec<QNT> &split_qnscts = splited_index.GetQNScts();
  std::vector<TenT> res_list;
  res_list.reserve(task_size);
  const int worker_size = mpi_size - 1;
  IndexVec<QNT> res_indexes(3);
  res_indexes[2] = splited_index;
  res_indexes[1] = mpo.GetIndexes()[0];
  res_indexes[0] = InverseIndex(splited_index);
  TenT res_shell = TenT(res_indexes);
  for (size_t j = 0; j < task_size; j++) {
    res_list.push_back(res_shell);
  }
  std::cout << " Master : task_size, worker_size" << task_size << "," << worker_size << std::endl;
  MPI_Barrier(comm);
  if (worker_size < task_size) {
    // workers will firstly be accomplished the first `worker_save` tasks according their rank respectively.
    std::vector<size_t> task_difficuty(task_size);
    for (size_t i = 0; i < task_size; i++) {
      task_difficuty[i] = split_qnscts[i].GetDegeneracy();
    }
    size_t remain_task_size = task_size - worker_size;
    std::vector<size_t> arraging_tasks(remain_task_size);
    std::iota(arraging_tasks.begin(), arraging_tasks.end(), worker_size);
#ifdef QLMPS_MPI_TIMING_MODE
    Timer sort_timer("grow_env_master_sort_task");
#endif
    std::sort(arraging_tasks.begin(),
              arraging_tasks.end(),
              [&task_difficuty](size_t task1, size_t task2) {
                return task_difficuty[task1] > task_difficuty[task2];
              });
#ifdef QLMPS_MPI_TIMING_MODE
    sort_timer.PrintElapsed();
#endif
    for (size_t i = 0; i < task_size - worker_size; i++) {
      auto &bsdt = res_list[i].GetBlkSparDataTen();
      MPI_Status status = bsdt.MPI_Recv(comm, MPI_ANY_SOURCE, MPI_ANY_TAG);
      hp_numeric::MPI_Send(arraging_tasks[i], status.MPI_SOURCE, TaskTag(status.MPI_SOURCE), comm);
    }
    size_t final_signal = FinalSignal(task_size);
    for (size_t i = task_size - worker_size; i < task_size; i++) {
      auto &bsdt = res_list[i].GetBlkSparDataTen();
      std::cout << "master i : " << i << std::endl;
      MPI_Status status = bsdt.MPI_Recv(comm, MPI_ANY_SOURCE, MPI_ANY_TAG);
      std::cout << "master get here i : " << i << std::endl;
      hp_numeric::MPI_Send(final_signal, status.MPI_SOURCE, TaskTag(status.MPI_SOURCE), comm);
    }
  } else {
    size_t final_signal = FinalSignal(task_size);
    for (size_t i = 0; i < task_size; i++) {
      auto &bsdt = res_list[i].GetBlkSparDataTen();
      MPI_Status status = bsdt.MPI_Recv(comm, MPI_ANY_SOURCE, MPI_ANY_TAG);
      hp_numeric::MPI_Send(final_signal, status.MPI_SOURCE, TaskTag(status.MPI_SOURCE), comm);
    }
  }

#ifdef QLMPS_MPI_TIMING_MODE
  Timer sum_state_timer(" parallel_summation_reduce");
#endif
  TenT *res = new TenT();
  CollectiveLinearCombine(res_list, *res);
#ifdef QLMPS_MPI_TIMING_MODE
  sum_state_timer.PrintElapsed();
#endif
  return res;
}

///< used in initially generate the environment tensors.
///< because at that time workers have no env data, so master need send more data to worker.
template<typename TenElemT, typename QNT>
inline QLTensor<TenElemT, QNT> *MasterGrowLeftEnvironmentInit(
    QLTensor<TenElemT, QNT> &lenv,
    QLTensor<TenElemT, QNT> &mpo,
    QLTensor<TenElemT, QNT> &mps,
    const MPI_Comm &comm
) {
  mpo.MPI_Bcast(kMPIMasterRank, comm);
  lenv.MPI_Bcast(kMPIMasterRank, comm);
  return MasterGrowLeftEnvironment(mpo, mps, comm);
}

template<typename TenElemT, typename QNT>
inline QLTensor<TenElemT, QNT> *MasterGrowRightEnvironmentInit(
    QLTensor<TenElemT, QNT> &renv,
    QLTensor<TenElemT, QNT> &mpo,
    QLTensor<TenElemT, QNT> &mps,
    const MPI_Comm &comm
) {
  mpo.MPI_Bcast(kMPIMasterRank, comm);
  renv.MPI_Bcast(kMPIMasterRank, comm);
  return MasterGrowRightEnvironment(mpo, mps, comm);
}
}//qlmps

#endif //GRACEQ_MPS2_INCLUDE_QLMPS_ALGO_MPI_ENV_TENSOR_UPDATE_MASTER_H
