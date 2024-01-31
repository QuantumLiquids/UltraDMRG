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
#include "boost/mpi.hpp"
#include "mps_algo_order.h"

namespace qlmps {
using namespace qlten;
namespace mpi = boost::mpi;

/** Growing left environment tensors, master function
 *
 * @tparam TenElemT
 * @tparam QNT
 * @param lenv  old left environment tensor
 * @param mpo   the one mpo tensor been using in this step of the update
 * @param mps   the one mps tensor been using in this step of the update
 * @param world
 * @return the new left environment tensor
 *
 * @note Note the returned tensor are newed in memory and need to be deleted in some place
 * @note Suppose threads number >= slave number to support omp parallel in master note
 */
template<typename TenElemT, typename QNT>
inline QLTensor<TenElemT, QNT> *MasterGrowLeftEnvironment(
    const QLTensor<TenElemT, QNT> &lenv,
    const QLTensor<TenElemT, QNT> &mpo,
    const QLTensor<TenElemT, QNT> &mps,
    mpi::communicator &world
) {
  using TenT = QLTensor<TenElemT, QNT>;
#ifdef QLMPS_MPI_TIMING_MODE
  Timer broadcast_mps_timer("grow_env_broadcast_mps_send");
#endif
  SendBroadCastQLTensor(world, mps, kMasterRank);
#ifdef QLMPS_MPI_TIMING_MODE
  broadcast_mps_timer.PrintElapsed();
#endif
  const size_t split_idx = 2; //index of mps tensor
  const Index<QNT> &splited_index = mps.GetIndexes()[split_idx];
  const size_t task_size = splited_index.GetQNSctNum();
  const QNSectorVec<QNT> &split_qnscts = splited_index.GetQNScts();
  std::vector<TenT> res_list;
  res_list.reserve(task_size);
  const size_t slave_size = world.size() - 1;
  IndexVec<QNT> res_indexes(3);
  res_indexes[2] = splited_index;
  res_indexes[1] = mpo.GetIndexes()[3];
  res_indexes[0] = InverseIndex(splited_index);
  TenT res_shell = TenT(res_indexes);
  for (size_t j = 0; j < task_size; j++) {
    res_list.push_back(res_shell);
  }
  if (slave_size < task_size) {
    std::vector<size_t> task_difficuty(task_size);
    for (size_t i = 0; i < task_size; i++) {
      task_difficuty[i] = split_qnscts[i].GetDegeneracy();
    }
    std::vector<size_t> arraging_tasks(task_size - slave_size);
    std::iota(arraging_tasks.begin(), arraging_tasks.end(), slave_size);
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
    /* ** parallel communication
      #pragma omp parallel default(none)\
                          shared(task_size, slave_size, res_list, world, arraging_tasks)\
                          num_threads(slave_size)
      {
        size_t controlling_slave = omp_get_thread_num()+1;

        auto& bsdt = res_list[controlling_slave-1].GetBlkSparDataTen();
        const size_t task = controlling_slave-1;
        mpi::status recv_status = bsdt.MPIRecv(world, controlling_slave, task);

        #pragma omp for nowait schedule(dynamic)
        for(size_t i = 0; i < task_size - slave_size; i++){
          world.send(controlling_slave, 2*controlling_slave, arraging_tasks[i]);
          auto& bsdt = res_list[i+slave_size].GetBlkSparDataTen();
          bsdt.MPIRecv(world, controlling_slave, arraging_tasks[i]);
        }

        world.send(controlling_slave, 2*controlling_slave, 2*task_size);//finish signal
      }
    */
    for (size_t i = 0; i < task_size - slave_size; i++) {
      auto &bsdt = res_list[i].GetBlkSparDataTen();
      mpi::status recv_status = bsdt.MPIRecv(world, mpi::any_source, mpi::any_tag);
      int slave_identifier = recv_status.source();
      world.send(slave_identifier, 2 * slave_identifier, arraging_tasks[i]);
    }
    for (size_t i = task_size - slave_size; i < task_size; i++) {
      auto &bsdt = res_list[i].GetBlkSparDataTen();
      mpi::status recv_status = bsdt.MPIRecv(world, mpi::any_source, mpi::any_tag);
      int slave_identifier = recv_status.source();
      world.send(slave_identifier, 2 * slave_identifier, 2 * task_size);//finish signal
    }
  } else {//slave_size >= task_size
    for (size_t i = 0; i < task_size; i++) {
      auto &bsdt = res_list[i].GetBlkSparDataTen();
      mpi::status recv_status = bsdt.MPIRecv(world, mpi::any_source, mpi::any_tag);
      int slave_identifier = recv_status.source();
      world.send(slave_identifier, 2 * slave_identifier, 2 * task_size);
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

///< Growing right environment tensors, master function.
///< Refer to `MasterGrowLeftEnvironment` to find more details.
template<typename TenElemT, typename QNT>
inline QLTensor<TenElemT, QNT> *MasterGrowRightEnvironment(
    const QLTensor<TenElemT, QNT> &renv,
    const QLTensor<TenElemT, QNT> &mpo,
    const QLTensor<TenElemT, QNT> &mps,
    mpi::communicator &world
) {
  using TenT = QLTensor<TenElemT, QNT>;
#ifdef QLMPS_MPI_TIMING_MODE
  Timer broadcast_mps_timer("grow_env_broadcast_mps_send");
#endif
  SendBroadCastQLTensor(world, mps, kMasterRank);
#ifdef QLMPS_MPI_TIMING_MODE
  broadcast_mps_timer.PrintElapsed();
#endif
  const size_t split_idx = 0; //index of mps tensor
  const Index<QNT> &splited_index = mps.GetIndexes()[split_idx];
  const size_t task_size = splited_index.GetQNSctNum();
  const QNSectorVec<QNT> &split_qnscts = splited_index.GetQNScts();
  std::vector<TenT> res_list;
  res_list.reserve(task_size);
  const size_t slave_size = world.size() - 1;
  IndexVec<QNT> res_indexes(3);
  res_indexes[2] = splited_index;
  res_indexes[1] = mpo.GetIndexes()[0];
  res_indexes[0] = InverseIndex(splited_index);
  TenT res_shell = TenT(res_indexes);
  for (size_t j = 0; j < task_size; j++) {
    res_list.push_back(res_shell);
  }
  if (slave_size < task_size) {
    // slave will firstly accomplish the first `slave_save` tasks according their id (rank) respectively.
    std::vector<size_t> task_difficuty(task_size);
    for (size_t i = 0; i < task_size; i++) {
      task_difficuty[i] = split_qnscts[i].GetDegeneracy();
    }
    size_t remain_task_size = task_size - slave_size;
    std::vector<size_t> arraging_tasks(remain_task_size);
    std::iota(arraging_tasks.begin(), arraging_tasks.end(), slave_size);
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
    /*
      #pragma omp parallel default(none)\
                          shared(task_size, slave_size, res_list, world, arraging_tasks)\
                          num_threads(slave_size)
      {
        size_t controlling_slave = omp_get_thread_num()+1;

        auto& bsdt = res_list[controlling_slave-1].GetBlkSparDataTen();
        const size_t task = controlling_slave-1;
        mpi::status recv_status = bsdt.MPIRecv(world, controlling_slave, task);

        #pragma omp for nowait schedule(dynamic)
        for(size_t i = 0; i < task_size - slave_size; i++){
          world.send(controlling_slave, 2*controlling_slave, arraging_tasks[i]);
          auto& bsdt = res_list[i+slave_size].GetBlkSparDataTen();
          bsdt.MPIRecv(world, controlling_slave, arraging_tasks[i]);
        }

        world.send(controlling_slave, 2*controlling_slave, 2*task_size);//finish signal
      }
    */
    /*    std::cout << "remain_task = " << remain_task_size << std::endl;
    for (size_t i = 0; i < task_size; i++) {
      std::cout << "." ;
      auto &bsdt = res_list[i].GetBlkSparDataTen();
      mpi::status recv_status = bsdt.MPIRecv(world, mpi::any_source, mpi::any_tag);
      if(i < remain_task_size) {
        int slave_id = recv_status.source();
        world.send(slave_id, 2 * slave_id, arraging_tasks[i]);
      }
    }
    for (size_t slave = 1; slave < slave_size; slave++) {
      std::cout << "." << std::endl;
      world.send(slave, 2 * slave, 2 * task_size);//finish signal
    } */
    for (size_t i = 0; i < task_size - slave_size; i++) {
      auto &bsdt = res_list[i].GetBlkSparDataTen();
      mpi::status recv_status = bsdt.MPIRecv(world, mpi::any_source, mpi::any_tag);
      int slave_identifier = recv_status.source();
      world.send(slave_identifier, 2 * slave_identifier, arraging_tasks[i]);
    }
    for (size_t i = task_size - slave_size; i < task_size; i++) {
      auto &bsdt = res_list[i].GetBlkSparDataTen();
      mpi::status recv_status = bsdt.MPIRecv(world, mpi::any_source, mpi::any_tag);
      int slave_identifier = recv_status.source();
      world.send(slave_identifier, 2 * slave_identifier, 2 * task_size);//finish signal
    }
  } else {//slave_size >= task_size
    for (size_t i = 0; i < task_size; i++) {
      auto &bsdt = res_list[i].GetBlkSparDataTen();
      mpi::status recv_status = bsdt.MPIRecv(world, mpi::any_source, mpi::any_tag);
      int slave_identifier = recv_status.source();
      world.send(slave_identifier, 2 * slave_identifier, 2 * task_size);
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
///< because at that time slave has no env data, so master need send more data to slave.
template<typename TenElemT, typename QNT>
inline QLTensor<TenElemT, QNT> *MasterGrowLeftEnvironmentInit(
    const QLTensor<TenElemT, QNT> &lenv,
    const QLTensor<TenElemT, QNT> &mpo,
    const QLTensor<TenElemT, QNT> &mps,
    mpi::communicator &world
) {
  SendBroadCastQLTensor(world, mpo, kMasterRank);
  SendBroadCastQLTensor(world, lenv, kMasterRank);
  return MasterGrowLeftEnvironment(lenv, mpo, mps, world);
}

template<typename TenElemT, typename QNT>
inline QLTensor<TenElemT, QNT> *MasterGrowRightEnvironmentInit(
    const QLTensor<TenElemT, QNT> &renv,
    const QLTensor<TenElemT, QNT> &mpo,
    const QLTensor<TenElemT, QNT> &mps,
    mpi::communicator &world
) {
  SendBroadCastQLTensor(world, mpo, kMasterRank);
  SendBroadCastQLTensor(world, renv, kMasterRank);
  return MasterGrowRightEnvironment(renv, mpo, mps, world);
}
}//qlmps

#endif //GRACEQ_MPS2_INCLUDE_QLMPS_ALGO_MPI_ENV_TENSOR_UPDATE_MASTER_H
