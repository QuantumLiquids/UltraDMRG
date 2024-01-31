// SPDX-License-Identifier: LGPL-3.0-only

/*
* Author: Hao-Xin Wang <wanghaoxin1996@gmail.com>
* Creation Date: 2023-04-10
*
* Description: QuantumLiquids/UltraDMRG project. Environment tensors update function, slave side.
*/

/**
 @file env_tensor_update_slave.h
 @brief Environment tensors update function, slave side.
*/
#ifndef QLMPS_ALGO_MPI_ENV_TENSOR_UPDATE_SLAVE_H
#define QLMPS_ALGO_MPI_ENV_TENSOR_UPDATE_SLAVE_H

#include "qlten/qlten.h"
#include "boost/mpi.hpp"
#include "mps_algo_order.h"

namespace qlmps {
using namespace qlten;
namespace mpi = boost::mpi;

/** Growing left environment tensors, slave function
 *
 * @tparam TenElemT
 * @tparam QNT
 * @param lenv old left environment tensor
 * @param mpo the one mpo tensor been using in this step of the update
 * @param world
 *
 * Mps tensor will be received from master side.
 * The results will be gathered at master.
 */
template<typename TenElemT, typename QNT>
inline void SlaveGrowLeftEnvironment(
    const QLTensor<TenElemT, QNT> &lenv,
    const QLTensor<TenElemT, QNT> &mpo,
    mpi::communicator &world
) {
  using TenT = QLTensor<TenElemT, QNT>;
  TenT mps;
#ifdef QLMPS_MPI_TIMING_MODE
  Timer broadcast_mps_timer("grow_env_broadcast_mps_recv");
#endif
  RecvBroadCastQLTensor(world, mps, kMasterRank);
#ifdef QLMPS_MPI_TIMING_MODE
  broadcast_mps_timer.PrintElapsed();
#endif
  const size_t split_idx = 2; //index of mps tensor
  const Index<QNT> &splited_index = mps.GetIndexes()[split_idx];
  const size_t task_size = splited_index.GetQNSctNum();
  TenT mps_dag = Dag(mps);
  size_t task_count = 0;
  const size_t slave_id = world.rank();//number from 1
  if (slave_id > task_size) {
#ifdef QLMPS_MPI_TIMING_MODE
    std::cout << "Slave has done task_count = " << task_count << std::endl;
#endif
    return;
  }
#ifdef QLMPS_MPI_TIMING_MODE
  Timer salve_communication_timer(" slave " + std::to_string(slave_id) + "'s communication");
  salve_communication_timer.Suspend();
  Timer slave_work_timer(" slave " + std::to_string(slave_id) + "'s work");
#endif
  //first task
  size_t task = slave_id - 1;
  TenT env_times_mps;
  TenT temp, res;
  //First contract
  TensorContraction1SectorExecutor<TenElemT, QNT> ctrct_executor(
      &mps,
      split_idx,
      task,
      &lenv,
      {{0}, {2}},
      &env_times_mps
  );
  ctrct_executor.Execute();

  Contract<TenElemT, QNT, true, true>(env_times_mps, mpo, 3, 0, 2, temp);
  env_times_mps.GetBlkSparDataTen().Clear();
  Contract<TenElemT, QNT, false, true>(mps_dag, temp, 0, 1, 2, res);
  temp.GetBlkSparDataTen().Clear();
  auto &bsdt = res.GetBlkSparDataTen();
  task_count++;
#ifdef QLMPS_MPI_TIMING_MODE
  salve_communication_timer.Restart();
#endif
  bsdt.MPISend(world, kMasterRank, task);//tag = task
  world.recv(kMasterRank, 2 * slave_id, task);//tag = 2*slave_id
#ifdef QLMPS_MPI_TIMING_MODE
  salve_communication_timer.Suspend();
#endif
  while (task < task_size) {
    TenT temp, res;
    ctrct_executor.SetSelectedQNSect(task);
    ctrct_executor.Execute();
    Contract<TenElemT, QNT, true, true>(env_times_mps, mpo, 3, 0, 2, temp);
    env_times_mps.GetBlkSparDataTen().Clear();
    Contract<TenElemT, QNT, false, true>(mps_dag, temp, 0, 1, 2, res);
    auto &bsdt = res.GetBlkSparDataTen();
    task_count++;
#ifdef QLMPS_MPI_TIMING_MODE
    salve_communication_timer.Restart();
#endif
    bsdt.MPISend(world, kMasterRank, task);//tag = task
    world.recv(kMasterRank, 2 * slave_id, task);
#ifdef QLMPS_MPI_TIMING_MODE
    salve_communication_timer.Suspend();
#endif
  }
#ifdef QLMPS_MPI_TIMING_MODE
  slave_work_timer.PrintElapsed();
  salve_communication_timer.PrintElapsed();
  std::cout << "Slave " << slave_id << " has done task_count = " << task_count << std::endl;
#endif
}

///< Growing right environment tensors, slave function.
///< Refer to `SlaveGrowLeftEnvironment` to find more details.
template<typename TenElemT, typename QNT>
inline void SlaveGrowRightEnvironment(
    const QLTensor<TenElemT, QNT> &renv,
    const QLTensor<TenElemT, QNT> &mpo,
    mpi::communicator &world
) {
  using TenT = QLTensor<TenElemT, QNT>;
  TenT mps;
#ifdef QLMPS_MPI_TIMING_MODE
  Timer broadcast_mps_timer("grow_env_broadcast_mps_recv");
#endif
  RecvBroadCastQLTensor(world, mps, kMasterRank);
#ifdef QLMPS_MPI_TIMING_MODE
  broadcast_mps_timer.PrintElapsed();
#endif
  const size_t split_idx = 0; //index of mps tensor
  const Index<QNT> &splited_index = mps.GetIndexes()[split_idx];
  const size_t task_size = splited_index.GetQNSctNum();
  TenT mps_dag = Dag(mps);
  size_t task_count = 0;
  const size_t slave_id = world.rank();//number from 1
  if (slave_id > task_size) {
#ifdef QLMPS_MPI_TIMING_MODE
    std::cout << "Slave has done task_count = " << task_count << std::endl;
#endif
    return;
  }
#ifdef QLMPS_MPI_TIMING_MODE
  Timer salve_communication_timer(" slave " + std::to_string(slave_id) + "'s communication");
  salve_communication_timer.Suspend();
  Timer slave_work_timer(" slave " + std::to_string(slave_id) + "'s work");
#endif
  //first task
  size_t task = slave_id - 1;
  TenT env_times_mps;
  TenT temp, res;
  //First contract
  TensorContraction1SectorExecutor<TenElemT, QNT> ctrct_executor(
      &mps_dag,
      split_idx,
      task,
      &renv,
      {{2}, {0}},
      &env_times_mps
  );
  ctrct_executor.Execute();
  Contract<TenElemT, QNT, true, false>(env_times_mps, mpo, 1, 2, 2, temp);
  env_times_mps.GetBlkSparDataTen().Clear();
  Contract<TenElemT, QNT, true, false>(temp, mps, 3, 1, 2, res);
  temp.GetBlkSparDataTen().Clear();
  auto &bsdt = res.GetBlkSparDataTen();
  task_count++;
#ifdef QLMPS_MPI_TIMING_MODE
  salve_communication_timer.Restart();
#endif
  bsdt.MPISend(world, kMasterRank, task);//tag = task
  world.recv(kMasterRank, 2 * slave_id, task);//tag = 2*slave_id
#ifdef QLMPS_MPI_TIMING_MODE
  salve_communication_timer.Suspend();
#endif
  while (task < task_size) {
    TenT temp, res;
    ctrct_executor.SetSelectedQNSect(task);
    ctrct_executor.Execute();
    Contract<TenElemT, QNT, true, false>(env_times_mps, mpo, 1, 2, 2, temp);
    env_times_mps.GetBlkSparDataTen().Clear();
    Contract<TenElemT, QNT, true, false>(temp, mps, 3, 1, 2, res);
    auto &bsdt = res.GetBlkSparDataTen();
    task_count++;
#ifdef QLMPS_MPI_TIMING_MODE
    salve_communication_timer.Restart();
#endif
    bsdt.MPISend(world, kMasterRank, task);//tag = task
    world.recv(kMasterRank, 2 * slave_id, task);
#ifdef QLMPS_MPI_TIMING_MODE
    salve_communication_timer.Suspend();
#endif
  }
#ifdef QLMPS_MPI_TIMING_MODE
  slave_work_timer.PrintElapsed();
  salve_communication_timer.PrintElapsed();
  std::cout << "Slave " << slave_id << " has done task_count = " << task_count << std::endl;
#endif
}

///< used in initially generate the environment tensors, because at that time slave has no env data.
template<typename TenElemT, typename QNT>
inline void SlaveGrowLeftEnvironmentInit(mpi::communicator &world) {
  QLTensor<TenElemT, QNT> mpo;
  QLTensor<TenElemT, QNT> lenv;
  RecvBroadCastQLTensor(world, mpo, kMasterRank);
  RecvBroadCastQLTensor(world, lenv, kMasterRank);
  SlaveGrowLeftEnvironment(lenv, mpo, world);
}

template<typename TenElemT, typename QNT>
inline void SlaveGrowRightEnvironmentInit(mpi::communicator &world) {
  QLTensor<TenElemT, QNT> mpo;
  QLTensor<TenElemT, QNT> renv;
  RecvBroadCastQLTensor(world, mpo, kMasterRank);
  RecvBroadCastQLTensor(world, renv, kMasterRank);
  SlaveGrowRightEnvironment(renv, mpo, world);
}
}//qlmps
#endif //QLMPS_ALGO_MPI_ENV_TENSOR_UPDATE_SLAVE_H
