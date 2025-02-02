// SPDX-License-Identifier: LGPL-3.0-only

/*
* Author: Hao-Xin Wang <wanghaoxin1996@gmail.com>
* Creation Date: 2021-08-06
*
* Description: QuantumLiquids/UltraDMRG project. Initialize for two-site update finite size vMPS with MPI Parallel, slave side.
*/


#ifndef QLMPS_ALGO_MPI_VMPS_VMPS_MPI_INIT_SLAVE_H
#define QLMPS_ALGO_MPI_VMPS_VMPS_MPI_INIT_SLAVE_H

#include "qlten/qlten.h"
#include "qlmps/algo_mpi/mps_algo_order.h"
#include "qlmps/algo_mpi/env_ten_update_slave.h"  //SlaveGrowRightEnvironmentInit
namespace qlmps {
using namespace qlten;
template<typename TenElemT, typename QNT>
void InitEnvsSlave(const MPI_Comm &comm) {
  auto order = SlaveGetBroadcastOrder(kMPIMasterRank, comm);
  while (order != init_grow_env_finish) {
    assert(order == init_grow_env_grow);
    SlaveGrowRightEnvironmentInit<TenElemT, QNT>(comm);
    order = SlaveGetBroadcastOrder(kMPIMasterRank, comm);
  }
  return;
}
}//qlmps

#endif //QLMPS_ALGO_MPI_VMPS_VMPS_MPI_INIT_SLAVE_H
