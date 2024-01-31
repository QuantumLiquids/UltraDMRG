// SPDX-License-Identifier: LGPL-3.0-only

/*
* Author: Hao-Xin Wang <wanghaoxin1996@gmail.com>
* Creation Date: 2021-08-11
*
* Description: QuantumLiquids/UltraDMRG project. Basic set up for parallel VMPS and TDVP.
*/

/**
@file  mps_algo_order.h
@brief  Basic set up for parallel VMPS and TDVP.
*/

#ifndef QLMPS_ALGO_MPI_MPS_ALGO_ORDER_H
#define QLMPS_ALGO_MPI_MPS_ALGO_ORDER_H

#include "qlten/qlten.h"
#include "boost/mpi.hpp"

namespace qlmps {
using namespace qlten;
const size_t kMasterRank = 0;

///< variational mps orders send by master
enum MPS_AlGO_ORDER {
  program_start,        ///< when vmps start
  init_grow_env,        ///< if need to grow env before the first sweep
  init_grow_env_grow,   ///< when grow env when initially growing env
  init_grow_env_finish,  ///< when the growing env works before the first sweep finished.
  lanczos,              ///< when lanczos start
  svd,                  ///< before svd
  lanczos_mat_vec_dynamic, ///< before do lanczos' matrix vector multiplication, dynamic schedule the tasks
  lanczos_mat_vec_static,  ///< before do lanczos' matrix vector multiplication, schedule according to the previous tasks
  lanczos_finish,       ///< when lanczos finished
  contract_for_right_moving_expansion, ///< contraction and fuse index operations in expansion when right moving
  contract_for_left_moving_expansion, ///< contraction and fuse index operations in expansion when left moving
  growing_left_env,     ///< growing left environment
  growing_right_env,    ///< growing right environment
  program_final         /// when vmps finished
};

const size_t two_site_eff_ham_size = 4;
namespace mpi = boost::mpi;

inline void MasterBroadcastOrder(const MPS_AlGO_ORDER order,
                                 mpi::communicator &world) {
  mpi::broadcast(world, const_cast<MPS_AlGO_ORDER &>(order), kMasterRank);
}

inline MPS_AlGO_ORDER SlaveGetBroadcastOrder(mpi::communicator world) {
  MPS_AlGO_ORDER order;
  mpi::broadcast(world, order, kMasterRank);
  return order;
}

}//qlmps

#endif //QLMPS_ALGO_MPI_MPS_ALGO_ORDER_H