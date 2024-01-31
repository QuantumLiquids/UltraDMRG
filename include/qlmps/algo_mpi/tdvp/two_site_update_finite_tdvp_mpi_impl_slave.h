// SPDX-License-Identifier: LGPL-3.0-only

/*
* Author: Hao-Xin Wang <wanghaoxin1996@gmail.com>
* Creation Date: 2023-04-11
*
* Description: QuantumLiquids/UltraDMRG project. Two-site update finite size TDVP with MPI Parallel, slave side.
*/


#ifndef QLMPS_ALGO_MPI_TDVP_TWO_SITE_UPDATE_FINITE_TDVP_MPI_IMPL_SLAVE_H
#define QLMPS_ALGO_MPI_TDVP_TWO_SITE_UPDATE_FINITE_TDVP_MPI_IMPL_SLAVE_H

#include "qlmps/algo_mpi/mps_algo_order.h"                            //kMasterRank, Order...
#include "qlmps/algo_mpi/tdvp/two_site_update_finite_tdvp_mpi.h" //MPITDVPSweepParams
#include "qlmps/algorithm/tdvp/tdvp_evolve_params.h"    //DynamicMeasuRes..
#include "lanczos_expmv_solver_mpi.h"             //MasterLanczosExpmvSolver, SlaveLanczosExpmvSolver
#include "qlmps/algo_mpi/env_tensor_update_slave.h"

namespace qlmps {
using namespace qlten;
namespace mpi = boost::mpi;

template<typename TenElemT, typename QNT>
void SlaveTwoSiteFiniteTDVP(
    const MPO<QLTensor<TenElemT, QNT>> &mpo,
    mpi::communicator &world
) {
  using TenT = QLTensor<TenElemT, QNT>;
  std::vector<TenT *> eff_ham(two_site_eff_ham_size);

  MPS_AlGO_ORDER order = program_start;
  while (order != program_final) {
    order = SlaveGetBroadcastOrder(world);
    switch (order) {
      case program_start:world.send(kMasterRank, 2 * world.rank(), world.rank());
        break;
      case lanczos: {
        size_t lsite_idx;
        broadcast(world, lsite_idx, kMasterRank);
        size_t rsite_idx = lsite_idx + 1;
        eff_ham[0] = new TenT();
        eff_ham[1] = const_cast<TenT *>(&mpo[lsite_idx]);
        eff_ham[2] = const_cast<TenT *>(&mpo[rsite_idx]);
        eff_ham[two_site_eff_ham_size-1] = new TenT();
        SlaveLanczosSolver<TenT>(eff_ham, world);
      }
        break;
      case svd: {
        MPISVDSlave<TenElemT>(world);
      }
        break;
      case contract_for_right_moving_expansion: {//dir='r'
        std::cout << "Slave doesn't have the functionality of contract_for_right_moving_expansion. Aborting."
                  << std::endl;
        world.abort(1);
      }
        break;
      case contract_for_left_moving_expansion: {//dir='l'
        std::cout << "Slave doesn't have the functionality of contract_for_left_moving_expansion. Aborting."
                  << std::endl;
        world.abort(1);
      }
        break;
      case growing_left_env: {
        delete eff_ham[two_site_eff_ham_size-1];
        SlaveGrowLeftEnvironment(*eff_ham[0], *eff_ham[1], world);
        delete eff_ham[0];
      }
        break;
      case growing_right_env: {
        delete eff_ham[0];
        SlaveGrowRightEnvironment(*eff_ham[3], *eff_ham[2], world);
        delete eff_ham[two_site_eff_ham_size-1];
      }
        break;
      case program_final:std::cout << "Slave" << world.rank() << " will stop." << std::endl;
        break;
      default:std::cout << "Slave " << world.rank() << " doesn't understand the order " << order << std::endl;
        break;
    }
  }
}
}

#endif //QLMPS_ALGO_MPI_TDVP_TWO_SITE_UPDATE_FINITE_TDVP_MPI_IMPL_SLAVE_H
