// SPDX-License-Identifier: LGPL-3.0-only

/*
* Author: Hao-Xin Wang <wanghaoxin1996@gmail.com>
* Creation Date: 2023-04-10
*
* Description: QuantumLiquids/UltraDMRG project. Two-site update finite size vMPS with MPI Parallel, slave nodes.
*/

#ifndef QLMPS_ALGO_MPI_VMPS_TWO_SITE_UPDATE_FINITE_VMPS_MPI_IMPL_SLAVE_H
#define QLMPS_ALGO_MPI_VMPS_TWO_SITE_UPDATE_FINITE_VMPS_MPI_IMPL_SLAVE_H
#include "qlten/qlten.h"
#include "qlmps/algorithm/lanczos_params.h"                        //LanczosParams
#include "qlmps/algorithm/finite_vmps_sweep_params.h"
#include "qlmps/algo_mpi/mps_algo_order.h"                              //VMPSORDER
#include "qlmps/algo_mpi/env_ten_update_slave.h"                //MasterGrowLeftEnvironment, MasterGrowRightEnvironment
#include "qlmps/algo_mpi/vmps/vmps_mpi_init_slave.h"               //InitEnvsSlave
#include "qlmps/algo_mpi/vmps/two_site_update_finite_vmps_mpi.h"   //TwoSiteMPIVMPSSweepParams
#include "lanczos_solver_mpi_slave.h"
namespace qlmps {
using namespace qlten;

//forward declaration
template<typename TenElemT, typename QNT>
void SlaveTwoSiteFiniteVMPSRightMovingExpand(
    const std::vector<QLTensor<TenElemT, QNT> *> &,
    const MPI_Comm &
);

template<typename TenElemT, typename QNT>
void SlaveTwoSiteFiniteVMPSLeftMovingExpand(
    const std::vector<QLTensor<TenElemT, QNT> *> &,
    const MPI_Comm &
);

template<typename TenElemT, typename QNT>
void SlaveTwoSiteFiniteVMPS(
    const MPO<QLTensor<TenElemT, QNT>> &mpo,
    const MPI_Comm &comm
) {
  using TenT = QLTensor<TenElemT, QNT>;
  int rank;
  MPI_Comm_rank(comm, &rank);
  size_t node_num = rank;
  //global variables, and please careful the memory controlling for these variables.
  std::vector<TenT *> eff_ham(two_site_eff_ham_size);

  MPS_AlGO_ORDER order = program_start;
  while (order != program_final) {
    order = SlaveGetBroadcastOrder(kMPIMasterRank, comm);
    switch (order) {
      case program_start:hp_numeric::MPI_Send(node_num, kMPIMasterRank, 2 * rank, comm);
        break;
      case init_grow_env:InitEnvsSlave<TenElemT, QNT>(comm);
        break;
      case lanczos: {
        size_t lsite_idx;
        HANDLE_MPI_ERROR(::MPI_Bcast(&lsite_idx, 1, MPI_UNSIGNED_LONG_LONG, kMPIMasterRank, comm));
        size_t rsite_idx = lsite_idx + 1;
        eff_ham[0] = new TenT();
        eff_ham[1] = const_cast<TenT *>(&mpo[lsite_idx]);
        eff_ham[2] = const_cast<TenT *>(&mpo[rsite_idx]);
        eff_ham[two_site_eff_ham_size - 1] = new TenT();
        SlaveLanczosSolver<TenT>(eff_ham, comm);
      }
        break;
      case svd: {
        MPISVDSlave<TenElemT>(comm);
      }
        break;
      case contract_for_right_moving_expansion: {//dir='r'
        SlaveTwoSiteFiniteVMPSRightMovingExpand(eff_ham, comm);
      }
        break;
      case contract_for_left_moving_expansion: {//dir='l'
        SlaveTwoSiteFiniteVMPSLeftMovingExpand(eff_ham, comm);
      }
        break;
      case growing_left_env: {
        delete eff_ham[two_site_eff_ham_size - 1];
        SlaveGrowLeftEnvironment(*eff_ham[0], *eff_ham[1], comm);
        delete eff_ham[0];
      }
        break;
      case growing_right_env: {
        delete eff_ham[0];
        SlaveGrowRightEnvironment(*eff_ham[3], *eff_ham[2], comm);
        delete eff_ham[two_site_eff_ham_size - 1];
      }
        break;
      case program_final:std::cout << "Node" << rank << " will stop." << std::endl;
        break;
      default:std::cout << "Node " << rank << " cannot understand the order " << order << std::endl;
        break;
    }
  }

}

}

#endif //QLMPS_ALGO_MPI_VMPS_TWO_SITE_UPDATE_FINITE_VMPS_MPI_IMPL_SLAVE_H
