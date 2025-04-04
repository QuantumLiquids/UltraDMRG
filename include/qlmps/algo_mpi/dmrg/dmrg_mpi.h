// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Hao-Xin Wang <wanghaoxin1996@gmail.com>
* Creation Date: 2023-05-11
*
* Description: QuantumLiquids/MPS project. Two-site update finite size DMRG with MPI Parallelization
*/

#ifndef QLMPS_ALGO_MPI_DMRG_DMRG_MPI_H
#define QLMPS_ALGO_MPI_DMRG_DMRG_MPI_H

#include "qlten/qlten.h"
#include "dmrg_mpi_impl_master.h"
#include "dmrg_mpi_impl_slave.h"

namespace qlmps {
using namespace qlten;

template<typename TenElemT, typename QNT>
inline QLTEN_Double FiniteDMRG(
    FiniteMPS<TenElemT, QNT> &mps,
    const MatReprMPO<QLTensor<TenElemT, QNT>> &mat_repr_mpo,
    const FiniteVMPSSweepParams &sweep_params,
    const MPI_Comm &comm
) {
  QLTEN_Double e0(0.0);
  int mpi_size, rank;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &mpi_size);
  if (mpi_size == 1) {
    DMRGExecutor<TenElemT, QNT> dmrg_executor = DMRGExecutor(mat_repr_mpo, sweep_params);
    dmrg_executor.Execute();
    return dmrg_executor.GetEnergy();
  }

  if (rank == kMPIMasterRank) {
    DMRGMPIMasterExecutor<TenElemT, QNT> dmrg_executor = DMRGMPIMasterExecutor(mat_repr_mpo, sweep_params, comm);
    dmrg_executor.Execute();
    e0 = dmrg_executor.GetEnergy();
  } else {
    DMRGMPISlaveExecutor<TenElemT, QNT> dmrg_executor = DMRGMPISlaveExecutor(mat_repr_mpo, comm);
    dmrg_executor.Execute();
  }
  return e0;
}
}

#endif //QLMPS_ALGO_MPI_DMRG_DMRG_MPI_H
