// SPDX-License-Identifier: LGPL-3.0-only

/*
* Author: Hao-Xin Wang <wanghaoxin1996@gmail.com>
* Creation Date: 2021-12-21
*
* Description: QuantumLiquids/UltraDMRG project. Two-site update finite size vMPS with MPI Paralization
*/

#ifndef QLMPS_ALGO_MPI_TDVP_TWO_SITE_UPDATE_FINITE_TDVP_MPI_H
#define QLMPS_ALGO_MPI_TDVP_TWO_SITE_UPDATE_FINITE_TDVP_MPI_H

#include "qlmps/algorithm/tdvp/tdvp_evolve_params.h"    //TDVPEvolveParams

namespace qlmps {

template <typename QNT>
struct MPITDVPSweepParams : public TDVPEvolveParams<QNT> {
  MPITDVPSweepParams() = default;
  MPITDVPSweepParams(
      const double tau, const size_t step,
      const size_t site_0,
      const QLTensor<QLTEN_Complex, QNT>& op0,
      const QLTensor<QLTEN_Complex, QNT>& inst0,
      const QLTensor<QLTEN_Complex, QNT>& op1,
      const QLTensor<QLTEN_Complex, QNT>& inst1,
      const double e0,
      const size_t dmin, const size_t dmax, const double trunc_err,
      const LanczosParams &lancz_params,
      const std::string mps_path = kMpsPath,
      const std::string initial_mps_path = "initial_" + kMpsPath,
      const std::string temp_path = kRuntimeTempPath,
      const std::string measure_temp_path = ".measure_temp"
      ) : TDVPEvolveParams<QNT>(
          tau, step, site_0, op0, inst0, op1, inst1, e0,
          dmin, dmax, trunc_err, lancz_params, mps_path, initial_mps_path,
          temp_path, measure_temp_path
          ) {}

};

}





#include "qlmps/algo_mpi/tdvp/two_site_update_finite_tdvp_mpi_impl_master.h"
#include "qlmps/algo_mpi/tdvp/two_site_update_finite_tdvp_mpi_impl_slave.h"
#endif //QLMPS_ALGO_MPI_TDVP_TWO_SITE_UPDATE_FINITE_TDVP_MPI_H