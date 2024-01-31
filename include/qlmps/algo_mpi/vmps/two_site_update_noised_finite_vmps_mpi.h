// SPDX-License-Identifier: LGPL-3.0-only

/*
* Author: Hao-Xin Wang <wanghaoxin1996@gmail.com>
* Creation Date: 2021-08-31
*
* Description: QuantumLiquids/UltraDMRG project. Two-site update finite size vMPS with MPI Paralization
*/

/**
@file two_site_update_noised_finite_vmps_mpi.h
@brief Two-site update noised finite size vMPS with MPI Paralization
*/
#ifndef QLMPS_ALGO_MPI_VMPS_TWO_SITE_UPDATE_NOISED_FINITE_VMPS_MPI_H
#define QLMPS_ALGO_MPI_VMPS_TWO_SITE_UPDATE_NOISED_FINITE_VMPS_MPI_H

#include <stdlib.h>
#include "qlmps/algorithm/lanczos_params.h"                        //LanczosParams
#include "qlmps/algorithm/finite_vmps_sweep_params.h"      //TwoSiteMPIVMPSSweepParams


namespace qlmps {

//struct TwoSiteMPINoisedVMPSSweepParams : public TwoSiteMPIVMPSSweepParams {
//  TwoSiteMPINoisedVMPSSweepParams(
//    const size_t sweeps,
//    const size_t dmin, const size_t dmax, const double trunc_err,
//    const LanczosParams &lancz_params,
//    const std::vector<double> noises = std::vector<double>(1, 0.0),
//    const std::string mps_path = kMpsPath,
//    const std::string temp_path = kRuntimeTempPath
//  ) :
//    TwoSiteMPIVMPSSweepParams(sweeps, dmin, dmax, trunc_err,
//    lancz_params, mps_path, temp_path), noises(noises) {}
//public:
//  std::vector<double> noises;
//};

}//qlmps

#include "qlmps/algo_mpi/vmps/two_site_update_noised_finite_vmps_mpi_impl.h"

#endif