// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Hao-Xin Wang <wanghaoxin1996@gmail.com>
* Creation Date: 2021/11/1.
*
* Description: QuantumLiquids/UltraDMRG project. Implementation details for two site tdvp.
*/

#ifndef QLMPS_ALGORITHM_TDVP_TDVP_EVOLVE_PARAMS_H
#define QLMPS_ALGORITHM_TDVP_TDVP_EVOLVE_PARAMS_H

#include "qlmps/consts.h"                      // kMpsPath, kRuntimeTempPath
#include "qlmps/algorithm/lanczos_params.h"    // LanczParams
#include "qlmps/algorithm/finite_vmps_sweep_params.h" //FiniteVMPSSweepParams
#include <string>                               // string

namespace qlmps {
using namespace qlten;

template<typename QNT>
struct TDVPEvolveParams {
  TDVPEvolveParams() = default;
  TDVPEvolveParams(
      const double tau, const size_t step,
      const size_t site_0,
      const QLTensor<QLTEN_Complex, QNT> &op0,
      const QLTensor<QLTEN_Complex, QNT> &inst0,
      const QLTensor<QLTEN_Complex, QNT> &op1,
      const QLTensor<QLTEN_Complex, QNT> &inst1,
      const double e0,
      const size_t dmin, const size_t dmax, const double trunc_err,
      const LanczosParams &lancz_params,
      const std::string mps_path = kMpsPath,
      const std::string initial_mps_path = "initial_" + kMpsPath,
      const std::string temp_path = kRuntimeTempPath,
      const std::string measure_temp_path = ".measure_temp"
  ) : tau(tau), step(step), site_0(site_0),
      local_op0(op0), inst0(inst0),
      local_op1(op1), inst1(inst1),
      e0(e0),
      Dmin(dmin), Dmax(dmax), trunc_err(trunc_err),
      lancz_params(lancz_params),
      mps_path(mps_path),
      initial_mps_path(initial_mps_path),
      temp_path(temp_path),
      measure_temp_path(measure_temp_path) {}

  operator FiniteVMPSSweepParams() const {
    return FiniteVMPSSweepParams(
        step, Dmin, Dmax, trunc_err,
        lancz_params,
        mps_path, temp_path
    );
  }

  double tau;
  size_t step;
  size_t site_0;

  bool local_op_corr = true;
  QLTensor<QLTEN_Complex, QNT> local_op0;
  QLTensor<QLTEN_Complex, QNT> inst0;
  QLTensor<QLTEN_Complex, QNT> local_op1;
  QLTensor<QLTEN_Complex, QNT> inst1;

  double e0;  //energy value of ground state(initial state)

  size_t Dmin;
  size_t Dmax;
  double trunc_err;

  LanczosParams lancz_params;

  // Advanced parameters
  /// Evolution MPS directory path
  std::string mps_path;

  /// Initial state
  std::string initial_mps_path;

  /// Runtime temporary files directory path
  std::string temp_path;

  std::string measure_temp_path;
};

}//qlmps

#include "qlmps/algorithm/tdvp/two_site_update_finite_tdvp_impl.h"

#endif //QLMPS_ALGORITHM_TDVP_TWO_SITE_UPDATE_FINITE_TDVP_H
