// SPDX-License-Identifier: LGPL-3.0-only

/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2020-08-08 16:45
*
* Description: QuantumLiquids/UltraDMRG project. Finite size VMPS and DMRG sweep parameters.
*/

/**
@file finite_vmps_sweep_params.h
@brief finite size VMPS and DMRG sweep parameters.
*/
#ifndef QLMPS_ALGORITHM_FINITE_VMPS_SWEEP_PARAMS_H
#define QLMPS_ALGORITHM_FINITE_VMPS_SWEEP_PARAMS_H


#include "qlmps/consts.h"                      // kMpsPath, kRuntimeTempPath
#include "qlmps/algorithm/lanczos_params.h"    // LanczParams

#include <string>     // string


namespace qlmps {

const double kVMPSMaxNoise = 1.0; //maximal noise
const double kVMPSNoiseIncreaseRate = 1.02;
const double kVMPSNoiseDecreaseRate = 0.95;
const double kVMPSNoiseAlpha = 0.3;

struct FiniteVMPSSweepParams {
  FiniteVMPSSweepParams(
      const size_t sweeps,
      const size_t dmin, const size_t dmax, const double trunc_err,
      const LanczosParams &lancz_params,
      const std::string mps_path = kMpsPath,
      const std::string temp_path = kRuntimeTempPath
  ) :
      sweeps(sweeps),
      Dmin(dmin), Dmax(dmax), trunc_err(trunc_err),
      lancz_params(lancz_params),
      mps_path(mps_path),
      temp_path(temp_path),
      noise_valid(false),
      noises({}),
      max_noise(0.0),
      noise_increase(0.0),
      noise_decrease(0.0),
      alpha(0.0) {}

  FiniteVMPSSweepParams(
      const size_t sweeps,
      const size_t dmin, const size_t dmax, const double trunc_err,
      const LanczosParams &lancz_params,
      const std::vector<double> noises,
      const double max_noise = kVMPSMaxNoise,
      const double noise_increase = kVMPSNoiseIncreaseRate,
      const double noise_decrease = kVMPSNoiseDecreaseRate,
      const double alpha = kVMPSNoiseAlpha,
      const std::string mps_path = kMpsPath,
      const std::string temp_path = kRuntimeTempPath
  ) : sweeps(sweeps),
      Dmin(dmin), Dmax(dmax), trunc_err(trunc_err),
      lancz_params(lancz_params),
      mps_path(mps_path),
      temp_path(temp_path),
      noises(noises),
      max_noise(max_noise),
      noise_increase(noise_increase),
      noise_decrease(noise_decrease),
      alpha(alpha) {
    if (noises.size() == 0) {
      noise_valid = false;
    } else {
      for (auto noise: noises) {
        if (std::abs(noise) > 0.0) {
          noise_valid = true;
          return;
        }
      }
      noise_valid = false;
    }
  }


  size_t sweeps;

  size_t Dmin;
  size_t Dmax;
  double trunc_err;

  LanczosParams lancz_params;

  // Advanced parameters
  /// MPS directory path
  std::string mps_path;

  /// Runtime temporary files directory path
  std::string temp_path;

  bool noise_valid;
  std::vector<double> noises;
  double max_noise;
  double noise_increase;
  double noise_decrease;
  double alpha;
};
} /* qlmps */


// Implementation details
#include "qlmps/algorithm/vmps/two_site_update_finite_vmps_impl.h"


#endif /* ifndef QLMPS_ALGORITHM_FINITE_VMPS_SWEEP_PARAMS_H */
