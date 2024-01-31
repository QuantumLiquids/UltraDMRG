// SPDX-License-Identifier: LGPL-3.0-only

/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2020-08-08 14:22
*
* Description: QuantumLiquids/UltraDMRG project. Lanczos params.
*/

/**
@file lanczos_params.h
@brief lanczos parameters used in VMPS, DMRG and TDVP
*/
#ifndef QLMPS_ALGORITHM_LANCZOS_PARAMS_H
#define QLMPS_ALGORITHM_LANCZOS_PARAMS_H


#include <stdlib.h>     // size_t


namespace qlmps {


/**
Parameters used by the Lanczos solver.
*/
struct LanczosParams {
  /**
  Setup Lanczos solver parameters.

  @param error The Lanczos tolerated error.
  @param max_iterations The maximal iteration times.
  */
  LanczosParams(double err, size_t max_iter) :
      error(err), max_iterations(max_iter) {}

  LanczosParams(double err) : LanczosParams(err, 200) {}

  LanczosParams(void) : LanczosParams(1.0E-7, 200) {}

  LanczosParams(const LanczosParams &lancz_params) :
      LanczosParams(lancz_params.error, lancz_params.max_iterations) {}

  double error;             ///< The Lanczos tolerated error.
  size_t max_iterations;    ///< The maximal iteration times.
};
} /* qlmps */


#endif // QLMPS_ALGORITHM_LANCZOS_PARAMS_H
