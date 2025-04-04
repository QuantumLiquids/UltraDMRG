// SPDX-License-Identifier: LGPL-3.0-only

/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2019-05-11 14:37
*
* Description: QuantumLiquids/UltraDMRG project. The main header file.
*/

/**
@file qlmps.h
@brief The main header file.
*/
#ifndef QLMPS_QLMPS_H
#define QLMPS_QLMPS_H

#define QLMPS_VERSION_MAJOR "0"
#define QLMPS_VERSION_MINOR "1"

#include "qlmps/case_params_parser.h"                              // CaseParamsParserBasic
#include "qlmps/site_vec.h"                                        // SiteVec
// MPS class and its initializations and measurements
#include "qlmps/one_dim_tn/mps_all.h"                              // MPS, ...
// MPO and its generator
#include "qlmps/one_dim_tn/mpo/mpo.h"                              // MPO
#include "qlmps/one_dim_tn/mpo/mpogen/mpogen.h"                    // MPOGenerator
#include "qlmps/one_dim_tn/mpo/mpogen/impo_gen.h"
// Algorithms
#include "qlmps/algorithm/vmps/vmps_all.h"                         // TwoSiteFiniteVMPS, SingleSiteFiniteVMPS
#include "qlmps/algorithm/tdvp/tdvp_evolve_params.h"               // TwoSiteFiniteTDVP
#include "qlmps/algorithm/dmrg/dmrg.h"                             // DMRG
// MPI Algorithms
#include "qlmps/algo_mpi/vmps/two_site_update_finite_vmps_mpi.h"   //TwoSiteFiniteVMPSWithNoise
#include "qlmps/algo_mpi/tdvp/two_site_update_finite_tdvp_mpi.h"
#include "qlmps/algo_mpi/dmrg/dmrg_mpi.h"

// Mock Algorithms
#include "qlmps/algorithm/vmps/two_site_update_noise_finite_vmps_impl.h"
#include "qlmps/algo_mpi/vmps/two_site_update_noised_finite_vmps_mpi.h" //TwoSiteFiniteVMPSWithNoise

// Model relevant
#include "qlmps/model_relevant/sites/model_site_base.h"
#include "qlmps/model_relevant/operators/operators_all.h"

#endif /* ifndef QLMPS_QLMPS_H */
